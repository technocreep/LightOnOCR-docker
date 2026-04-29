import sys
import os
import io
import asyncio
import base64
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import cv2
import numpy as np
import secrets
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header, Request
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("ocr-app")

app = FastAPI()

VLLM_URL = os.getenv("VLLM_URL", "http://vllm-server:8099/v1/chat/completions")
MODEL_NAME = "lightonai/LightOnOCR-1B-1025"
OUTPUT_DIR = Path("./output_texts")
OCR_EXTRACTION_PROMPT = (
    "Extract all visible text from the image, including tables, captions, headers, footers, "
    "and small text near images. Do not omit any text. Preserve reading order and table "
    "contents. Output only extracted text."
)

PDF_DPI = max(220, int(os.getenv("PDF_DPI", "144")))
SAVE_QUALITY = max(95, min(95, int(os.getenv("SAVE_QUALITY", "80"))))
OCR_MAX_TOKENS = max(7000, int(os.getenv("OCR_MAX_TOKENS", "1024")))
OCR_PROCESSING_MODE = os.getenv("OCR_PROCESSING_MODE", "sequential").strip().lower()
MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("MAX_CONCURRENT_REQUESTS", "1")))
OCR_CONVERT_WORKERS = max(1, int(os.getenv("OCR_CONVERT_WORKERS", "1")))
OCR_PAGE_PIPELINE_DEPTH = max(
    1,
    int(os.getenv("OCR_PAGE_PIPELINE_DEPTH", str(max(2, OCR_CONVERT_WORKERS + MAX_CONCURRENT_REQUESTS)))),
)
OCR_SIDEWAYS_PROJECTION_RATIO = float(os.getenv("OCR_SIDEWAYS_PROJECTION_RATIO", "1.2"))
OCR_SPREAD_CENTER_BAND_RATIO = float(os.getenv("OCR_SPREAD_CENTER_BAND_RATIO", "0.18"))
OCR_TEXT_MIN_DENSITY = float(os.getenv("OCR_TEXT_MIN_DENSITY", "0.015"))
OCR_SPLIT_GAP_RATIO = float(os.getenv("OCR_SPLIT_GAP_RATIO", "0.02"))
OCR_SEPARATOR_BAND_RATIO = float(os.getenv("OCR_SEPARATOR_BAND_RATIO", "0.025"))
OCR_SEPARATOR_NEIGHBOR_BAND_RATIO = float(os.getenv("OCR_SEPARATOR_NEIGHBOR_BAND_RATIO", "0.06"))
OCR_SEPARATOR_DENSITY_RATIO = float(os.getenv("OCR_SEPARATOR_DENSITY_RATIO", "0.6"))
OCR_SEPARATOR_TEXTURE_RATIO = float(os.getenv("OCR_SEPARATOR_TEXTURE_RATIO", "0.75"))
OCR_MODEL_CONNECT_TIMEOUT_SECONDS = float(os.getenv("OCR_MODEL_CONNECT_TIMEOUT_SECONDS", "120"))
OCR_MODEL_READ_TIMEOUT_SECONDS = float(os.getenv("OCR_MODEL_READ_TIMEOUT_SECONDS", "21600"))
OCR_MODEL_TOTAL_TIMEOUT_SECONDS = float(os.getenv("OCR_MODEL_TOTAL_TIMEOUT_SECONDS", "21600"))
VLLM_RETRY_ATTEMPTS = max(1, int(os.getenv("VLLM_RETRY_ATTEMPTS", "10")))
VLLM_RETRY_DELAY_SECONDS = float(os.getenv("VLLM_RETRY_DELAY_SECONDS", "5"))
VLLM_STARTUP_MAX_WAIT_SECONDS = float(os.getenv("VLLM_STARTUP_MAX_WAIT_SECONDS", "1800"))
VLLM_HEALTHCHECK_TIMEOUT_SECONDS = float(os.getenv("VLLM_HEALTHCHECK_TIMEOUT_SECONDS", "10"))

executor = ThreadPoolExecutor(max_workers=OCR_CONVERT_WORKERS)


def _build_client_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(
        total=OCR_MODEL_TOTAL_TIMEOUT_SECONDS,
        connect=OCR_MODEL_CONNECT_TIMEOUT_SECONDS,
        sock_connect=OCR_MODEL_CONNECT_TIMEOUT_SECONDS,
        sock_read=OCR_MODEL_READ_TIMEOUT_SECONDS,
    )


def _use_sequential_processing() -> bool:
    return OCR_PROCESSING_MODE != "parallel" or MAX_CONCURRENT_REQUESTS == 1


def _effective_pipeline_depth(total_pages: int) -> int:
    return max(1, min(total_pages, OCR_PAGE_PIPELINE_DEPTH))


def _build_vllm_health_url() -> str:
    parts = urlsplit(VLLM_URL)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def _clean_ocr_output(text: Optional[str]) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    prompt_variants = [
        OCR_EXTRACTION_PROMPT,
        f'"{OCR_EXTRACTION_PROMPT}"',
        f"'{OCR_EXTRACTION_PROMPT}'",
    ]

    for prompt_variant in prompt_variants:
        if cleaned.startswith(prompt_variant):
            cleaned = cleaned[len(prompt_variant):].lstrip("\n\r :-\t")

    return cleaned


def _build_text_mask(pil_image: Image.Image) -> np.ndarray:
    gray = np.array(pil_image.convert("L"))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _smooth_projection(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _projection_signal_strength(values: np.ndarray) -> float:
    smoothed = _smooth_projection(values, max(5, len(values) // 80))
    mean_value = float(np.mean(smoothed))
    if mean_value <= 1e-6:
        return 0.0
    return float(np.std(smoothed) / mean_value)


def _text_density(mask: np.ndarray) -> float:
    return float(np.mean(mask > 0))


def _ink_center_y(mask: np.ndarray) -> float:
    row_weights = (mask > 0).sum(axis=1).astype(np.float32)
    total = float(row_weights.sum())
    if total <= 1e-6:
        return 0.5
    positions = np.arange(mask.shape[0], dtype=np.float32)
    return float(np.dot(row_weights, positions) / total / max(1, mask.shape[0] - 1))


def _detect_vertical_separator(pil_image: Image.Image, mask: np.ndarray, split_x: int) -> tuple[bool, int]:
    width = mask.shape[1]
    separator_half = max(3, int(width * OCR_SEPARATOR_BAND_RATIO / 2.0))
    neighbor_width = max(separator_half + 2, int(width * OCR_SEPARATOR_NEIGHBOR_BAND_RATIO))

    separator_start = max(0, split_x - separator_half)
    separator_end = min(width, split_x + separator_half + 1)
    left_start = max(0, separator_start - neighbor_width)
    left_end = separator_start
    right_start = separator_end
    right_end = min(width, separator_end + neighbor_width)

    if separator_end - separator_start < 3 or left_end - left_start < 3 or right_end - right_start < 3:
        return False, split_x

    separator_mask = mask[:, separator_start:separator_end]
    left_mask = mask[:, left_start:left_end]
    right_mask = mask[:, right_start:right_end]

    separator_density = _text_density(separator_mask)
    left_density = _text_density(left_mask)
    right_density = _text_density(right_mask)
    neighbor_density = min(left_density, right_density)

    if neighbor_density <= 1e-6:
        return False, split_x

    gray = np.array(pil_image.convert("L"), dtype=np.float32)
    horizontal_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    separator_texture = float(np.mean(horizontal_gradient[:, separator_start:separator_end]))
    left_texture = float(np.mean(horizontal_gradient[:, left_start:left_end]))
    right_texture = float(np.mean(horizontal_gradient[:, right_start:right_end]))
    neighbor_texture = min(left_texture, right_texture)

    density_ratio = separator_density / neighbor_density
    texture_ratio = separator_texture / max(neighbor_texture, 1e-6)

    if density_ratio >= OCR_SEPARATOR_DENSITY_RATIO:
        return False, split_x
    if neighbor_texture > 1e-6 and texture_ratio >= OCR_SEPARATOR_TEXTURE_RATIO:
        return False, split_x

    column_density = np.mean(separator_mask > 0, axis=0)
    low_density_columns = np.where(column_density <= min(0.08, neighbor_density * OCR_SEPARATOR_DENSITY_RATIO))[0]
    if low_density_columns.size > 0:
        refined_x = separator_start + int(round(float(np.mean(low_density_columns))))
        return True, refined_x

    refined_x = separator_start + (separator_end - separator_start) // 2
    return True, refined_x


def _is_sideways_page(pil_image: Image.Image) -> bool:
    mask = _build_text_mask(pil_image)
    if _text_density(mask) < OCR_TEXT_MIN_DENSITY:
        return False

    row_signal = _projection_signal_strength(mask.sum(axis=1))
    col_signal = _projection_signal_strength(mask.sum(axis=0))
    return col_signal > row_signal * OCR_SIDEWAYS_PROJECTION_RATIO


def _normalize_sideways_page(pil_image: Image.Image) -> tuple[Image.Image, bool]:
    working_image = pil_image.copy()
    if not _is_sideways_page(working_image):
        return working_image, False

    clockwise = working_image.rotate(-90, expand=True)
    counter_clockwise = working_image.rotate(90, expand=True)

    clockwise_center = _ink_center_y(_build_text_mask(clockwise))
    counter_center = _ink_center_y(_build_text_mask(counter_clockwise))

    working_image.close()
    if clockwise_center <= counter_center:
        counter_clockwise.close()
        return clockwise, True

    clockwise.close()
    return counter_clockwise, True


def _detect_two_page_spread(pil_image: Image.Image) -> tuple[bool, Optional[int]]:
    width, _ = pil_image.size

    mask = _build_text_mask(pil_image)
    if _text_density(mask) < OCR_TEXT_MIN_DENSITY:
        return False, None

    half_width = width // 2
    left_density = _text_density(mask[:, :half_width])
    right_density = _text_density(mask[:, half_width:])
    if left_density < OCR_TEXT_MIN_DENSITY or right_density < OCR_TEXT_MIN_DENSITY:
        return False, None

    projection = mask.sum(axis=0).astype(np.float32)
    smooth_window = max(21, (width // 40) | 1)
    smoothed = _smooth_projection(projection, smooth_window)

    center_band_half = max(10, int(width * OCR_SPREAD_CENTER_BAND_RATIO / 2.0))
    center_start = max(0, half_width - center_band_half)
    center_end = min(width, half_width + center_band_half)
    center_band = smoothed[center_start:center_end]
    if center_band.size == 0:
        return False, None

    left_band = smoothed[int(width * 0.08): max(int(width * 0.45), int(width * 0.08) + 1)]
    right_band = smoothed[min(int(width * 0.55), width - 1): int(width * 0.92)]
    if left_band.size == 0 or right_band.size == 0:
        return False, None

    valley_index = center_start + int(np.argmin(center_band))
    valley_value = float(smoothed[valley_index])
    left_level = float(np.mean(left_band))
    right_level = float(np.mean(right_band))
    shoulder_level = min(left_level, right_level)

    if shoulder_level <= 1e-6:
        return False, None

    balance = abs(left_level - right_level) / max(left_level, right_level)
    center_ratio = valley_value / shoulder_level

    if center_ratio < 0.72 and balance < 0.6:
        has_separator, refined_split_x = _detect_vertical_separator(pil_image, mask, valley_index)
        if has_separator:
            return True, refined_split_x

    return False, None


def _split_two_page_spread(pil_image: Image.Image, split_x: int) -> List[Image.Image]:
    width, height = pil_image.size
    gap = max(2, int(width * OCR_SPLIT_GAP_RATIO))
    left_end = max(1, min(width - 1, split_x - gap))
    right_start = max(1, min(width - 1, split_x + gap))

    left_page = pil_image.crop((0, 0, left_end, height)).copy()
    right_page = pil_image.crop((right_start, 0, width, height)).copy()
    return [left_page, right_page]


def _prepare_page_images(pil_image: Image.Image) -> tuple[List[Image.Image], Dict[str, Any]]:
    working_image = pil_image.copy()
    rotated = False
    is_spread, split_x = _detect_two_page_spread(working_image)

    if is_spread and split_x is not None:
        split_images = _split_two_page_spread(working_image, split_x)
        working_image.close()
        return split_images, {"rotated": rotated, "spread": True, "split_x": split_x}

    return [working_image], {"rotated": rotated, "spread": False, "split_x": None}


async def _wait_for_vllm_ready() -> None:
    deadline = time.time() + VLLM_STARTUP_MAX_WAIT_SECONDS
    health_url = _build_vllm_health_url()
    timeout = aiohttp.ClientTimeout(
        total=VLLM_HEALTHCHECK_TIMEOUT_SECONDS,
        connect=VLLM_HEALTHCHECK_TIMEOUT_SECONDS,
        sock_connect=VLLM_HEALTHCHECK_TIMEOUT_SECONDS,
        sock_read=VLLM_HEALTHCHECK_TIMEOUT_SECONDS,
    )

    attempt = 0
    last_error = "unknown error"
    while time.time() < deadline:
        attempt += 1
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as resp:
                    if resp.status < 500:
                        logger.info("vLLM is ready at %s", health_url)
                        return
                    last_error = f"HTTP {resp.status}"
        except Exception as exc:
            last_error = str(exc)

        logger.warning(
            "Waiting for vLLM at %s (attempt %s, last error: %s)",
            health_url,
            attempt,
            last_error,
        )
        await asyncio.sleep(VLLM_RETRY_DELAY_SECONDS)

    raise RuntimeError(f"vLLM did not become ready at {health_url}: {last_error}")


async def _post_to_vllm(session: aiohttp.ClientSession, payload: Dict[str, Any]) -> Dict[str, Any]:
    last_exception: Exception | None = None
    last_status_error = ""

    for attempt in range(1, VLLM_RETRY_ATTEMPTS + 1):
        try:
            async with session.post(VLLM_URL, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()

                last_status_error = await resp.text()
                if resp.status not in {502, 503, 504} or attempt == VLLM_RETRY_ATTEMPTS:
                    raise HTTPException(status_code=502, detail=f"vLLM OCR failed: {last_status_error}")

                logger.warning(
                    "vLLM temporary HTTP error %s on attempt %s/%s: %s",
                    resp.status,
                    attempt,
                    VLLM_RETRY_ATTEMPTS,
                    last_status_error,
                )
        except HTTPException:
            raise
        except Exception as exc:
            last_exception = exc
            logger.warning(
                "vLLM connection attempt %s/%s failed: %s",
                attempt,
                VLLM_RETRY_ATTEMPTS,
                exc,
            )

        if attempt < VLLM_RETRY_ATTEMPTS:
            await asyncio.sleep(VLLM_RETRY_DELAY_SECONDS)

    if last_exception is not None:
        raise HTTPException(status_code=502, detail=f"vLLM connection failed after retries: {last_exception}")
    raise HTTPException(status_code=502, detail=f"vLLM OCR failed after retries: {last_status_error}")


# --- Mistral OCR API compatibility (OpenWebUI mistral_ocr loader) ---
_STORE_DIR = Path(os.getenv("OCR_STORE_DIR", "/tmp/local-ocr-store"))
_STORE_DIR.mkdir(parents=True, exist_ok=True)

# file_id -> metadata
_FILE_INDEX: Dict[str, Dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _file_path(file_id: str) -> Path:
    return _STORE_DIR / f"{file_id}.bin"


def _delete_file_local(file_id: str) -> None:
    _FILE_INDEX.pop(file_id, None)
    try:
        _file_path(file_id).unlink(missing_ok=True)
    except Exception:
        pass


def _prune_store() -> None:
    ttl_s = int(os.getenv("OCR_STORE_TTL_SECONDS", "3600"))
    max_files = int(os.getenv("OCR_STORE_MAX_FILES", "200"))

    cutoff = _now() - ttl_s
    expired = [fid for fid, meta in list(_FILE_INDEX.items()) if float(meta.get("created_at", 0)) < cutoff]
    for fid in expired:
        _delete_file_local(fid)

    if len(_FILE_INDEX) > max_files:
        ordered = sorted(_FILE_INDEX.items(), key=lambda kv: float(kv[1].get("created_at", 0)))
        for fid, _ in ordered[: max(0, len(_FILE_INDEX) - max_files)]:
            _delete_file_local(fid)


def _is_pdf(filename: Optional[str], content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False


async def _ocr_pdf_bytes_to_pages(content: bytes, doc_name: str, total_pages: int) -> List[Dict[str, Any]]:
    target_dir = OUTPUT_DIR / "mistral" / doc_name / "images"
    target_dir.mkdir(parents=True, exist_ok=True)

    timeout = _build_client_timeout()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await _run_pdf_page_pipeline(session, content, target_dir, total_pages)

    results.sort(key=lambda x: x[0])
    return [{"index": page_num - 1, "markdown": text} for page_num, text in results]


async def _run_pdf_page_pipeline(
    session: aiohttp.ClientSession,
    content: bytes,
    target_dir: Path,
    total_pages: int,
) -> List[tuple[int, str]]:
    semaphore = None if _use_sequential_processing() else asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    pipeline_depth = _effective_pipeline_depth(total_pages)

    in_flight: Dict[asyncio.Task, int] = {}
    results: List[tuple[int, str]] = []
    next_page = 1

    while next_page <= total_pages and len(in_flight) < pipeline_depth:
        task = asyncio.create_task(
            convert_and_ocr_page(session, semaphore, content, next_page, target_dir, total_pages)
        )
        in_flight[task] = next_page
        next_page += 1

    while in_flight:
        done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            page_num = in_flight.pop(task)
            try:
                results.append(await task)
            except Exception as e:
                logger.exception("Pipeline task failed for page %s", page_num)
                results.append((page_num, f"[Exception page {page_num}: {e}]"))

            if next_page <= total_pages:
                next_task = asyncio.create_task(
                    convert_and_ocr_page(session, semaphore, content, next_page, target_dir, total_pages)
                )
                in_flight[next_task] = next_page
                next_page += 1

    return results


async def _ocr_image_bytes_to_pages(content: bytes) -> List[Dict[str, Any]]:
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    timeout = _build_client_timeout()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        prepared_images, metadata = _prepare_page_images(image)
        image.close()
        pages: List[Dict[str, Any]] = []

        if metadata["spread"]:
            logger.info("Detected two-page spread in image upload; splitting at x=%s", metadata["split_x"])

        for index, prepared_image in enumerate(prepared_images):
            try:
                b64_image = preprocess_image_for_ocr(prepared_image)
                payload = {
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": OCR_EXTRACTION_PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                            ],
                        }
                    ],
                    "max_tokens": OCR_MAX_TOKENS,
                }
                result = await _post_to_vllm(session, payload)
                text = (result.get("choices") or [{}])[0].get("message", {}).get("content")
                pages.append({"index": index, "markdown": _clean_ocr_output(text)})
            finally:
                prepared_image.close()

    return pages


@app.post("/v1/files")
async def v1_files_create(
    authorization: Optional[str] = Header(default=None),
    purpose: str = Form(default="ocr"),
    file: UploadFile = File(...),
):
    """Minimal subset of Mistral's file upload API used by OpenWebUI."""
    # Auth intentionally not enforced here.
    if purpose and purpose != "ocr":
        raise HTTPException(status_code=400, detail="Unsupported purpose")

    _prune_store()
    file_id = secrets.token_urlsafe(18)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    _file_path(file_id).write_bytes(data)
    _FILE_INDEX[file_id] = {
        "id": file_id,
        "filename": file.filename,
        "content_type": file.content_type or "application/octet-stream",
        "size": len(data),
        "created_at": _now(),
    }

    return {"id": file_id, "object": "file", "filename": file.filename}


@app.get("/v1/files/{file_id}/url")
async def v1_files_url(
    file_id: str,
    authorization: Optional[str] = Header(default=None),
    expiry: Optional[int] = None,
):
    if file_id not in _FILE_INDEX:
        raise HTTPException(status_code=404, detail="File not found")
    return {"url": f"local://{file_id}"}


@app.delete("/v1/files/{file_id}")
async def v1_files_delete(
    file_id: str,
    authorization: Optional[str] = Header(default=None),
):
    existed = file_id in _FILE_INDEX
    _delete_file_local(file_id)
    return {"id": file_id, "deleted": existed}


@app.post("/v1/ocr")
async def v1_ocr(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    payload = await request.json()
    document = payload.get("document") or {}
    doc_url = document.get("document_url")
    if not doc_url or not isinstance(doc_url, str):
        raise HTTPException(status_code=400, detail="Missing document.document_url")

    if not doc_url.startswith("local://"):
        raise HTTPException(status_code=400, detail="Only local:// URLs are supported")

    file_id = doc_url[len("local://") :]
    meta = _FILE_INDEX.get(file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="File not found")

    content = _file_path(file_id).read_bytes()
    filename = meta.get("filename") or "uploaded"
    content_type = meta.get("content_type") or "application/octet-stream"

    t0 = time.time()
    if _is_pdf(filename, content_type):
        try:
            reader = PdfReader(io.BytesIO(content))
            total_pages = len(reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")

        pages = await _ocr_pdf_bytes_to_pages(content, doc_name=file_id, total_pages=total_pages)
    else:
        pages = await _ocr_image_bytes_to_pages(content)

    return JSONResponse(
        content={
            "pages": pages,
            "model": payload.get("model") or "local-ocr",
            "usage": {},
        },
        headers={"X-OCR-Ms": str(int((time.time() - t0) * 1000))},
    )

def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), SAVE_QUALITY])
    return base64.b64encode(buffer).decode('utf-8')

def convert_page_sync(content, page_num, target_dir):
    """Sync page conversion"""
    start = time.time()
    try:
        page_images = convert_from_bytes(
            content, dpi=PDF_DPI, fmt="jpeg",
            first_page=page_num, last_page=page_num
        )
        
        if page_images:
            img = page_images[0]
            img_path = target_dir / f'image_{page_num-1:04d}.jpeg'
            img.save(str(img_path), "JPEG", quality=SAVE_QUALITY)
            img.close()
            elapsed = time.time() - start
            print(f'[Page {page_num}] Conversion took {elapsed:.2f}s', flush=True)
            return img_path
        return None
    except Exception as e:
        logger.error(f"Error converting page {page_num}: {e}")
        return None

async def process_single_page(session, semaphore, img_path, page_num):
    """Send single page to vLLM"""
    async def _run_request():
        start = time.time()
        try:
            prep_start = time.time()
            with Image.open(img_path) as img:
                prepared_images, metadata = _prepare_page_images(img.convert("RGB"))

            print(f'[Page {page_num}] Image preprocessing took {time.time()-prep_start:.2f}s', flush=True)
            if metadata["spread"]:
                print(f'[Page {page_num}] ⇆ Detected two-page spread, splitting at x={metadata["split_x"]}', flush=True)

            req_start = time.time()
            segment_texts: List[str] = []
            for segment_index, prepared_image in enumerate(prepared_images, start=1):
                try:
                    b64_image = preprocess_image_for_ocr(prepared_image)
                    payload = {
                        "model": MODEL_NAME,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": OCR_EXTRACTION_PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                            ]
                        }],
                        "max_tokens": OCR_MAX_TOKENS
                    }
                    result = await _post_to_vllm(session, payload)
                    segment_text = _clean_ocr_output(
                        (result.get("choices") or [{}])[0].get("message", {}).get("content")
                    )
                    segment_texts.append(segment_text)
                    if metadata["spread"]:
                        print(f'[Page {page_num}] Segment {segment_index}/{len(prepared_images)} OCR complete', flush=True)
                finally:
                    prepared_image.close()

            elapsed = time.time() - start
            req_time = time.time() - req_start
            print(f'[Page {page_num}] ✓ OCR request took {req_time:.2f}s, total OCR: {elapsed:.2f}s', flush=True)
            return (page_num, "\n\n".join([text for text in segment_texts if text.strip()]))
        except Exception as e:
            elapsed = time.time() - start
            print(f'[Page {page_num}] ✗ Exception after {elapsed:.2f}s: {str(e)}', flush=True)
            return (page_num, f"[Exception page {page_num}: {str(e)}]")

    if semaphore is None:
        return await _run_request()

    async with semaphore:
        return await _run_request()

async def convert_and_ocr_page(session, semaphore, content, page_num, target_dir, total_pages):
    """Convert page then sand to OCR"""
    page_start = time.time()
    loop = asyncio.get_running_loop()
    
    print(f'[Page {page_num}/{total_pages}] ⏳ Starting conversion...', flush=True)
    conv_start = time.time()
    img_path = await loop.run_in_executor(
        executor, 
        convert_page_sync, 
        content, 
        page_num, 
        target_dir
    )
    conv_time = time.time() - conv_start
    
    if img_path is None:
        return (page_num, f"[Error: Failed to convert page {page_num}]")
    
    # 2. OCR
    print(f'[Page {page_num}/{total_pages}] 🚀 Converted in {conv_time:.2f}s, sending to OCR...', flush=True)
    ocr_start = time.time()
    result = await process_single_page(session, semaphore, img_path, page_num)
    ocr_time = time.time() - ocr_start
    
    total_time = time.time() - page_start
    print(f'[Page {page_num}/{total_pages}] ✅ DONE in {total_time:.2f}s (conv: {conv_time:.2f}s, ocr: {ocr_time:.2f}s)', flush=True)
    return result

@app.post("/ocr")
async def ocr_pdf(file: UploadFile = File(...)):
    overall_start = time.time()
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    filename_stem = Path(file.filename).stem
    target_dir = OUTPUT_DIR / filename_stem / "images"
    target_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(io.BytesIO(content))
    total_pages = len(reader.pages)
    
    print(f'\n{"="*80}')
    print(f'📄 PDF: {file.filename} | Pages: {total_pages}')
    mode_label = "sequential" if _use_sequential_processing() else f"parallel ({MAX_CONCURRENT_REQUESTS})"
    print(f'🔧 OCR request mode: {mode_label}')
    print(f'📦 Page pipeline depth: {_effective_pipeline_depth(total_pages)}')
    print(f'{"="*80}\n', flush=True)

    timeout = _build_client_timeout()

    tasks_start = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        mode_name = "bounded sequential" if _use_sequential_processing() else f"bounded parallel ({MAX_CONCURRENT_REQUESTS})"
        print(
            f'⚡ Starting {mode_name} processing for {total_pages} pages with pipeline depth {_effective_pipeline_depth(total_pages)}...\n',
            flush=True,
        )
        gather_start = time.time()
        results = await _run_pdf_page_pipeline(session, content, target_dir, total_pages)
        gather_time = time.time() - gather_start

    results.sort(key=lambda x: x[0])
    full_text = "\n\n--- Page Break ---\n\n".join([text for _, text in results])
    
    result_path = OUTPUT_DIR / filename_stem / "result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    total_time = time.time() - overall_start
    print(f'\n{"="*80}')
    print(f'✅ ALL DONE in {total_time:.2f}s')
    print(f'   - Parallel execution: {gather_time:.2f}s')
    print(f'   - Average per page: {total_time/total_pages:.2f}s')
    print(f'{"="*80}\n', flush=True)
    
    return {"filename": file.filename, "text": full_text, "processing_time": total_time}


@app.on_event("startup")
async def startup_event():
    await _wait_for_vllm_ready()

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)