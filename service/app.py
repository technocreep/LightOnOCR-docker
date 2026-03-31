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

PDF_DPI = max(72, int(os.getenv("PDF_DPI", "144")))
SAVE_QUALITY = max(40, min(95, int(os.getenv("SAVE_QUALITY", "80"))))
OCR_MAX_TOKENS = max(128, int(os.getenv("OCR_MAX_TOKENS", "1024")))
OCR_PROCESSING_MODE = os.getenv("OCR_PROCESSING_MODE", "sequential").strip().lower()
MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("MAX_CONCURRENT_REQUESTS", "1")))
OCR_CONVERT_WORKERS = max(1, int(os.getenv("OCR_CONVERT_WORKERS", "1")))
OCR_PAGE_PIPELINE_DEPTH = max(
    1,
    int(os.getenv("OCR_PAGE_PIPELINE_DEPTH", str(max(2, OCR_CONVERT_WORKERS + MAX_CONCURRENT_REQUESTS)))),
)
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

    b64_image = preprocess_image_for_ocr(image)
    image.close()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this page. Output only text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            }
        ],
        "max_tokens": OCR_MAX_TOKENS,
    }

    timeout = _build_client_timeout()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        result = await _post_to_vllm(session, payload)

    text = (result.get("choices") or [{}])[0].get("message", {}).get("content")
    return [{"index": 0, "markdown": text or ""}]


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
                b64_image = preprocess_image_for_ocr(img)
            print(f'[Page {page_num}] Image preprocessing took {time.time()-prep_start:.2f}s', flush=True)

            payload = {
                "model": MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "OCR this page. Output only text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }],
                "max_tokens": OCR_MAX_TOKENS
            }

            req_start = time.time()
            result = await _post_to_vllm(session, payload)
            elapsed = time.time() - start
            req_time = time.time() - req_start
            print(f'[Page {page_num}] ✓ OCR request took {req_time:.2f}s, total OCR: {elapsed:.2f}s', flush=True)
            return (page_num, result['choices'][0]['message']['content'])
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