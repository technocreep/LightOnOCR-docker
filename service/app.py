import sys
import os
import io
import asyncio
import base64
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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

VLLM_URL = os.getenv("VLLM_URL", "http://vllm-server:8001/v1/chat/completions")
MODEL_NAME = "lightonai/LightOnOCR-1B-1025"
OUTPUT_DIR = Path("./output_texts")

PDF_DPI = 200
SAVE_QUALITY = 90
MAX_CONCURRENT_REQUESTS = 8

executor = ThreadPoolExecutor(max_workers=8)


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

    timeout = aiohttp.ClientTimeout(total=3600)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            convert_and_ocr_page(session, semaphore, content, page_num, target_dir, total_pages)
            for page_num in range(1, total_pages + 1)
        ]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: x[0])
    return [{"index": page_num - 1, "markdown": text} for page_num, text in results]


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
        "max_tokens": 2048,
    }

    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(VLLM_URL, json=payload) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                raise HTTPException(status_code=502, detail=f"vLLM OCR failed: {err_text}")
            result = await resp.json()

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
    async with semaphore:
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
                "max_tokens": 2048
            }

            req_start = time.time()
            async with session.post(VLLM_URL, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    elapsed = time.time() - start
                    req_time = time.time() - req_start
                    print(f'[Page {page_num}] ✓ OCR request took {req_time:.2f}s, total OCR: {elapsed:.2f}s', flush=True)
                    return (page_num, result['choices'][0]['message']['content'])
                else:
                    err_text = await resp.text()
                    return (page_num, f"[Error page {page_num}: {err_text}]")
        except Exception as e:
            elapsed = time.time() - start
            print(f'[Page {page_num}] ✗ Exception after {elapsed:.2f}s: {str(e)}', flush=True)
            return (page_num, f"[Exception page {page_num}: {str(e)}]")

async def convert_and_ocr_page(session, semaphore, content, page_num, target_dir, total_pages):
    """Convert page then sand to OCR"""
    page_start = time.time()
    loop = asyncio.get_event_loop()
    
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
    print(f'🔧 Max concurrent OCR requests: {MAX_CONCURRENT_REQUESTS}')
    print(f'{"="*80}\n', flush=True)

    timeout = aiohttp.ClientTimeout(total=3600)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    tasks_start = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            convert_and_ocr_page(session, semaphore, content, page_num, target_dir, total_pages)
            for page_num in range(1, total_pages + 1)
        ]
        
        print(f'⚡ All {len(tasks)} tasks created in {time.time()-tasks_start:.2f}s, starting parallel execution...\n', flush=True)
        
        gather_start = time.time()
        results = await asyncio.gather(*tasks)
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

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)