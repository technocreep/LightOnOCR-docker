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
from fastapi import FastAPI, UploadFile, File, HTTPException
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("ocr-app")

app = FastAPI()

VLLM_URL = os.getenv("VLLM_URL", "http://vllm-server:8000/v1/chat/completions")
MODEL_NAME = "lightonai/LightOnOCR-1B-1025"
OUTPUT_DIR = Path("./output_texts")

PDF_DPI = 200
SAVE_QUALITY = 90
MAX_CONCURRENT_REQUESTS = 8

executor = ThreadPoolExecutor(max_workers=8)

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