import base64
import io
import asyncio
import aiohttp
from fastapi import FastAPI, UploadFile, File, HTTPException
from pdf2image import convert_from_bytes
import cv2
import numpy as np
from PIL import Image


app = FastAPI()

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "lightonai/LightOnOCR-1B-1025"


PREPROCESS=True

if PREPROCESS:
    CLAHE_CLIP_LIMIT = 0.2
    CLAHE_GRID_SIZE = (4, 4)
    SHARP_STRENGTH = 1.8
    SAVE_QUALITY = 100
    PDF_DPI = 400
else:
    CLAHE_CLIP_LIMIT = 1.0       
    CLAHE_GRID_SIZE = (8, 8)
    SHARP_STRENGTH = 1.0         
    PDF_DPI = 200
    SAVE_QUALITY = 100


def preprocess_image_for_ocr(pil_image):

    img = np.array(pil_image)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    edge = (1 - SHARP_STRENGTH) / 4
    kernel = np.array([
        [0, edge, 0],
        [edge, SHARP_STRENGTH, edge],
        [0, edge, 0]
    ])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)
    return Image.fromarray(sharpened)

async def process_page(session, image, page_num):

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{img_b64}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": "Extract text."}
                ]
            }
        ],
        "max_tokens": 4096
    }

    try:
        async with session.post(VLLM_URL, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"Error page {page_num}: {text}")
                return f"[Error processing page {page_num}]"
            
            result = await resp.json()

            return result['choices'][0]['message']['content']
    except Exception as e:
        return f"[Exception on page {page_num}: {str(e)}]"

@app.post("/ocr")
async def ocr_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    

    try:

        images = convert_from_bytes(content, dpi=200, fmt="jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")


    async with aiohttp.ClientSession() as session:

        tasks = []
        for i, img_raw in enumerate(images):

            processed_img = preprocess_image_for_ocr(img_raw)

            task = asyncio.create_task(process_page(session, processed_img, i+1))
            tasks.append(task)
        results = await asyncio.gather(*tasks)

    full_text = "\n\n--- Page Break ---\n\n".join(results)
    return {"filename": file.filename, "text": full_text}
