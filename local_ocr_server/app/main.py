import os
import time
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

log = logging.getLogger("local_ocr_server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


app = FastAPI(title="Local OCR Server", version="0.1.0")


_STORE_DIR = Path(os.getenv("OCR_STORE_DIR", "/tmp/local-ocr-store"))
_STORE_DIR.mkdir(parents=True, exist_ok=True)

# file_id -> metadata
_FILE_INDEX: Dict[str, Dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _prune_store() -> None:
    """Best-effort pruning of old uploads to keep disk usage bounded."""
    ttl_s = int(os.getenv("OCR_STORE_TTL_SECONDS", "3600"))
    max_files = int(os.getenv("OCR_STORE_MAX_FILES", "200"))

    # Remove expired
    expired: List[str] = []
    cutoff = _now() - ttl_s
    for file_id, meta in list(_FILE_INDEX.items()):
        if float(meta.get("created_at", 0)) < cutoff:
            expired.append(file_id)

    for file_id in expired:
        _delete_file_local(file_id)

    # Enforce max files (oldest first)
    if len(_FILE_INDEX) > max_files:
        ordered = sorted(_FILE_INDEX.items(), key=lambda kv: float(kv[1].get("created_at", 0)))
        for file_id, _ in ordered[: max(0, len(_FILE_INDEX) - max_files)]:
            _delete_file_local(file_id)


def _file_path(file_id: str) -> Path:
    return _STORE_DIR / f"{file_id}.bin"


def _delete_file_local(file_id: str) -> None:
    meta = _FILE_INDEX.pop(file_id, None)
    try:
        _file_path(file_id).unlink(missing_ok=True)
    except Exception:
        pass


def _langs() -> List[str]:
    langs_env = os.getenv("OCR_LANGS", "en")
    return [x.strip() for x in langs_env.split(",") if x.strip()]


def _dpi() -> int:
    return int(os.getenv("PDF_RENDER_DPI", "200"))


def _require_api_key(authorization: Optional[str]) -> None:
    # Auth disabled by request: always allow.
    return


def _is_pdf(filename: Optional[str], content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False


def _get_reader(langs: List[str]):
    # Lazy import to keep startup fast and make health checks cheap.
    import torch
    import easyocr

    want_gpu = os.getenv("OCR_GPU", "auto").lower()
    if want_gpu == "auto":
        use_gpu = bool(torch.cuda.is_available())
    else:
        use_gpu = want_gpu in ("1", "true", "yes", "on")

    # Important: keep a single Reader instance to avoid double VRAM usage.
    # Cache keyed by language list.
    if not hasattr(_get_reader, "_cache"):
        _get_reader._cache = {}

    key = (tuple(langs), use_gpu)
    if key not in _get_reader._cache:
        log.info("Initializing EasyOCR Reader (langs=%s, gpu=%s)", langs, use_gpu)
        _get_reader._cache[key] = (easyocr.Reader(langs, gpu=use_gpu), use_gpu)

    return _get_reader._cache[key]


def _ocr_image_pil(image, langs: List[str]) -> Dict[str, Any]:
    import numpy as np

    reader, use_gpu = _get_reader(langs)
    arr = np.array(image)
    # paragraph=True tends to produce nicer reading order.
    lines = reader.readtext(arr, detail=0, paragraph=True)
    text = "\n".join([l for l in lines if isinstance(l, str) and l.strip()])
    return {"text": text, "gpu": use_gpu}


def _extract_pdf_text_or_ocr(pdf_bytes: bytes, langs: List[str], dpi: int) -> Dict[str, Any]:
    import fitz  # PyMuPDF
    from PIL import Image
    import io

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[Dict[str, Any]] = []

    used_gpu_any = False
    used_ocr_any = False

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        extracted = (page.get_text("text") or "").strip()

        # Heuristic: if PDF has meaningful embedded text, prefer it.
        if len(extracted) >= int(os.getenv("PDF_MIN_TEXT_CHARS", "40")):
            page_text = extracted
            used_ocr = False
            used_gpu = False
        else:
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            ocr = _ocr_image_pil(image, langs=langs)
            page_text = ocr["text"].strip()
            used_gpu = bool(ocr["gpu"])
            used_ocr = True

        used_gpu_any = used_gpu_any or used_gpu
        used_ocr_any = used_ocr_any or used_ocr

        pages.append(
            {
                "page": page_index + 1,
                "text": page_text,
                "used_ocr": used_ocr,
                "used_gpu": used_gpu,
            }
        )

    return {
        "pages": pages,
        "used_gpu": used_gpu_any,
        "used_ocr": used_ocr_any,
        "page_count": doc.page_count,
    }


# --- Mistral OCR API compatibility (OpenWebUI mistral_ocr loader) ---


@app.post("/v1/files")
async def v1_files_create(
    authorization: Optional[str] = Header(default=None),
    purpose: str = Form(default="ocr"),
    file: UploadFile = File(...),
):
    """Minimal subset of Mistral's file upload API used by OpenWebUI."""
    _require_api_key(authorization)

    if purpose and purpose != "ocr":
        # OpenWebUI uses purpose=ocr
        raise HTTPException(status_code=400, detail="Unsupported purpose")

    _prune_store()
    file_id = uuid.uuid4().hex
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    path = _file_path(file_id)
    path.write_bytes(data)

    _FILE_INDEX[file_id] = {
        "id": file_id,
        "filename": file.filename,
        "content_type": file.content_type or "application/octet-stream",
        "size": len(data),
        "created_at": _now(),
    }

    # OpenWebUI's MistralLoader only needs the "id" field.
    return {"id": file_id, "object": "file", "filename": file.filename}


@app.get("/v1/files/{file_id}/url")
async def v1_files_url(
    file_id: str,
    authorization: Optional[str] = Header(default=None),
    expiry: Optional[int] = None,
):
    """Return a signed URL. We return a local scheme URL consumed by our /v1/ocr."""
    _require_api_key(authorization)

    if file_id not in _FILE_INDEX:
        raise HTTPException(status_code=404, detail="File not found")

    # MistralLoader treats this as opaque.
    return {"url": f"local://{file_id}"}


@app.delete("/v1/files/{file_id}")
async def v1_files_delete(
    file_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_api_key(authorization)
    existed = file_id in _FILE_INDEX
    _delete_file_local(file_id)
    return {"id": file_id, "deleted": existed}


@app.post("/v1/ocr")
async def v1_ocr(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Minimal subset of Mistral OCR API used by OpenWebUI.

    Expected payload:
    {
      "model": "mistral-ocr-latest",
      "document": {"type": "document_url", "document_url": "local://<id>"},
      "include_image_base64": false
    }
    """
    _require_api_key(authorization)
    payload = await request.json()
    document = payload.get("document") or {}
    doc_url = document.get("document_url")
    if not doc_url or not isinstance(doc_url, str):
        raise HTTPException(status_code=400, detail="Missing document.document_url")

    if not doc_url.startswith("local://"):
        raise HTTPException(
            status_code=400,
            detail="Only local:// URLs are supported by this local OCR server",
        )

    file_id = doc_url[len("local://") :]
    meta = _FILE_INDEX.get(file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="File not found")

    data = _file_path(file_id).read_bytes()
    filename = meta.get("filename") or "uploaded"
    content_type = meta.get("content_type") or "application/octet-stream"

    langs = _langs()
    dpi = _dpi()

    t0 = time.time()
    try:
        pages: List[Dict[str, Any]] = []
        if _is_pdf(filename, content_type):
            result = _extract_pdf_text_or_ocr(data, langs=langs, dpi=dpi)
            for idx, p in enumerate(result["pages"]):
                pages.append(
                    {
                        "index": idx,  # 0-based, as expected by OpenWebUI's MistralLoader
                        "markdown": (p.get("text") or "").strip(),
                    }
                )
        else:
            # Treat as image
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(data)).convert("RGB")
            ocr = _ocr_image_pil(image, langs=langs)
            pages.append({"index": 0, "markdown": (ocr.get("text") or "").strip()})

        return JSONResponse(
            content={
                "pages": pages,
                "model": payload.get("model") or "local-ocr",
                "usage": {},
            },
            headers={"X-OCR-Ms": str(int((time.time() - t0) * 1000))},
        )
    except Exception as e:
        log.exception("/v1/ocr failed")
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
