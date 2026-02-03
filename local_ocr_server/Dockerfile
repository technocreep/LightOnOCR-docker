# GPU-ready base image (CUDA 12.1 + cuDNN) with Python and PyTorch
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /srv

# System deps for Pillow/OpenCV runtime used by EasyOCR
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY app /srv/app

ENV HOST=0.0.0.0
ENV PORT=8080
ENV LOG_LEVEL=INFO

# Keep a single worker so the OCR model loads once (VRAM/RAM friendly)
ENV UVICORN_WORKERS=1

EXPOSE 8080

CMD ["bash", "-lc", "uvicorn app.main:app --host ${HOST} --port ${PORT} --workers ${UVICORN_WORKERS}"]
