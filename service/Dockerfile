FROM vllm/vllm-openai:latest

USER root
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pdf2image \
    requests \
    aiohttp

RUN pip install opencv-python-headless

WORKDIR /app

COPY app.py .
COPY start.sh .

RUN chmod +x start.sh

ENTRYPOINT ["./start.sh"]