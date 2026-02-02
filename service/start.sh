#!/bin/bash

# 1. Запускаем vLLM сервер в фоне.
# --limit-mm-per-prompt image=1 обязателен для VLM в текущих версиях vLLM
# --gpu-memory-utilization 0.7 оставляет немного памяти для обработки картинок в Python
python3 -m vllm.entrypoints.openai.api_server \
    --model lightonai/LightOnOCR-1B-1025 \
    --port 8000 \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.2 &

# Сохраняем PID процесса vLLM
VLLM_PID=$!

echo "Waiting for vLLM to start..."
# Ожидание пока порт 8000 станет доступен
while ! curl -s localhost:8000/health > /dev/null; do
    sleep 5
    echo "Service loading..."
done
echo "vLLM is ready!"

# 2. Запускаем FastAPI wrapper на порту 8506
uvicorn app:app --host 0.0.0.0 --port 8506