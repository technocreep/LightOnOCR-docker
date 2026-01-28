#!/bin/bash

python3 -m vllm.entrypoints.openai.api_server \
    --model lightonai/LightOnOCR-1B-1025 \
    --port 8000 \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.2 &

VLLM_PID=$!

echo "Waiting for vLLM to start..."

while ! curl -s localhost:8000/health > /dev/null; do
    sleep 5
    echo "Service loading..."
done
echo "vLLM is ready!"


uvicorn app:app --host 0.0.0.0 --port 8506