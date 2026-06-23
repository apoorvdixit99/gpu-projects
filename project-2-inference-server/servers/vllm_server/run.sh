#!/bin/bash
set -e

echo "Starting vLLM server for GPT-2 on port 8001..."
docker run --rm --gpus all \
    -p 8001:8000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    vllm/vllm-openai:v0.6.6.post1 \
    --model gpt2 \
    --dtype float16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.3
