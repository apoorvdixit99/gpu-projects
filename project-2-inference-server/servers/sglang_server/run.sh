#!/bin/bash
set -e

echo "Starting SGLang server for GPT-2 on port 8003..."
docker run --rm --gpus all \
    -p 8003:30000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    lmsysorg/sglang:v0.4.6.post1-cu124 \
    python3 -m sglang.launch_server \
    --model-path gpt2 \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype float16
