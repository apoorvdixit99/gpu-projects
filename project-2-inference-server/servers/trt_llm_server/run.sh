#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="trtllm-gpt2"

if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building TRT-LLM image (first run only, ~5 min)..."
    docker build -t "$IMAGE" "$SCRIPT_DIR"
fi

echo "Starting TRT-LLM server on port 8004..."
echo "Note: first startup compiles GPT-2 TRT engines (~2-3 min)."
docker run --rm --gpus all \
    -p 8004:8000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v trtllm-engine-cache:/root/.cache/tensorrt_llm \
    "$IMAGE" \
    uvicorn server:app --host 0.0.0.0 --port 8000
