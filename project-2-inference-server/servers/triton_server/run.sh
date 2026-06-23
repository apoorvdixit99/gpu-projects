#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_REPO="$SCRIPT_DIR/model_repo"
IMAGE="tritonserver-gpt2"

if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building custom Triton image (first run only, ~5 min)..."
    docker build -t "$IMAGE" "$SCRIPT_DIR"
fi

echo "Starting Triton Inference Server on port 8002..."
docker run --rm --gpus all \
    -p 8002:8000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" \
    tritonserver --model-repository=/models --log-verbose=0
