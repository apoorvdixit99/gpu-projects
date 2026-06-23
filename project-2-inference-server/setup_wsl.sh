#!/bin/bash
set -e

echo "=== Project 2: Inference Server Setup ==="

VENV_DIR="/mnt/c/Users/apoor/Desktop/projects/Nvidia/.venv2"

echo "[1/4] Installing system Python 3.12..."
sudo apt-get update -q
sudo apt-get install -y python3.12 python3.12-venv python3-pip

echo "[2/4] Creating virtual environment at $VENV_DIR..."
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/4] Installing base requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Installing vLLM (this takes a few minutes)..."
pip install vllm

echo ""
echo "=== Setup complete ==="
echo "Activate env: source /mnt/c/Users/apoor/Desktop/projects/Nvidia/.venv2/bin/activate"
echo ""
echo "Run servers (each in a separate terminal):"
echo "  FastAPI : uvicorn servers.fastapi_hf.server:app --port 8000"
echo "  vLLM    : bash servers/vllm_server/run.sh"
echo "  Triton  : bash servers/triton_server/run.sh"
echo ""
echo "Run benchmark:"
echo "  python benchmark/benchmark.py --server all"
