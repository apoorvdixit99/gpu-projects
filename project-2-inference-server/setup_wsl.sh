#!/bin/bash
set -e

echo "=== Project 2: Inference Server Setup ==="

VENV_DIR="/mnt/c/Users/apoor/Desktop/projects/Nvidia/.venv2"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

echo "[1/4] Installing system Python 3.12..."
sudo apt-get update -q
sudo apt-get install -y python3.12 python3.12-venv python3-pip

echo "[2/4] Creating virtual environment at $VENV_DIR..."
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/4] Installing PyTorch (cu124) before other packages..."
# Pin torch to cu124 first — avoids pip pulling the default cu130 build
# when resolving downstream dependencies.
pip install --upgrade pip
pip install \
    "torch==2.6.0+cu124" \
    "torchaudio==2.6.0+cu124" \
    --extra-index-url "$TORCH_INDEX"

echo "[4/4] Installing remaining requirements..."
# requirements.txt is a full pip freeze; torch is already satisfied above.
pip install -r requirements.txt --extra-index-url "$TORCH_INDEX"

echo ""
echo "=== Setup complete ==="
echo "Note: vLLM, SGLang, and TRT-LLM run via Docker — no pip install needed."
echo ""
echo "Activate env:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run servers (one at a time, each in its own terminal):"
echo "  FastAPI  : uvicorn servers.fastapi_hf.server:app --port 8000"
echo "  vLLM     : bash servers/vllm_server/run.sh"
echo "  Triton   : bash servers/triton_server/run.sh"
echo "  SGLang   : bash servers/sglang_server/run.sh"
echo "  TRT-LLM  : bash servers/trt_llm_server/run.sh"
echo ""
echo "Run benchmark:"
echo "  python benchmark/benchmark.py --server fastapi"
echo "  python benchmark/benchmark.py --server vllm"
echo "  python benchmark/benchmark.py --server triton"
echo "  python benchmark/benchmark.py --server sglang"
echo "  python benchmark/benchmark.py --server trtllm"
