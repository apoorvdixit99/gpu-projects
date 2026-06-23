# Nvidia Portfolio - Shared Environment Setup
# Run from: Nvidia/   (this directory)
# Requires: Python 3.11, CUDA 12.6

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== Creating shared virtual environment ===" -ForegroundColor Cyan
py -3.11 -m venv .venv

Write-Host "=== Activating virtual environment ===" -ForegroundColor Cyan
.venv\Scripts\Activate.ps1

Write-Host "=== Upgrading pip ===" -ForegroundColor Cyan
pip install --upgrade pip

Write-Host "=== Installing PyTorch (CUDA 12.6) ===" -ForegroundColor Cyan
pip install torch --index-url https://download.pytorch.org/whl/cu126

Write-Host "=== Installing TensorRT (CUDA 12, from NVIDIA PyPI) ===" -ForegroundColor Cyan
pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com

Write-Host "=== Installing project dependencies ===" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "To activate later:  .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Quick-start (Project 1 - LLM Inference Optimization):"
Write-Host "  cd project-1-llm-inference-optimization"
Write-Host "  python src/run_benchmark.py --export            # export models + run all backends"
Write-Host "  python src/run_benchmark.py --backends pytorch  # PyTorch only (no export needed)"
Write-Host "  python src/run_benchmark.py --help              # see all options"
