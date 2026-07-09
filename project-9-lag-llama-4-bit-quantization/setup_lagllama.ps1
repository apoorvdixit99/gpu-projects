# Project 9 - Lag-Llama 4-bit Quantization: one-time setup
# Run from: project-9-lag-llama-4-bit-quantization/   (this directory)
#
# Uses its OWN venv (../.venv3), not the shared ../.venv. gluonts<=0.14.4 (required
# by the lag-llama repo's estimator API) pins numpy~=1.16 / pandas<2.2, which
# conflicts with the newer numpy/pandas pinned in the shared environment used by
# Projects 1 and 3-8. Same reasoning as Project 2's separate .venv2.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== Creating project-local virtual environment (.venv3) ===" -ForegroundColor Cyan
if (-not (Test-Path "..\.venv3")) {
    py -3.11 -m venv ..\.venv3
} else {
    Write-Host "..\.venv3 already exists, skipping creation."
}

Write-Host "=== Activating .venv3 ===" -ForegroundColor Cyan
..\.venv3\Scripts\Activate.ps1

Write-Host "=== Installing PyTorch (CUDA 12.6) ===" -ForegroundColor Cyan
pip install torch --index-url https://download.pytorch.org/whl/cu126

Write-Host "=== Installing project dependencies ===" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "=== Cloning lag-llama repo (vendored, not a git submodule) ===" -ForegroundColor Cyan
if (-not (Test-Path "vendor\lag-llama\.git")) {
    git clone https://github.com/time-series-foundation-models/lag-llama.git vendor\lag-llama
} else {
    Write-Host "vendor\lag-llama already present, skipping clone."
}

Write-Host "=== Downloading pretrained checkpoint (lag-llama.ckpt) ===" -ForegroundColor Cyan
if (-not (Test-Path "checkpoints\lag-llama.ckpt")) {
    python -c "from huggingface_hub import hf_hub_download; import shutil; p = hf_hub_download(repo_id='time-series-foundation-models/Lag-Llama', filename='lag-llama.ckpt'); shutil.copy(p, 'checkpoints/lag-llama.ckpt')"
} else {
    Write-Host "checkpoints\lag-llama.ckpt already present, skipping download."
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "To activate later:  ..\.venv3\Scripts\Activate.ps1"
Write-Host "Quick-start:"
Write-Host "  python src/run_benchmark.py --help"
