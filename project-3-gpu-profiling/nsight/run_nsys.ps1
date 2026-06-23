<#
.SYNOPSIS
    Profile GPT-2 inference with NVIDIA Nsight Systems.

.DESCRIPTION
    Launches nsys to collect a hardware-level timeline: CUDA kernel start/stop
    times, NVTX range markers, CPU OS runtime events, and memory transfer events.

    Output file: results/nsight/gpt2_nsys_bs<N>_seq<N>.nsys-rep
    Open in   : NVIDIA Nsight Systems desktop application

    NVTX ranges in the trace:
      gpt2_inference  — outer range covering all profiled iterations
      forward_N       — per-iteration range; look for GPU idle gaps between kernels

.EXAMPLE
    .\nsight\run_nsys.ps1
    .\nsight\run_nsys.ps1 -BatchSize 8 -SeqLen 256 -Iterations 20
#>

param (
    [int]$BatchSize  = 1,
    [int]$SeqLen     = 128,
    [int]$Iterations = 20,
    [int]$Warmup     = 10
)

$OutputDir = Join-Path $PSScriptRoot "..\results\nsight"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$ReportBase = Join-Path $OutputDir "gpt2_nsys_bs${BatchSize}_seq${SeqLen}"

Write-Host "=== Nsight Systems Profile ===" -ForegroundColor Green
Write-Host "  Batch size : $BatchSize"
Write-Host "  Seq length : $SeqLen"
Write-Host "  Iterations : $Iterations  (warmup: $Warmup, outside NVTX range)"
Write-Host "  Output     : $ReportBase.nsys-rep"
Write-Host ""

if (-not (Get-Command nsys -ErrorAction SilentlyContinue)) {
    Write-Error ("nsys not found on PATH.`n" +
                 "Add the Nsight Systems install directory to PATH, e.g.:`n" +
                 "  C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.x.x\target-windows-x64")
    exit 1
}

nsys profile `
    --trace=cuda,nvtx `
    --output="$ReportBase" `
    --force-overwrite=true `
    python src/profile_nsys_target.py `
        --batch-size $BatchSize `
        --seq-len    $SeqLen `
        --warmup     $Warmup `
        --iterations $Iterations

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done." -ForegroundColor Green
    Write-Host "Open $ReportBase.nsys-rep in the Nsight Systems desktop app."
    Write-Host ""
    Write-Host "What to look for in the timeline:"
    Write-Host "  NVTX row      : gpt2_inference and forward_N ranges"
    Write-Host "  CUDA kernels  : bars inside each forward_N range; zoom in to see gaps"
    Write-Host "  Idle gaps     : space between kernel bars = CPU dispatch latency"
    Write-Host "  Memory xfers  : HtoD/DtoH events (should be absent at steady state)"
} else {
    Write-Error "nsys exited with code $LASTEXITCODE."
}
