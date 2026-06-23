<#
.SYNOPSIS
    Profile a custom CUDA kernel with NVIDIA Nsight Systems.

.DESCRIPTION
    Launches nsys to collect a hardware-level timeline: CUDA kernel start/stop
    times, NVTX range markers, and CPU/GPU interleaving.

    Output file: results/nsight/cuda_kernels_nsys_<Kernel>_N<Size>.nsys-rep
    Open in   : NVIDIA Nsight Systems desktop application

    NVTX ranges in the trace:
      cuda_kernel_opt_<kernel>  outer range: all profiled iterations
      iter_N                    per-iteration; GPU idle gaps = kernel launch overhead

.EXAMPLE
    .\nsight\run_nsys.ps1
    .\nsight\run_nsys.ps1 -Kernel matmul_tiled -Size 1024 -Iterations 20
    .\nsight\run_nsys.ps1 -Kernel reduce_shuffle -Size 16 -Iterations 50
    .\nsight\run_nsys.ps1 -Kernel vec_add_opt -Size 64 -Iterations 30
#>

param (
    [ValidateSet("vec_add_naive","vec_add_opt",
                 "matmul_naive","matmul_tiled",
                 "reduce_naive","reduce_sequential","reduce_shuffle")]
    [string]$Kernel     = "matmul_tiled",
    [int]   $Size       = 1024,
    [int]   $Iterations = 20,
    [int]   $Warmup     = 5
)

$OutputDir = Join-Path $PSScriptRoot "..\results\nsight"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$ReportBase = Join-Path $OutputDir "cuda_kernels_nsys_${Kernel}_N${Size}"

Write-Host "=== Nsight Systems Profile ===" -ForegroundColor Green
Write-Host "  Kernel     : $Kernel"
Write-Host "  Size       : $Size  (millions of elements for vec/reduction; matrix side for matmul)"
Write-Host "  Warmup     : $Warmup  (outside NVTX range)"
Write-Host "  Iterations : $Iterations  (inside NVTX range)"
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
    python src/bench_nsys_target.py `
        --kernel     $Kernel `
        --size       $Size `
        --warmup     $Warmup `
        --iterations $Iterations

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done." -ForegroundColor Green
    Write-Host "Open $ReportBase.nsys-rep in the Nsight Systems desktop app."
    Write-Host ""
    Write-Host "What to look for in the timeline:"
    Write-Host "  NVTX row         : cuda_kernel_opt_$Kernel outer range"
    Write-Host "  Inner ranges     : iter_0 .. iter_$($Iterations - 1)"
    Write-Host "  CUDA kernel bars : should fill most of each iteration range"
    Write-Host "  Idle gaps        : space between bars = CPU kernel launch latency"
    Write-Host "  Warp occupancy   : check the SM Warp Occupancy row if visible"
} else {
    Write-Error "nsys exited with code $LASTEXITCODE."
}
