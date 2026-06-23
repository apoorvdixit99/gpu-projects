<#
.SYNOPSIS
    Profile a custom CUDA kernel with NVIDIA Nsight Compute.

.DESCRIPTION
    Runs ncu to collect detailed per-kernel hardware counters: SM occupancy,
    memory bandwidth utilization, instruction throughput, warp efficiency, and
    L1/L2 cache hit rates.  Use this to understand *why* a kernel is slow, not
    just *that* it is slow.

    Output file: results/nsight/cuda_kernels_ncu_<Kernel>_N<Size>.ncu-rep
    Open in   : NVIDIA Nsight Compute desktop application

.NOTES
    ncu replays each kernel 5-20x to collect all counter sets, so the
    profiled process runs 10-50x slower than normal.  Use small sizes
    and few iterations to keep runtime manageable.

    Key metrics to examine:
      SM Throughput      how close to peak compute utilisation
      Mem Throughput     how close to peak memory bandwidth (compare vec add naive vs opt)
      Warp Occupancy     active warps / max warps per SM
      L1/L2 Hit Rate     cache effectiveness
      Warp Divergence    branch divergence penalty (compare reduction naive vs sequential)

.EXAMPLE
    .\nsight\run_ncu.ps1
    .\nsight\run_ncu.ps1 -Kernel reduce_naive -Size 4
    .\nsight\run_ncu.ps1 -Kernel matmul_tiled -Size 512 -KernelFilter "_matmul_tiled"
#>

param (
    [ValidateSet("vec_add_naive","vec_add_opt",
                 "matmul_naive","matmul_tiled",
                 "reduce_naive","reduce_sequential","reduce_shuffle")]
    [string]$Kernel       = "matmul_tiled",
    [int]   $Size         = 512,
    [int]   $Iterations   = 3,
    [string]$KernelFilter = ""
)

$OutputDir = Join-Path $PSScriptRoot "..\results\nsight"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$ReportBase = Join-Path $OutputDir "cuda_kernels_ncu_${Kernel}_N${Size}"

Write-Host "=== Nsight Compute Profile ===" -ForegroundColor Green
Write-Host "  Kernel        : $Kernel"
Write-Host "  Size          : $Size"
Write-Host "  Iterations    : $Iterations  (ncu replays each kernel for counter collection)"
Write-Host "  Kernel filter : $(if ($KernelFilter) { $KernelFilter } else { '(all kernels)' })"
Write-Host "  Output        : $ReportBase.ncu-rep"
Write-Host ""

if (-not (Get-Command ncu -ErrorAction SilentlyContinue)) {
    Write-Error ("ncu not found on PATH.`n" +
                 "Add the Nsight Compute install directory to PATH, e.g.:`n" +
                 "  C:\Program Files\NVIDIA Nsight Compute 2024.x\host\windows-desktop-win7-x64")
    exit 1
}

$NcuArgs = @(
    "--set",              "full",
    "--export",           "$ReportBase",
    "--force-overwrite",
    "--target-processes", "all"
)
if ($KernelFilter) {
    $NcuArgs += @("--kernel-name", "regex:$KernelFilter")
}
$NcuArgs += @(
    "python", "src/bench_nsys_target.py",
    "--kernel",     "$Kernel",
    "--size",       "$Size",
    "--warmup",     "2",
    "--iterations", "$Iterations"
)

ncu @NcuArgs

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done." -ForegroundColor Green
    Write-Host "Open $ReportBase.ncu-rep in the Nsight Compute desktop app."
    Write-Host ""
    Write-Host "Suggested analysis workflow:"
    Write-Host "  1. Open the Speed Of Light (SOL) roofline — is the kernel compute- or memory-bound?"
    Write-Host "  2. For memory-bound kernels (vec add, reduction): check Mem Throughput vs peak"
    Write-Host "  3. For reduction_naive: compare Warp State Statistics to reduction_sequential"
    Write-Host "  4. For matmul_tiled: check L1 hit rate for shared memory effectiveness"
    Write-Host "  5. Warp Occupancy: low occupancy limits latency hiding"
} else {
    Write-Error "ncu exited with code $LASTEXITCODE."
}
