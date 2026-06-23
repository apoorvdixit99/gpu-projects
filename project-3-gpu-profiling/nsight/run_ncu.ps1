<#
.SYNOPSIS
    Profile individual CUDA kernels with NVIDIA Nsight Compute.

.DESCRIPTION
    Runs ncu to collect detailed per-kernel hardware counters: SM occupancy,
    memory bandwidth utilization, instruction throughput, warp efficiency, and
    L1/L2 cache hit rates.  Use this to understand *why* a kernel is slow, not
    just *that* it is slow.

    Output file: results/nsight/gpt2_ncu_bs<N>_seq<N>.ncu-rep
    Open in   : NVIDIA Nsight Compute desktop application

.NOTES
    ncu replays each kernel 5-20x to collect all counter sets, so the
    profiled process runs 10-50x slower than normal.  Use small batch sizes
    (bs=1) and few iterations to keep runtime manageable.

    Key metrics to examine in the report:
      SM Throughput   -- how close to peak compute utilisation
      Mem Throughput  -- how close to peak memory bandwidth
      Warp Occupancy  -- active warps / max warps per SM
      L1/L2 Hit Rate  -- cache effectiveness for weight reuse

.EXAMPLE
    .\nsight\run_ncu.ps1
    .\nsight\run_ncu.ps1 -BatchSize 1 -SeqLen 64 -KernelFilter "ampere_sgemm"
#>

param (
    [int]   $BatchSize    = 1,
    [int]   $SeqLen       = 64,
    [int]   $Iterations   = 3,
    [string]$KernelFilter = ""
)

$OutputDir = Join-Path $PSScriptRoot "..\results\nsight"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$ReportBase = Join-Path $OutputDir "gpt2_ncu_bs${BatchSize}_seq${SeqLen}"

Write-Host "=== Nsight Compute Profile ===" -ForegroundColor Green
Write-Host "  Batch size    : $BatchSize"
Write-Host "  Seq length    : $SeqLen"
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
    "--set",             "full",
    "--export",          "$ReportBase",
    "--force-overwrite",
    "--target-processes", "all"
)
if ($KernelFilter) {
    # Wrap in regex: prefix so PowerShell does not expand special chars
    $NcuArgs += @("--kernel-name", "regex:$KernelFilter")
}
$NcuArgs += @(
    "python", "src/profile_nsys_target.py",
    "--batch-size",  "$BatchSize",
    "--seq-len",     "$SeqLen",
    "--warmup",      "5",
    "--iterations",  "$Iterations"
)

ncu @NcuArgs

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done." -ForegroundColor Green
    Write-Host "Open $ReportBase.ncu-rep in the Nsight Compute desktop app."
} else {
    Write-Error "ncu exited with code $LASTEXITCODE."
}
