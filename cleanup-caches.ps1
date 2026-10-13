# ML Cache Cleanup
# Clears HuggingFace model cache (Windows + WSL), Ollama model weights, and
# Docker images/build cache inside WSL. Everything here is re-downloadable /
# rebuildable, so it's safe to run whenever disk space is tight.
#
# Usage:
#   .\cleanup-caches.ps1                  # interactive, asks before each category
#   .\cleanup-caches.ps1 -DryRun           # just report sizes, delete nothing
#   .\cleanup-caches.ps1 -SkipConfirm      # clear everything without prompting
#   .\cleanup-caches.ps1 -CompactVhdx      # also shrink the WSL vhdx after cleanup (needs admin)

param(
    [switch]$DryRun,
    [switch]$SkipConfirm,
    [switch]$CompactVhdx
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$WslDistro = "Ubuntu"
$WinHfHub = "$env:USERPROFILE\.cache\huggingface\hub"

function Get-SizeGB($Path) {
    if (-not (Test-Path $Path)) { return 0 }
    $items = Get-ChildItem $Path -Recurse -File -Force -ErrorAction SilentlyContinue
    if (-not $items) { return 0 }
    $stats = $items | Measure-Object -Property Length -Sum
    if (-not $stats.Sum) { return 0 }
    return [math]::Round($stats.Sum / 1GB, 2)
}

function Confirm-Step($Message) {
    if ($SkipConfirm -or $DryRun) { return $true }
    $resp = Read-Host "$Message [y/N]"
    return $resp -match '^[Yy]'
}

Write-Host "=== Current sizes ===" -ForegroundColor Cyan
$hfWinGB = Get-SizeGB $WinHfHub
Write-Host ("HuggingFace cache (Windows): {0} GB" -f $hfWinGB)

$hfWslGB = (wsl.exe -d $WslDistro -- bash -c "du -sb ~/.cache/huggingface 2>/dev/null | cut -f1")
$hfWslGB = if ($hfWslGB) { [math]::Round([double]$hfWslGB / 1GB, 2) } else { 0 }
Write-Host ("HuggingFace cache (WSL): {0} GB" -f $hfWslGB)

$ollamaModels = ollama list 2>$null | Select-Object -Skip 1
Write-Host ("Ollama models: {0} model(s)" -f ($ollamaModels | Measure-Object).Count)
$ollamaModels | ForEach-Object { Write-Host "  $_" }

$dockerDf = wsl.exe -d $WslDistro -- bash -c "docker system df 2>/dev/null"
Write-Host "Docker (WSL):"
$dockerDf | ForEach-Object { Write-Host "  $_" }
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run only - nothing was deleted." -ForegroundColor Yellow
    exit 0
}

# --- HuggingFace cache ---
if ($hfWinGB -gt 0 -or $hfWslGB -gt 0) {
    if (Confirm-Step "Clear HuggingFace model cache (Windows: $hfWinGB GB, WSL: $hfWslGB GB)?") {
        Write-Host "=== Clearing HuggingFace cache ===" -ForegroundColor Cyan
        if (Test-Path $WinHfHub) { Remove-Item "$WinHfHub\*" -Recurse -Force -ErrorAction SilentlyContinue }
        wsl.exe -d $WslDistro -- bash -c "rm -rf ~/.cache/huggingface/* 2>/dev/null || sudo rm -rf ~/.cache/huggingface/*"
        Write-Host "HuggingFace cache cleared." -ForegroundColor Green
    }
}

# --- Ollama models ---
if (($ollamaModels | Measure-Object).Count -gt 0) {
    if (Confirm-Step "Remove all Ollama model weights listed above?") {
        Write-Host "=== Clearing Ollama models ===" -ForegroundColor Cyan
        $ollamaModels | ForEach-Object {
            $name = ($_ -split '\s+')[0]
            if ($name) { ollama rm $name }
        }
        Write-Host "Ollama models cleared." -ForegroundColor Green
    }
}

# --- Docker images / build cache ---
if (Confirm-Step "Prune all unused Docker images + build cache inside WSL?") {
    Write-Host "=== Pruning Docker (WSL) ===" -ForegroundColor Cyan
    wsl.exe -d $WslDistro -- bash -c "docker system prune -a --volumes -f"
    Write-Host "Docker images/build cache pruned." -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Cleanup complete ===" -ForegroundColor Green

# --- Optional: shrink the WSL vhdx to reclaim the freed space on the C: drive ---
if ($CompactVhdx) {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Host "Skipping vhdx compaction: re-run this script from an elevated (Administrator) PowerShell with -CompactVhdx." -ForegroundColor Yellow
    } else {
        Write-Host "=== Compacting WSL vhdx ===" -ForegroundColor Cyan
        wsl --shutdown
        $vhdxPath = (Get-ChildItem "$env:LOCALAPPDATA\wsl" -Filter "ext4.vhdx" -Recurse -Force -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName)
        if ($vhdxPath) {
            $diskpartScript = @"
select vdisk file="$vhdxPath"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@
            $diskpartScript | diskpart
            Write-Host "vhdx compacted." -ForegroundColor Green
        } else {
            Write-Host "Could not locate ext4.vhdx under $env:LOCALAPPDATA\wsl" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "Tip: run with -CompactVhdx from an elevated PowerShell to also shrink the WSL virtual disk on the C: drive." -ForegroundColor DarkGray
}
