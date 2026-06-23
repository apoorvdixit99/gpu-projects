# Known Issues & Fixes

Issues encountered during development and first run, in chronological order.

---

## Issue 1 — `FunctionEventAvg` has no attribute `cuda_time_total` in PyTorch 2.12

**Error**
```
AttributeError: 'FunctionEventAvg' object has no attribute 'cuda_time_total'.
Did you mean: 'cpu_time_total'?
```
at `profile_torch.py`, line `total_cuda_us = sum(e.cuda_time_total for e in averages)`.

**Cause**
In PyTorch 2.x (observed on 2.12.1+cu126), `key_averages()` returns `FunctionEventAvg`
objects whose CUDA time is stored under `device_time_total` rather than
`cuda_time_total`. The attribute `cuda_time_total` does not exist on these objects
in this build. The `.table(sort_by="cuda_time_total")` method still works because it
maps the sort key string to the correct internal attribute internally, but direct
attribute access fails.

**Fix**
Added a `_cuda_us()` helper in `profile_torch.py` that tries both attribute names
and falls back to zero:
```python
def _cuda_us(e: object) -> float:
    v = getattr(e, "cuda_time_total", None)
    if v is None:
        v = getattr(e, "device_time_total", 0)
    return float(v)
```
All direct `e.cuda_time_total` accesses throughout `profile_kernels()` were replaced
with `_cuda_us(e)`.

---

## Issue 2 — `ProfilerStep*` appearing as the top kernel in all results

**Symptom**
Every batch size reported `ProfilerStep*` as the top kernel by CUDA time, accounting
for 20–58% of total CUDA time. The kernel breakdown chart and text reports were
dominated by this entry, burying the real attention and matmul kernels below it.

**Cause**
`ProfilerStep*` is a synthetic event that `torch.profiler` inserts to represent the
wall time of an entire profiler step (time between consecutive `prof.step()` calls).
It is not a real CUDA kernel — it is an accounting artifact that accumulates all
device time for the step into a single pseudo-event. Because `key_averages()` includes
it alongside real kernel events, it sorts to the top when ranked by CUDA time.

**Fix**
Filter out events whose key starts with `"ProfilerStep"` before sorting:
```python
real_events   = [e for e in averages if not e.key.startswith("ProfilerStep")]
total_cuda_us = sum(_cuda_us(e) for e in real_events)
top10 = sorted(real_events, key=_cuda_us, reverse=True)[:10]
```

---

## Issue 3 — libkineto USDT log lines printed to stdout during profiling

**Symptom**
Each profiler cycle printed two noisy lines to the terminal:
```
USDT:2026-06-21 16:51:26 16748:23148 ...ActivityProfilerController.cpp:415] profiler_start
USDT:2026-06-21 16:51:26 16748:23148 ...ActivityProfilerController.cpp:455] profiler_stop
```

**Cause**
PyTorch's profiler backend (libkineto) emits trace-level log messages at log level 0
(VERBOSE). On Windows these are written directly to stdout rather than stderr, so
they appear inline with the script's own output.

**Fix**
Set `KINETO_LOG_LEVEL=5` (ERROR) before any torch import so libkineto initialises
with a higher minimum log level:
```python
import os
os.environ.setdefault("KINETO_LOG_LEVEL", "5")
```
Added to `run_profiler.py` before the `import torch` chain.

---

## Issue 4 — `ncu` does not recognise `--output`; correct flag is `--export`

**Error**
```
==ERROR== unrecognised option '--output'. Use --help for further details.
```

**Cause**
Nsight Compute uses `--export` (not `--output`) for specifying the output report file.
The `--output` flag does not exist in the ncu CLI.

**Fix**
Changed `--output` to `--export` in `nsight/run_ncu.ps1`:
```powershell
$NcuArgs = @(
    "--set",    "full",
    "--export", "$ReportBase",
    ...
)
```

---

## Issue 5 — `ncu` permission denied: `ERR_NVGPUCTRPERM`

**Error**
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

**Cause**
Windows restricts access to low-level GPU hardware performance counters to
administrators only. Running `ncu` from a standard (non-elevated) PowerShell
session is blocked at the driver level.

**Fix**
Run PowerShell as Administrator before invoking `ncu`:
1. Search **PowerShell** in Start menu
2. Right-click → **Run as administrator**
3. Re-activate the venv and re-run the script:
```powershell
cd C:\Users\apoor\Desktop\projects\Nvidia\project-3-gpu-profiling
..\.venv\Scripts\Activate.ps1
.\nsight\run_ncu.ps1
```

---

## Issue 6 — `nsys` not found on PATH despite Nsight Systems being installed

**Error**
```
nsys : The term 'nsys' is not recognized as the name of a cmdlet, function,
script file, or operable program.
```

**Cause**
The Nsight Systems installer does not add its binary directory to the system or user
PATH. `nsys.exe` was found at:
```
C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe
```
(Two additional copies also exist — one bundled inside Nsight Compute 2024.3.0 and
one from an older Nsight Systems 2024.4.2 install.)

**Fix**
Add the latest version's `target-windows-x64` directory to PATH. To persist across
sessions, set it at the user level:
```powershell
[Environment]::SetEnvironmentVariable(
    "PATH",
    [Environment]::GetEnvironmentVariable("PATH", "User") + ";C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64",
    "User"
)
```
Open a new terminal after running this. Verify with `nsys --version`.

---

## Issue 7 — `nsys` rejects `--trace=osrt` on Nsight Systems 2026.x

**Error**
```
Illegal --trace argument 'osrt'
Possible --trace values are one or more of 'cuda', 'cuda-sw', 'nvtx', 'wddm' ...
```

**Cause**
`osrt` (OS runtime tracing) was removed in Nsight Systems 2026.x. The Windows
equivalent for CPU-side driver dispatch tracing is `wddm` (Windows Display Driver
Model).

**Fix**
Replaced `osrt` with `wddm` in `nsight/run_nsys.ps1`:
```powershell
nsys profile --trace=cuda,nvtx,wddm ...
```
(Further revised to drop `wddm` entirely — see Issue 8.)

---

## Issue 8 — `wddm` requires admin; `--gpu-metrics-device` deprecated and also needs admin

**Errors**
```
WARNING: Wddm trace requires administrative privileges, disabling.
Warning: '--gpu-metrics-device' is deprecated. Use '--gpu-metrics-devices' instead.
Illegal --gpu-metrics-devices usage.
None of the installed GPUs are supported: Ada AD104 | ... Insufficient privilege
```

**Cause**
Two separate problems surfaced together:
- `wddm` tracing (added in Issue 7 as an `osrt` replacement) also requires admin on
  Windows — it captures kernel-mode driver events.
- `--gpu-metrics-device=all` was renamed to `--gpu-metrics-devices` in Nsight Systems
  2026.x, and GPU hardware metrics collection also requires admin (same
  `ERR_NVGPUCTRPERM` restriction as `ncu`).

Neither flag is needed for the core use case of capturing CUDA kernel timelines and
NVTX ranges.

**Fix**
Removed both flags from `nsight/run_nsys.ps1`. The minimal non-admin trace command is:
```powershell
nsys profile --trace=cuda,nvtx --output="$ReportBase" ...
```
This captures all CUDA kernel activity and NVTX ranges without requiring elevated
privileges.

---

## Issue 9 — NVTX ranges not captured and CUDA falls back to software trace without admin

**Warnings in Diagnostics Summary**
```
No NVTX events collected. Does the process use NVTX?
CUDA hardware tracing is not supported on this system.
A legacy (software instrumented) trace was collected instead.
```

**Cause**
Two separate privilege restrictions on Windows:
- **NVTX**: nsys intercepts `torch.cuda.nvtx.range_push/pop` calls via system-level
  DLL injection. Without admin, the injection cannot hook into the NVTX API; the
  calls execute normally but produce no events visible to nsys. The `gpt2_inference`
  and `forward_N` range labels are absent from the timeline.
- **CUDA hardware trace**: nsys falls back to software-instrumented CUDA tracing
  (CUPTI software mode) when GPU hardware counter access is denied. Timing is
  slightly less precise than hardware mode but the kernel sequence and relative
  durations remain accurate. 16,714 CUPTI events were still collected.

**Resolution**
CUDA kernel data is fully captured and the trace is usable — all kernels appear in
the Timeline View under process 34608's threads. Only the NVTX labels are missing.
To get NVTX ranges, run `run_nsys.ps1` from an admin PowerShell session. The
`NVrmirbac` registry fix (Issue 5) resolves the hardware trace restriction for `ncu`
but does not affect nsys NVTX injection on Windows.

---

## Issue 10 — `--force-overwrite` fails with "Permission denied" when report is open in GUI

**Error**
```
Failed to truncate '...gpt2_nsys_bs1_seq128.nsys-rep': Permission denied.
```

**Cause**
The Nsight Systems desktop app had the previous `.nsys-rep` file open, locking it.
`--force-overwrite` can delete and recreate the file but cannot truncate a file held
open by another process, even when running as admin.

**Fix**
Close the Nsight Systems GUI (or close the open report tab) before rerunning the
script. The `.nsys-rep` file must not be open in any application when nsys tries to
overwrite it.

---

## Issue 11 — Nsight Systems 2026.1.3 throws `GetNumTpcs / NotInitializedException`

**Error**
```
**** Analysis failed with: Status: TargetProfilingFailed
...CudaEvent.h(454): Throw in function GetNumTpcs(void) const
NotInitializedException: Data member NumTpcs was not initialized
```

**Cause**
A bug in Nsight Systems 2026.1.3 with the RTX 4080 Laptop GPU (Ada AD104). The
hardware metrics subsystem (`QuadDCommon`) fails to initialise `NumTpcs` — likely
because the Laptop GPU variant exposes a different TPC (Texture Processing Cluster)
topology than the desktop Ada cards the 2026.1.3 build was validated against.

**Resolution**
The `GetNumTpcs` error fires at `00:00.004` — before any profiling begins — and
corrupts the post-profiling Analysis phase. Despite CUDA and NVTX injection
initialising successfully and 16,714 CUPTI events being produced, the Analysis phase
reports "No CUDA events collected" and "No NVTX events collected" because it cannot
attribute events to processes after the hardware metrics subsystem failed to
initialise.

**Fix**
Switch to the Nsight Systems 2024.4.2 binary (bundled with the CUDA Toolkit), which
does not have the `GetNumTpcs` bug on this GPU. Update PATH for the session:
```powershell
$env:PATH = ($env:PATH -replace [regex]::Escape("C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64"), "")
$env:PATH += ";C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64"
```
The 2024.4.2 GUI already installed matches this binary, so reports open correctly.
With 2024.4.2: no `GetNumTpcs` error, 16,100 CUDA events collected, 21 NVTX events
collected (1 `gpt2_inference` + 20 `forward_N` ranges).
