# GPU Profiling & Bottleneck Analysis

Profiles GPT-2 (124M) PyTorch inference with `torch.profiler`, CUDA events, and NVIDIA Nsight tools to understand where inference time is actually spent.

**Hardware:** NVIDIA RTX 4080 Laptop GPU (Ada Lovelace) · CUDA 12.6  
**Model:** GPT-2 124M (HuggingFace `gpt2`)

---

## What is being measured

A single GPT-2 forward pass is profiled across a range of batch sizes with a fixed sequence length:

| Dimension | Values |
|---|---|
| Batch size | 1, 2, 4, 8, 16, 32 |
| Sequence length | 128 (fixed) |

**Per-configuration metrics:**

| Module | Metric | Description |
|---|---|---|
| `profile_torch.py` | CUDA time per kernel (ms, %) | Where GPU time is actually spent |
| `profile_torch.py` | CPU time per op (ms) | Python dispatch cost per operation |
| `profile_torch.py` | Call count | How many times each kernel is invoked |
| `profile_bottlenecks.py` | GPU kernel time (ms) | Pure device execution time (CUDA events) |
| `profile_bottlenecks.py` | Wall time (ms) | Elapsed time including Python dispatch |
| `profile_bottlenecks.py` | CPU overhead (ms) | Wall time − GPU time |
| `profile_bottlenecks.py` | Overlap % | GPU time overlapping with CPU dispatch |
| `profile_memory.py` | Peak allocated (MB) | Tensor memory high-water mark |
| `profile_memory.py` | Peak reserved (MB) | CUDA allocator cache high-water mark |
| `profile_memory.py` | Fragmentation % | Cached-but-unused memory fraction |

---

## Analysis modules

### `profile_torch.py` — Kernel breakdown
Uses `torch.profiler` with a schedule (`wait=1, warmup=1, active=3`) so the profiler
reaches steady state before recording any data. Produces per batch size:
- **Chrome trace JSON** — open in [Perfetto UI](https://ui.perfetto.dev) for a full
  visual timeline of every CPU op and CUDA kernel (do not use `chrome://tracing` for
  large traces — see Issues)
- **Text report** — top-10 kernels ranked by CUDA time, written to `results/reports/`

### `profile_bottlenecks.py` — CPU/GPU timing
Each forward pass is timed with both a CUDA event pair (pure GPU kernel time) and
`time.perf_counter()` (wall time). The difference is the CPU dispatch overhead. A
model is GPU-bound when `overlap_pct ≈ 100%`; CPU-bound when `wall_time >> gpu_time`.

### `profile_memory.py` — Memory usage
Calls `torch.cuda.reset_peak_memory_stats()` before each measurement pass, then
reports both `max_memory_allocated` (actual tensor data) and `max_memory_reserved`
(allocator cache, always ≥ allocated). The gap between the two is memory the
allocator holds cached to avoid `cudaMalloc` on the next call.

### Nsight Systems (`nsight/run_nsys.ps1`)
Launches the process under `nsys` to collect a CUDA kernel timeline with NVTX range
annotations. Captures `--trace=cuda,nvtx`: kernel start/stop times and the
`gpt2_inference` / `forward_N` range markers that label each forward pass in the
timeline. Requires admin PowerShell for NVTX injection and the Nsight Systems 2024.4.2
binary specifically (see Issues for why 2026.1.3 does not work on this GPU).

### Nsight Compute (`nsight/run_ncu.ps1`)
Runs `ncu --set full` to collect detailed per-kernel hardware counters: SM occupancy,
memory bandwidth utilization, warp efficiency, and L1/L2 cache hit rates. Use this to
understand *why* a kernel is slow, not just *that* it is slow. Note: `ncu` replays
each kernel 5–20×, so the profiled process runs 10–50× slower than normal.

---

## Project structure

```
project-3-gpu-profiling/
├── src/
│   ├── run_profiler.py          CLI entry point — orchestrates all three modules
│   ├── profile_torch.py         torch.profiler kernel breakdown + Chrome traces
│   ├── profile_bottlenecks.py   CPU/GPU timing with CUDA events + CPU timer
│   ├── profile_memory.py        GPU memory allocation and fragmentation
│   ├── profile_nsys_target.py   Thin wrapper with NVTX ranges for Nsight tools
│   └── plot_results.py          Generate all four charts
├── nsight/
│   ├── run_nsys.ps1             Launch nsys timeline capture
│   └── run_ncu.ps1              Launch ncu per-kernel profiling
├── results/                     Generated output (gitignored)
│   ├── kernels_*.csv            Top-10 kernels per batch size
│   ├── bottlenecks_*.csv        GPU/wall/overhead times per batch size
│   ├── memory_*.csv             Memory stats per batch size
│   ├── traces/                  Chrome/Perfetto trace JSON files
│   ├── reports/                 Top-kernel text reports per batch size
│   ├── plots/                   PNG charts (four files)
│   └── nsight/                  .nsys-rep and .ncu-rep report files
├── ISSUES.md                    Log of issues hit during development and fixes applied
├── requirements.txt
└── README.md
```

---

## How to run

> Activate the shared venv from the `Nvidia/` parent directory first:
> ```powershell
> .venv\Scripts\Activate.ps1
> cd project-3-gpu-profiling
> ```

**Full profiling run** — all three modules across all batch sizes:
```powershell
python src/run_profiler.py
```

**Custom batch sizes or sequence length:**
```powershell
python src/run_profiler.py --batch-sizes 1 4 16 32 --seq-len 64
python src/run_profiler.py --iterations 50 --no-plot
```

**Run a single module directly:**
```powershell
python src/profile_torch.py          # kernel breakdown only
python src/profile_bottlenecks.py    # CPU/GPU timing only
python src/profile_memory.py         # memory only
```

**Nsight Systems timeline** (admin PowerShell required for NVTX capture):
```powershell
# Ensure nsys 2024.4.2 is on PATH — 2026.1.3 has a bug on this GPU (see Issues)
$env:PATH += ";C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64"

.\nsight\run_nsys.ps1
.\nsight\run_nsys.ps1 -BatchSize 8 -SeqLen 256 -Iterations 20
```
Open the resulting `.nsys-rep` in the **Nsight Systems 2024.4.2** desktop app.

**Nsight Compute per-kernel metrics** (requires Nsight Compute desktop app):
```powershell
.\nsight\run_ncu.ps1
.\nsight\run_ncu.ps1 -BatchSize 1 -SeqLen 64 -KernelFilter "ampere_sgemm"
```

Results are saved to `results/` with timestamps. Chrome traces open in
[ui.perfetto.dev](https://ui.perfetto.dev) — drag and drop the JSON file.

---

## Future scope

### Nsight Compute per-kernel analysis
`nsight/run_ncu.ps1` is implemented and ready to run but results are not yet included
in this project. Nsight Compute (`ncu --set full`) collects detailed per-kernel
hardware counters that answer *why* a kernel is slow:

| Counter | What it reveals |
|---|---|
| SM Throughput | How close to peak compute utilisation |
| Memory Throughput | How close to peak memory bandwidth |
| Warp Occupancy | Active warps / max warps per SM |
| L1 / L2 Hit Rate | Cache effectiveness for weight reuse |

Planned analysis:
- Identify the top GEMM kernel from `profile_torch.py` results and drill into its occupancy and memory access pattern
- Compare attention vs feed-forward kernel efficiency
- Check whether the model is compute-bound or memory-bandwidth-bound at each batch size

Requires admin PowerShell and `--target-processes all` (see Issues).

---

## Key design decisions

**Why CUDA events alongside `time.perf_counter()`?**  
CUDA events measure only the time the GPU spent executing kernels; they have no
visibility into Python overhead. `time.perf_counter()` captures the full wall time
including kernel launch latency and Python dispatch. Running both in the same pass
separates GPU work from CPU overhead without requiring two separate profiling runs.

**Why `torch.profiler` with a schedule instead of a plain context manager?**  
Starting the profiler from step zero captures Python import time, model loading, and
the profiler's own startup overhead. The schedule's `wait` steps let the interpreter
reach steady state and `warmup` steps let the profiler's internal observer stabilise,
so only the `active` steps are recorded. These represent the real steady-state kernel
distribution.

**Why `reset_peak_memory_stats()` after warmup but before measurement?**  
The CUDA allocator accumulates allocations across calls. Calling the reset before the
single measurement pass means the peak stats reflect only that pass. Without the
reset, peak values would include all allocations since process start, inflating the
numbers by 2–3× on the first batch.

**Why separate `peak_allocated` and `peak_reserved` metrics?**  
`max_memory_allocated` is the true high-water mark of memory in use by tensors.
`max_memory_reserved` is the high-water mark of memory the allocator holds from CUDA
(including empty cached blocks). Reporting only `reserved` inflates the apparent
memory cost on small batch sizes where the allocator aggressively pre-caches blocks.

**Why a dedicated `profile_nsys_target.py` instead of profiling `run_profiler.py`?**  
`run_profiler.py` runs all three modules sequentially, producing a trace that is
hundreds of seconds long with no clear structure. The dedicated target runs only `N`
forward passes with NVTX annotations, producing a short, navigable trace where every
forward pass is labelled.
