# CUDA Kernel Optimization

Custom CUDA kernels written from scratch, compiled as a PyTorch extension, and benchmarked against CPU baselines and cuBLAS.  Demonstrates how GPU memory hierarchy and thread execution patterns affect real performance.

**Hardware:** NVIDIA RTX 4080 Laptop GPU (Ada Lovelace) · CUDA 12.6  
**Interface:** PyTorch custom extension (`torch.utils.cpp_extension.load`)

---

## What is being measured

Three fundamental GPU operations are implemented at multiple optimization levels:

| Kernel | Variants | Problem sizes |
|---|---|---|
| Vector Addition | naive · float4 grid-stride | N = 1M, 4M, 16M, 64M, 256M |
| Matrix Multiplication | naive · tiled (shared mem) · cuBLAS | N = 256, 512, 1024, 2048 |
| Parallel Reduction | naive · sequential · warp-shuffle | N = 1M, 4M, 16M, 64M, 256M |

**Per-configuration metrics:**

| Module | Metric | Description |
|---|---|---|
| `bench_vector_add.py` | Latency (ms) | CUDA events (GPU) / perf_counter (CPU) |
| `bench_vector_add.py` | Bandwidth (GB/s) | 3 × N × 4 bytes / latency (read A, read B, write C) |
| `bench_matmul.py` | Latency (ms) | CUDA events |
| `bench_matmul.py` | GFLOPS | 2N³ / latency_s / 1e9 |
| `bench_reduction.py` | Latency (ms) | Kernel pass only (CUDA events) |
| `bench_reduction.py` | Bandwidth (GB/s) | N × 4 bytes / latency (read once) |
| All | Speedup vs NumPy | NumPy latency / CUDA latency |

---

## Kernel implementations

### `kernels/vector_add.cu` — Vector Addition

**Naive:** One thread per element, one 32-bit load per thread.  Access is coalesced
(consecutive threads read consecutive memory) but each thread only issues a 32-bit
load, leaving the memory bus under-utilized.

**Optimized (float4 grid-stride):** Each thread loads `float4` (128 bits) per
iteration, quadrupling the effective memory bandwidth per thread.  A grid-stride loop
covers the full array with a fixed-size, SM-saturating grid regardless of N.

### `kernels/matmul.cu` — Matrix Multiplication

**Naive:** Each thread computes one output element by reading an entire row of A and
column of B from global memory.  For N×N matrices, every thread issues N global
reads — O(N³) total global memory traffic.

**Tiled (TILE=16):** Threads cooperate to load 16×16 sub-tiles of A and B into
shared memory before computing partial dot products.  Global reads drop by a factor of
TILE, from O(N³) to O(N³/16).  Shared memory (2 KB/block) is fast and on-chip.
`__syncthreads()` separates the load phase from the compute phase.

**cuBLAS reference:** `torch.mm(A, B)` on GPU, which dispatches to cuBLAS GEMM.
Used as the upper-bound reference; TF32 is disabled to ensure float32 throughout.

### `kernels/reduction.cu` — Parallel Reduction

**Naive (interleaved):** Each step `s`, threads where `tid % (2*s) == 0` are active.
With `s=1`, warp 0 has alternating active/inactive threads → warp divergence every
iteration.  Access pattern also causes shared memory bank conflicts at even strides.

**Sequential (no divergence):** Active set is always `tid < s` — the first `s` threads.
No warp in the grid mixes active/inactive threads; once a warp becomes fully idle it
stays idle.  Sequential access pattern eliminates bank conflicts.

**Warp shuffle:** Loads two elements per thread (halves block count), reduces in shared
memory to 64 active values, then uses `__shfl_down_sync` for the final 5 steps.
Shuffle communicates register values within a warp without touching shared memory at
all — zero latency for the last log₂(32) steps.

---

## Project structure

```
project-6-cuda-kernel-optimization/
├── kernels/
│   ├── kernels.cpp              PyTorch extension bindings (pybind11)
│   ├── vector_add.cu            naive + float4 grid-stride
│   ├── matmul.cu                naive + tiled (TILE=16)
│   └── reduction.cu             naive + sequential + warp-shuffle
├── src/
│   ├── run_benchmark.py         CLI entry point — orchestrates all three benchmarks
│   ├── _ext.py                  Singleton extension loader (JIT compile once)
│   ├── bench_vector_add.py      Vector add: CPU baselines + CUDA variants
│   ├── bench_matmul.py          Matmul: CPU baselines + CUDA variants + cuBLAS
│   ├── bench_reduction.py       Reduction: CPU baselines + CUDA variants
│   ├── bench_nsys_target.py     NVTX-annotated target for Nsight tools
│   └── plot_results.py          Generate all four charts
├── nsight/
│   ├── run_nsys.ps1             Launch Nsight Systems timeline capture
│   └── run_ncu.ps1              Launch Nsight Compute per-kernel profiling
├── results/                     Generated output (gitignored)
│   ├── vec_add_*.csv            Vector add results per run
│   ├── matmul_*.csv             Matmul results per run
│   ├── reduction_*.csv          Reduction results per run
│   ├── plots/                   PNG charts (four files)
│   └── nsight/                  .nsys-rep and .ncu-rep report files
├── ISSUES.md                    Log of issues hit during development
├── requirements.txt
└── README.md
```

---

## How to run

### Prerequisites (one-time setup)

This project compiles a CUDA C++ extension at runtime, which requires:

**1. VS 2022 Build Tools** with the "Desktop development with C++" workload.  
Download: `https://aka.ms/vs/17/release/vs_BuildTools.exe`  
VS 2019 does **not** work — PyTorch 2.12 uses C++20 headers that require MSVC 14.4x.
The installer places files in `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\`.

**2. MSVC environment auto-loaded via PowerShell profile.**  
Add the following to `$PROFILE` (`Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1`)
so `cl.exe` and the MSVC lib paths are available in every new terminal automatically:

```powershell
$_vc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if (Test-Path $_vc) {
    $envLines = cmd /c "`"$_vc`" x64 2>nul && set"
    foreach ($line in $envLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    Remove-Variable _vc
}
```

Once the profile is in place, open any new PowerShell window and the MSVC environment
is ready — no manual step required before running benchmarks.

### Running benchmarks

> Activate the shared venv from the `Nvidia/` parent directory first:
> ```powershell
> .venv\Scripts\Activate.ps1
> cd project-6-cuda-kernel-optimization
> ```
>
> The first run compiles the CUDA extension (~30–60 s). Subsequent runs use the
> cached binary and start immediately.

**Full benchmark run** — all three kernels, all sizes:
```powershell
python src/run_benchmark.py
```

**Skip CPU baselines** (significantly faster):
```powershell
python src/run_benchmark.py --no-cpu
```

**Custom sizes or iteration count:**
```powershell
python src/run_benchmark.py --mat-sizes 256 512 1024 --warmup 5 --iterations 50
python src/run_benchmark.py --vec-sizes 1048576 16777216 268435456
```

**Run a single benchmark directly:**
```powershell
python src/bench_vector_add.py
python src/bench_matmul.py
python src/bench_reduction.py
```

**Nsight Systems timeline** (admin PowerShell may be required for NVTX):
```powershell
$env:PATH += ";C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.x.x\target-windows-x64"

.\nsight\run_nsys.ps1                                      # matmul_tiled, N=1024
.\nsight\run_nsys.ps1 -Kernel reduce_shuffle -Size 16      # reduction, 16M elements
.\nsight\run_nsys.ps1 -Kernel vec_add_opt -Size 64         # vector add, 64M elements
```
Open the resulting `.nsys-rep` in the **Nsight Systems** desktop app.

**Nsight Compute per-kernel metrics:**
```powershell
.\nsight\run_ncu.ps1                                       # matmul_tiled, N=512
.\nsight\run_ncu.ps1 -Kernel reduce_naive -Size 4          # reduction naive, 4M elements
.\nsight\run_ncu.ps1 -Kernel vec_add_naive -KernelFilter "_vec_add_naive"
```
Open the resulting `.ncu-rep` in the **Nsight Compute** desktop app.

Results are saved to `results/` with timestamps.

---

## Key design decisions

**Why PyTorch custom extensions instead of standalone CUDA binaries?**  
`torch.utils.cpp_extension.load()` compiles the `.cu` files with `nvcc` and links
them against PyTorch's ATen library, giving direct access to `torch::Tensor` inside
the kernel launchers.  This means inputs and outputs are standard PyTorch tensors —
timing with CUDA events, correctness checks with `torch.allclose`, and memory
management are all handled by the existing PyTorch stack.  No separate build system,
no ctypes or subprocess plumbing.

**Why time the reduction kernel pass only (not the partial-sum collection)?**  
Parallel reduction is a two-step operation: the CUDA kernel reduces N elements to
`ceil(N/THREADS)` partial sums, then the Python side calls `partial.sum()`.  The
kernel pass is what differs between the three variants; the partial-sum step is
identical and tiny (at most a few thousand elements).  Timing only the kernel isolates
the algorithmic difference.  The README notes this so the latency numbers are
interpreted correctly.

**Why disable TF32 for matmul?**  
Ada Lovelace GPUs default to TF32 precision for GEMM operations through cuBLAS, which
effectively treats float32 mantissa bits as 10-bit.  Disabling TF32 ensures `torch.mm`
(cuBLAS) and the custom kernels all operate in true float32, making the comparison
numerically fair.  TF32 would make cuBLAS appear faster than it actually is at float32.

**Why float4 rather than float2 for the optimized vector-add kernel?**  
float4 issues a single 128-bit load instruction (LDG.128 on NVIDIA hardware), which
is the widest load supported by CUDA.  Each thread needs only one instruction per
4-element chunk, minimizing instruction overhead.  PyTorch CUDA tensors are at least
64-byte aligned, so reinterpreting the data pointer as `float4*` is always safe.

**Why `skip_cpu_above` rather than always measuring CPU?**  
NumPy matrix multiplication at N=2048 takes ~2 seconds per call.  Running 100
iterations would add 3+ minutes to the benchmark with no additional insight —
the speedup at N=1024 already demonstrates the effect clearly.  The cutoff is
explicit and documented so results tables don't appear incomplete.
