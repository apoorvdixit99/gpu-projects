"""Matrix multiplication benchmark: CPU vs naive CUDA vs tiled CUDA vs cuBLAS.

Metrics per configuration
-------------------------
  latency_ms_*   Kernel execution time via CUDA events (GPU) / perf_counter (CPU)
  gflops         Arithmetic throughput: 2 * M * K * N / latency_s / 1e9
                 For square matrices (M=K=N): 2 * N³ FLOPs
  speedup_vs_numpy  Relative speedup over NumPy baseline
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _ext import get_ext

# Disable TF32 so torch.mm uses the same float32 precision as our custom kernels.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False


def _time_cuda(fn, warmup: int, iters: int) -> np.ndarray:
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.array(times, dtype=np.float32)


def _time_cpu(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def _gflops(M: int, K: int, N: int, latency_ms: float) -> float:
    return 2 * M * K * N / (latency_ms / 1e3) / 1e9


def benchmark(
    sizes:          list[int] = [256, 512, 1024, 2048],
    warmup:         int       = 10,
    iterations:     int       = 100,
    skip_cpu_above: int       = 1024,
) -> list[dict]:
    """Benchmark square matrix multiplication (M=K=N) across all variants.

    Args:
        sizes:          Square matrix dimensions to sweep.
        warmup:         Warmup iterations before timing.
        iterations:     Timed iterations.
        skip_cpu_above: Skip CPU baseline when N > this (NumPy is very slow
                        for large matrices; speedup data becomes impractical).
    """
    ext = get_ext()
    rows: list[dict] = []

    print("\n=== Matrix Multiplication Benchmark (square, M=K=N) ===")
    print(f"{'Variant':<20} {'N':>6} {'mean ms':>10} {'GFLOPS':>10} {'speedup':>9}")
    print("-" * 62)

    for N in sizes:
        A_np = np.random.randn(N, N).astype(np.float32)
        B_np = np.random.randn(N, N).astype(np.float32)
        A_cu = torch.from_numpy(A_np).cuda()
        B_cu = torch.from_numpy(B_np).cuda()

        # ── Correctness check ─────────────────────────────────────────────
        ref = torch.mm(A_cu, B_cu)
        naive_out = ext.matmul_naive(A_cu, B_cu)
        tiled_out = ext.matmul_tiled(A_cu, B_cu)
        # float32 accumulation error grows with N; use generous tolerance.
        tol = 1e-2 * N ** 0.5
        assert torch.allclose(naive_out, ref, atol=tol), f"matmul_naive mismatch at N={N}"
        assert torch.allclose(tiled_out, ref, atol=tol), f"matmul_tiled mismatch at N={N}"

        # ── CPU baselines ─────────────────────────────────────────────────
        if N <= skip_cpu_above:
            A_tc = torch.from_numpy(A_np)
            B_tc = torch.from_numpy(B_np)
            numpy_ms     = _time_cpu(lambda: np.dot(A_np, B_np), warmup, min(iterations, 20))
            torch_cpu_ms = _time_cpu(lambda: torch.mm(A_tc, B_tc), warmup, min(iterations, 20))
        else:
            numpy_ms     = None
            torch_cpu_ms = None

        # ── CUDA kernels ──────────────────────────────────────────────────
        times_naive  = _time_cuda(lambda: ext.matmul_naive(A_cu, B_cu), warmup, iterations)
        times_tiled  = _time_cuda(lambda: ext.matmul_tiled(A_cu, B_cu), warmup, iterations)
        times_cublas = _time_cuda(lambda: torch.mm(A_cu, B_cu),          warmup, iterations)

        base_ms = numpy_ms if numpy_ms is not None else float(times_naive.mean())

        def _print(variant, ms, base):
            gf = _gflops(N, N, N, ms)
            sp = base / ms
            print(f"  {variant:<18} {N:>6} {ms:>10.3f} {gf:>10.2f} {sp:>8.1f}x")

        if numpy_ms is not None:
            _print("numpy",     numpy_ms,     numpy_ms)
            _print("torch_cpu", torch_cpu_ms, numpy_ms)
            rows.append({"variant": "numpy",     "N": N, "latency_ms_mean": round(numpy_ms, 4),
                         "latency_ms_std": 0.0, "latency_ms_p50": round(numpy_ms, 4),
                         "latency_ms_p95": round(numpy_ms, 4),
                         "gflops": round(_gflops(N, N, N, numpy_ms), 2), "speedup_vs_numpy": 1.0})
            rows.append({"variant": "torch_cpu", "N": N, "latency_ms_mean": round(torch_cpu_ms, 4),
                         "latency_ms_std": 0.0, "latency_ms_p50": round(torch_cpu_ms, 4),
                         "latency_ms_p95": round(torch_cpu_ms, 4),
                         "gflops": round(_gflops(N, N, N, torch_cpu_ms), 2),
                         "speedup_vs_numpy": round(numpy_ms / torch_cpu_ms, 2)})

        for variant, times in [("cuda_naive", times_naive), ("cuda_tiled", times_tiled), ("cublas", times_cublas)]:
            mean_ms = float(times.mean())
            _print(variant, mean_ms, base_ms)
            rows.append({
                "variant":          variant,
                "N":                N,
                "latency_ms_mean":  round(mean_ms, 4),
                "latency_ms_std":   round(float(times.std()), 4),
                "latency_ms_p50":   round(float(np.percentile(times, 50)), 4),
                "latency_ms_p95":   round(float(np.percentile(times, 95)), 4),
                "gflops":           round(_gflops(N, N, N, mean_ms), 2),
                "speedup_vs_numpy": round(base_ms / mean_ms, 2) if numpy_ms is not None else float("nan"),
            })

    return rows


if __name__ == "__main__":
    import pandas as pd
    rows = benchmark()
    print("\n" + pd.DataFrame(rows).to_string(index=False))
