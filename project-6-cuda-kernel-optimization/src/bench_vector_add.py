"""Vector-addition benchmark: CPU (NumPy, PyTorch) vs naive CUDA vs optimized CUDA.

Metrics per configuration
-------------------------
  latency_ms_*       Kernel execution time measured with CUDA events (GPU variants)
                     or time.perf_counter() (CPU variants)
  bandwidth_gb_s     Achieved memory bandwidth: 3 * N * 4 bytes / latency_s
                     (read A, read B, write C — three full passes over memory)
  speedup_vs_numpy   latency_numpy / latency_cuda
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _ext import get_ext


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
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def _bandwidth(N: int, latency_ms: float) -> float:
    bytes_transferred = 3 * N * 4  # read A, read B, write C
    return bytes_transferred / (latency_ms / 1e3) / 1e9


def _row(variant: str, N: int, times_ms, numpy_ms: float) -> dict:
    mean_ms = float(times_ms.mean()) if isinstance(times_ms, np.ndarray) else float(times_ms)
    return {
        "variant":          variant,
        "N":                N,
        "latency_ms_mean":  round(mean_ms, 4),
        "latency_ms_std":   round(float(times_ms.std()), 4) if isinstance(times_ms, np.ndarray) else 0.0,
        "latency_ms_p50":   round(float(np.percentile(times_ms, 50)), 4) if isinstance(times_ms, np.ndarray) else round(mean_ms, 4),
        "latency_ms_p95":   round(float(np.percentile(times_ms, 95)), 4) if isinstance(times_ms, np.ndarray) else round(mean_ms, 4),
        "bandwidth_gb_s":   round(_bandwidth(N, mean_ms), 2),
        "speedup_vs_numpy": round(numpy_ms / mean_ms, 2),
    }


def benchmark(
    sizes:      list[int] = [2**20, 2**22, 2**24, 2**26, 2**28],
    warmup:     int       = 10,
    iterations: int       = 100,
    skip_cpu_above: int   = 2**26,
) -> list[dict]:
    """Benchmark vector addition across all variants and sizes.

    Args:
        sizes:          Element counts (must be divisible by 4).
        warmup:         Warmup iterations before timing.
        iterations:     Timed iterations.
        skip_cpu_above: Skip CPU baselines for N larger than this (they become
                        very slow and the speedup data is not interesting there).
    """
    ext = get_ext()
    rows: list[dict] = []

    print("\n=== Vector Addition Benchmark ===")
    print(f"{'Variant':<20} {'N':>10} {'mean ms':>10} {'GB/s':>10} {'speedup':>9}")
    print("-" * 65)

    for N in sizes:
        assert N % 4 == 0, f"N must be divisible by 4, got {N}"

        A_np = np.random.randn(N).astype(np.float32)
        B_np = np.random.randn(N).astype(np.float32)
        A_cu = torch.from_numpy(A_np).cuda()
        B_cu = torch.from_numpy(B_np).cuda()
        A_tc = torch.from_numpy(A_np)
        B_tc = torch.from_numpy(B_np)

        # ── Correctness check ───────────────────────────────────────────────
        ref = A_cu + B_cu
        naive_out = ext.vec_add_naive(A_cu, B_cu)
        opt_out   = ext.vec_add_opt(A_cu, B_cu)
        assert torch.allclose(naive_out, ref, atol=1e-5), "vec_add_naive result mismatch"
        assert torch.allclose(opt_out,   ref, atol=1e-5), "vec_add_opt result mismatch"

        # ── CPU baselines ────────────────────────────────────────────────────
        if N <= skip_cpu_above:
            numpy_ms     = _time_cpu(lambda: np.add(A_np, B_np, out=np.empty_like(A_np)), warmup, iterations)
            torch_cpu_ms = _time_cpu(lambda: torch.add(A_tc, B_tc), warmup, iterations)
        else:
            # Use the last measured numpy_ms for speedup calculation at larger sizes.
            numpy_ms     = None
            torch_cpu_ms = None

        # ── CUDA naive ──────────────────────────────────────────────────────
        times_naive = _time_cuda(lambda: ext.vec_add_naive(A_cu, B_cu), warmup, iterations)

        # ── CUDA optimized ──────────────────────────────────────────────────
        times_opt = _time_cuda(lambda: ext.vec_add_opt(A_cu, B_cu), warmup, iterations)

        # ── cuBLAS (torch.add on GPU) ────────────────────────────────────────
        times_cublas = _time_cuda(lambda: torch.add(A_cu, B_cu), warmup, iterations)

        # ── Reference speedup base ───────────────────────────────────────────
        base_ms = numpy_ms if numpy_ms is not None else float(times_naive.mean()) * 10

        def _fmt(variant, t, base):
            mean = float(t.mean()) if isinstance(t, np.ndarray) else float(t)
            bw   = _bandwidth(N, mean)
            sp   = base / mean
            print(f"  {variant:<18} {N:>10,} {mean:>10.3f} {bw:>10.2f} {sp:>8.1f}x")
            return mean

        if numpy_ms is not None:
            np_mean = numpy_ms
            print(f"  {'numpy':<18} {N:>10,} {np_mean:>10.3f} {_bandwidth(N, np_mean):>10.2f} {'1.0x':>9}")
            print(f"  {'torch_cpu':<18} {N:>10,} {torch_cpu_ms:>10.3f} {_bandwidth(N, torch_cpu_ms):>10.2f} {np_mean/torch_cpu_ms:>8.1f}x")
            rows.append({
                "variant": "numpy", "N": N,
                "latency_ms_mean": round(np_mean, 4), "latency_ms_std": 0.0,
                "latency_ms_p50": round(np_mean, 4), "latency_ms_p95": round(np_mean, 4),
                "bandwidth_gb_s": round(_bandwidth(N, np_mean), 2), "speedup_vs_numpy": 1.0,
            })
            rows.append({
                "variant": "torch_cpu", "N": N,
                "latency_ms_mean": round(torch_cpu_ms, 4), "latency_ms_std": 0.0,
                "latency_ms_p50": round(torch_cpu_ms, 4), "latency_ms_p95": round(torch_cpu_ms, 4),
                "bandwidth_gb_s": round(_bandwidth(N, torch_cpu_ms), 2),
                "speedup_vs_numpy": round(np_mean / torch_cpu_ms, 2),
            })

        naive_ms  = _fmt("cuda_naive",   times_naive,   base_ms)
        opt_ms    = _fmt("cuda_opt",     times_opt,     base_ms)
        _          = _fmt("torch_add_gpu", times_cublas, base_ms)

        sp_base = numpy_ms if numpy_ms is not None else np.nan
        rows.append(_row("cuda_naive",    N, times_naive,   sp_base if numpy_ms else 1.0))
        rows.append(_row("cuda_opt",      N, times_opt,     sp_base if numpy_ms else 1.0))
        rows.append(_row("torch_add_gpu", N, times_cublas,  sp_base if numpy_ms else 1.0))

    return rows


if __name__ == "__main__":
    import pandas as pd
    rows = benchmark()
    print("\n" + pd.DataFrame(rows).to_string(index=False))
