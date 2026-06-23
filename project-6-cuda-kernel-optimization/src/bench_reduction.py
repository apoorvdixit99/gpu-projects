"""Parallel reduction benchmark: CPU vs naive / sequential / warp-shuffle CUDA.

What is timed
-------------
  CPU variants:  numpy.sum() / torch.sum() measured with time.perf_counter()
  CUDA variants: the kernel pass only (CUDA events), NOT the subsequent
                 host-side summation of partial results.  This isolates the
                 reduction algorithm from transfer overhead.

Metrics
-------
  bandwidth_gb_s   N * 4 bytes / latency_s  (each element read once)
  speedup_vs_numpy latency_numpy / latency_kernel
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
    return (time.perf_counter() - t0) / iters * 1e3


def _bandwidth(N: int, latency_ms: float) -> float:
    return N * 4 / (latency_ms / 1e3) / 1e9  # read once


def benchmark(
    sizes:          list[int] = [2**20, 2**22, 2**24, 2**26, 2**28],
    warmup:         int       = 10,
    iterations:     int       = 100,
    skip_cpu_above: int       = 2**26,
) -> list[dict]:
    """Benchmark parallel reduction across all variants and sizes."""
    ext  = get_ext()
    rows: list[dict] = []

    print("\n=== Parallel Reduction Benchmark ===")
    print(f"{'Variant':<22} {'N':>10} {'mean ms':>10} {'GB/s':>9} {'speedup':>9}")
    print("-" * 66)

    for N in sizes:
        data_np = np.random.randn(N).astype(np.float32)
        data_cu = torch.from_numpy(data_np).cuda()

        # ── Correctness check ──────────────────────────────────────────────
        expected = float(data_cu.sum().item())
        for name, fn in [("reduce_naive",      ext.reduce_naive),
                         ("reduce_sequential", ext.reduce_sequential),
                         ("reduce_shuffle",    ext.reduce_shuffle)]:
            partial = fn(data_cu)
            result  = float(partial.sum().item())
            rel_err = abs(result - expected) / (abs(expected) + 1e-8)
            assert rel_err < 1e-3, f"{name} relative error {rel_err:.2e} at N={N}"

        # ── CPU baselines ──────────────────────────────────────────────────
        if N <= skip_cpu_above:
            data_tc      = torch.from_numpy(data_np)
            numpy_ms     = _time_cpu(lambda: np.sum(data_np), warmup, iterations)
            torch_cpu_ms = _time_cpu(lambda: data_tc.sum().item(), warmup, iterations)
        else:
            numpy_ms     = None
            torch_cpu_ms = None

        # ── CUDA kernels ───────────────────────────────────────────────────
        times = {
            "reduce_naive":      _time_cuda(lambda: ext.reduce_naive(data_cu),      warmup, iterations),
            "reduce_sequential": _time_cuda(lambda: ext.reduce_sequential(data_cu), warmup, iterations),
            "reduce_shuffle":    _time_cuda(lambda: ext.reduce_shuffle(data_cu),    warmup, iterations),
        }

        base_ms = numpy_ms if numpy_ms is not None else float(times["reduce_naive"].mean())

        def _print(variant, ms, base):
            bw = _bandwidth(N, ms)
            sp = base / ms
            print(f"  {variant:<20} {N:>10,} {ms:>10.4f} {bw:>9.2f} {sp:>8.1f}x")

        if numpy_ms is not None:
            _print("numpy",     numpy_ms,     numpy_ms)
            _print("torch_cpu", torch_cpu_ms, numpy_ms)
            rows += [
                {"variant": "numpy",     "N": N, "latency_ms_mean": round(numpy_ms, 4),
                 "latency_ms_std": 0.0, "bandwidth_gb_s": round(_bandwidth(N, numpy_ms), 2),
                 "speedup_vs_numpy": 1.0},
                {"variant": "torch_cpu", "N": N, "latency_ms_mean": round(torch_cpu_ms, 4),
                 "latency_ms_std": 0.0, "bandwidth_gb_s": round(_bandwidth(N, torch_cpu_ms), 2),
                 "speedup_vs_numpy": round(numpy_ms / torch_cpu_ms, 2)},
            ]

        for variant, t in times.items():
            mean_ms = float(t.mean())
            _print(variant, mean_ms, base_ms)
            rows.append({
                "variant":          variant,
                "N":                N,
                "latency_ms_mean":  round(mean_ms, 4),
                "latency_ms_std":   round(float(t.std()), 4),
                "latency_ms_p50":   round(float(np.percentile(t, 50)), 4),
                "latency_ms_p95":   round(float(np.percentile(t, 95)), 4),
                "bandwidth_gb_s":   round(_bandwidth(N, mean_ms), 2),
                "speedup_vs_numpy": round(base_ms / mean_ms, 2) if numpy_ms is not None else float("nan"),
            })

    return rows


if __name__ == "__main__":
    import pandas as pd
    rows = benchmark()
    print("\n" + pd.DataFrame(rows).to_string(index=False))
