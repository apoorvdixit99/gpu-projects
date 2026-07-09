"""Latency / throughput / peak-memory benchmark for a Lag-Llama predictor.

Unlike Projects 1 and 4 (pure GPT-2 forward pass, timed with CUDA events),
Lag-Llama's zero-shot inference is a full gluonts `predictor.predict()` call:
data transformation -> lag/context construction -> autoregressive sampling
of `num_parallel_samples` trajectories. That whole pipeline is what a real
forecasting workload pays for, so timing is done wall-clock
(`time.perf_counter`) around the full call rather than isolating a single
forward pass.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from load_model import build_predictor


def _time_predict(predictor, series: list, warmup: int, iterations: int) -> np.ndarray:
    for _ in range(warmup):
        list(predictor.predict(series))
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        list(predictor.predict(series))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1_000)  # ms

    return np.array(times, dtype=np.float64)


def benchmark(
    precision: str,
    test_series: list,
    context_lengths: list[int] = [32, 64, 128],
    prediction_length: int = 24,
    warmup: int = 1,
    iterations: int = 5,
) -> list[dict]:
    results = []
    for context_length in context_lengths:
        predictor, _ = build_predictor(
            precision=precision,
            context_length=context_length,
            prediction_length=prediction_length,
        )

        for _ in range(warmup):
            list(predictor.predict(test_series))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        list(predictor.predict(test_series))
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

        times = _time_predict(predictor, test_series, warmup=0, iterations=iterations)
        mean_ms = float(times.mean())
        throughput = len(test_series) / (mean_ms / 1_000)

        row = {
            "backend": f"lagllama_{precision}",
            "context_length": context_length,
            "prediction_length": prediction_length,
            "num_series": len(test_series),
            "latency_ms_mean": round(mean_ms, 2),
            "latency_ms_std": round(float(times.std()), 2),
            "latency_ms_p50": round(float(np.percentile(times, 50)), 2),
            "throughput_series_per_sec": round(throughput, 3),
            "gpu_memory_mb": round(peak_mem_mb, 1),
        }
        results.append(row)
        print(
            f"  [{precision}] ctx={context_length:4d} | "
            f"{mean_ms:8.1f} ms ±{times.std():.1f} | "
            f"{throughput:6.2f} series/s | "
            f"{peak_mem_mb:.0f} MB"
        )

        del predictor
        torch.cuda.empty_cache()

    return results
