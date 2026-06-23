"""Latency, throughput, and MFU measurement via CUDA events.

For each model × batch size:
  - CUDA FP16 inference timed with CUDA event pairs (start/end)
  - Warmup iterations excluded from timing
  - MFU = (batch_size × flops_per_sample) / (latency_s × peak_fp16_flops)

MFU (Model FLOP Utilization) shows how much of the GPU's theoretical FP16
throughput is actually being used.  Values near 100% mean near-peak efficiency.
"""
from __future__ import annotations

import numpy as np
import torch

from models import ModelSpec


def measure_latency(
    specs: list[ModelSpec],
    batch_sizes: list[int],
    warmup: int,
    iterations: int,
    peak_fp16_tflops: float,
    flops_per_sample: dict[str, int],
) -> list[dict]:
    """Return per-model per-batch-size latency and throughput rows."""
    print("\n=== Latency, Throughput & MFU (CUDA events, FP16) ===")

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    peak_flops_per_s = peak_fp16_tflops * 1e12

    rows: list[dict] = []

    for spec in specs:
        print(f"\n  {spec.label}")
        model = spec.load(cuda=True, fp16=True)
        fps   = flops_per_sample.get(spec.name, 0)

        for bs in batch_sizes:
            try:
                raw_inputs  = spec.make_inputs(bs)
                model_dtype = next(model.parameters()).dtype
                inputs = {
                    k: v.to(model_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                    for k, v in raw_inputs.items()
                }

                with torch.no_grad():
                    for _ in range(warmup):
                        model(**inputs)
                torch.cuda.synchronize()

                times: list[float] = []
                with torch.no_grad():
                    for _ in range(iterations):
                        start_ev.record()
                        model(**inputs)
                        end_ev.record()
                        torch.cuda.synchronize()
                        times.append(start_ev.elapsed_time(end_ev))

            except torch.cuda.OutOfMemoryError:
                print(f"    bs={bs:>3} | OOM — skipped")
                torch.cuda.empty_cache()
                continue

            arr      = np.array(times, dtype=np.float32)
            mean_ms  = float(arr.mean())
            lat_s    = mean_ms / 1_000.0
            tput     = (bs * spec.throughput_scale) / lat_s

            mfu_pct = 0.0
            if fps > 0 and peak_flops_per_s > 0:
                mfu_pct = 100.0 * (bs * fps) / (lat_s * peak_flops_per_s)

            rows.append({
                "model":                spec.name,
                "label":               spec.label,
                "modality":            spec.modality,
                "batch_size":          bs,
                "latency_ms_mean":     round(mean_ms,                    3),
                "latency_ms_std":      round(float(arr.std()),           3),
                "latency_ms_p50":      round(float(np.percentile(arr, 50)), 3),
                "latency_ms_p95":      round(float(np.percentile(arr, 95)), 3),
                "latency_ms_p99":      round(float(np.percentile(arr, 99)), 3),
                "throughput":          round(tput),
                "throughput_unit":     spec.throughput_unit,
                "mfu_pct":             round(mfu_pct, 2),
            })

            print(
                f"    bs={bs:>3}"
                f" | {mean_ms:>7.2f} ms ±{arr.std():.2f}"
                f" | {tput:>9,.0f} {spec.throughput_unit}"
                f" | MFU {mfu_pct:.1f}%"
            )

        del model
        torch.cuda.empty_cache()

    return rows


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from models import MODELS
    rows = measure_latency(
        MODELS,
        batch_sizes=[1, 4, 8, 16, 32],
        warmup=10,
        iterations=50,
        peak_fp16_tflops=121.9,
        flops_per_sample={},
    )
