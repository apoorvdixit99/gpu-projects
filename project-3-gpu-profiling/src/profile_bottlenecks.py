"""CPU/GPU bottleneck analysis using CUDA events and a CPU timer.

Each forward pass is wrapped with both a CUDA event pair (pure GPU kernel time)
and time.perf_counter() (wall time including Python dispatch and kernel launch
overhead).  The difference between the two is the CPU overhead.

Interpretation
--------------
  overlap_pct ≈ 100%  → GPU is saturated; CPU launches faster than GPU executes
  overlap_pct < ~60%  → CPU dispatch is the bottleneck; GPU is idle between kernels
  bottleneck = "GPU"  → cpu_overhead < 50% of gpu_time (GPU is the limiting factor)
  bottleneck = "CPU"  → cpu_overhead ≥ 50% of gpu_time (Python/dispatch is limiting)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).parent.parent


def _load_model() -> torch.nn.Module:
    print("Loading GPT-2 …")
    return GPT2LMHeadModel.from_pretrained("gpt2").eval().half().cuda()


def _make_ids(batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids  = torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long, device="cuda")
    mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")
    return ids, mask


def profile_bottlenecks(
    batch_sizes: list[int],
    seq_len: int,
    warmup: int = 5,
    iterations: int = 20,
) -> list[dict]:
    """Measure GPU kernel time vs wall time to identify CPU/GPU bottlenecks."""
    print("\n=== Bottleneck Analysis (CUDA events + CPU timer) ===")
    model = _load_model()
    rows: list[dict] = []

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    for bs in batch_sizes:
        ids, mask = _make_ids(bs, seq_len)

        with torch.no_grad():
            for _ in range(warmup):
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        torch.cuda.synchronize()

        gpu_times_ms:  list[float] = []
        wall_times_ms: list[float] = []

        with torch.no_grad():
            for _ in range(iterations):
                t0 = time.perf_counter()
                start_ev.record()
                model(input_ids=ids, attention_mask=mask, use_cache=False)
                end_ev.record()
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                gpu_times_ms.append(start_ev.elapsed_time(end_ev))
                wall_times_ms.append((t1 - t0) * 1e3)

        gpu_arr  = np.array(gpu_times_ms,  dtype=np.float32)
        wall_arr = np.array(wall_times_ms, dtype=np.float32)

        gpu_mean     = float(gpu_arr.mean())
        wall_mean    = float(wall_arr.mean())
        cpu_overhead = max(0.0, wall_mean - gpu_mean)
        overlap_pct  = max(0.0, 100.0 * (1.0 - cpu_overhead / wall_mean)) if wall_mean > 0 else 0.0
        throughput   = (bs * seq_len) / (gpu_mean / 1e3)
        bottleneck   = "GPU" if cpu_overhead < 0.5 * gpu_mean else "CPU"

        rows.append({
            "batch_size":             bs,
            "seq_len":                seq_len,
            "gpu_time_ms":            round(gpu_mean,      3),
            "wall_time_ms":           round(wall_mean,     3),
            "cpu_overhead_ms":        round(cpu_overhead,  3),
            "overlap_pct":            round(overlap_pct,   1),
            "throughput_tok_per_sec": round(throughput),
            "bottleneck":             bottleneck,
        })

        print(
            f"  bs={bs:>3}  seq={seq_len}"
            f" | GPU {gpu_mean:>7.2f} ms"
            f" | wall {wall_mean:>7.2f} ms"
            f" | overhead {cpu_overhead:>6.2f} ms"
            f" | overlap {overlap_pct:>5.1f}%"
            f" | {bottleneck}-bound"
        )

    return rows


if __name__ == "__main__":
    profile_bottlenecks(
        batch_sizes=[1, 2, 4, 8, 16, 32],
        seq_len=128,
        warmup=5,
        iterations=20,
    )
