"""PyTorch FP16 inference benchmark for GPT-2."""

from __future__ import annotations

import numpy as np
import torch
from transformers import GPT2LMHeadModel


def _time_fn(fn, warmup: int, iterations: int) -> np.ndarray:
    """Return per-iteration GPU latencies (ms) measured with CUDA events."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    return np.array(times, dtype=np.float32)


def benchmark(
    batch_sizes: list[int] = [1, 4, 8, 16],
    seq_lens: list[int] = [64, 128, 256],
    fp16: bool = True,
    warmup: int = 10,
    iterations: int = 100,
) -> list[dict]:
    device = torch.device("cuda")
    tag = "pytorch_fp16" if fp16 else "pytorch_fp32"
    print(f"Loading GPT-2 [{tag}] …")

    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
    if fp16:
        model = model.half()

    results = []

    for bs in batch_sizes:
        for seq_len in seq_lens:
            ids = torch.randint(0, 50257, (bs, seq_len), dtype=torch.long, device=device)
            mask = torch.ones(bs, seq_len, dtype=torch.long, device=device)

            def fn():
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=mask, use_cache=False)

            # Warm the model into cache, then measure peak memory over one pass.
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

            times = _time_fn(fn, warmup, iterations)
            mean_ms = float(times.mean())
            throughput = (bs * seq_len) / (mean_ms / 1_000)

            row = {
                "backend": tag,
                "batch_size": bs,
                "seq_len": seq_len,
                "latency_ms_mean": round(mean_ms, 3),
                "latency_ms_std": round(float(times.std()), 3),
                "latency_ms_p50": round(float(np.percentile(times, 50)), 3),
                "latency_ms_p95": round(float(np.percentile(times, 95)), 3),
                "latency_ms_p99": round(float(np.percentile(times, 99)), 3),
                "throughput_tok_per_sec": round(throughput),
                "gpu_memory_mb": round(peak_mem_mb, 1),
            }
            results.append(row)
            print(
                f"  bs={bs:2d}  seq={seq_len:3d} | "
                f"{mean_ms:7.2f} ms ±{times.std():.2f} | "
                f"{throughput:>8,.0f} tok/s | "
                f"{peak_mem_mb:.0f} MB"
            )

    return results
