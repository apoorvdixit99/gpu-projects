"""PyTorch inference benchmark for GPT-2.

Three variants selected by the *mode* parameter:

  "eager"   – standard autograd, no kernel fusion (FP32 or FP16)
  "sdpa"    – Flash Attention via attn_implementation="sdpa"
  "compile" – CUDA-graph replay via torch.compile(backend="cudagraphs")
              Works on Windows; does not require Triton.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import GPT2LMHeadModel

_MODES = ("eager", "sdpa", "compile")


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
    mode: str = "eager",
) -> list[dict]:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    device = torch.device("cuda")
    precision = "fp16" if fp16 else "fp32"
    prefix = "pytorch" if mode == "eager" else f"pytorch_{mode}"
    tag = f"{prefix}_{precision}"
    print(f"Loading GPT-2 [{tag}] …")

    load_kwargs = {"attn_implementation": "sdpa"} if mode == "sdpa" else {}
    base = GPT2LMHeadModel.from_pretrained("gpt2", **load_kwargs).eval().to(device)
    if fp16:
        base = base.half()

    results = []

    for bs in batch_sizes:
        for seq_len in seq_lens:
            ids  = torch.randint(0, 50257, (bs, seq_len), dtype=torch.long, device=device)
            mask = torch.ones(bs, seq_len, dtype=torch.long, device=device)

            # CUDA graphs capture a static graph for a fixed input shape.
            # Reset dynamo state and create a fresh compiled wrapper per shape
            # so graphs from previous shapes don't carry over.
            if mode == "compile":
                torch._dynamo.reset()
                model = torch.compile(base, backend="cudagraphs")
            else:
                model = base

            def fn():
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=mask, use_cache=False)

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
