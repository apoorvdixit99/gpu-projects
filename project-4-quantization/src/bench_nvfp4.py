"""NVFP4 weight-only quantization benchmark via torchao.

Note: NVFP4 is a Blackwell-native format (SM100+). On Ada Lovelace this runs
through a software emulation path -- dequantize-then-compute in FP32. Results
reflect the overhead of that software path, not Blackwell hardware performance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
from torchao.quantization import quantize_

_BACKEND = "torchao_nvfp4"


def _conv1d_to_linear(model: nn.Module) -> None:
    """Replace HuggingFace Conv1D with nn.Linear so torchao's quantize_ finds the layers.

    Conv1D weight shape is (in_features, out_features); nn.Linear is (out_features, in_features),
    so we transpose. The forward pass is mathematically identical after the swap.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, Conv1D):
            in_f, out_f = child.weight.shape
            lin = nn.Linear(in_f, out_f, bias=True)
            lin.weight = nn.Parameter(child.weight.T.contiguous())
            lin.bias   = nn.Parameter(child.bias.clone())
            setattr(model, name, lin)
        else:
            _conv1d_to_linear(child)


def _time_fn(fn, warmup: int, iterations: int) -> np.ndarray:
    """Return per-iteration GPU latencies (ms) measured with CUDA events."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

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


def _nvfp4_filter(module: nn.Module, fqn: str) -> bool:
    """Skip layers whose weight dims are not both divisible by 16.

    NVFP4 kernels require the last two weight dimensions to be multiples of 16.
    GPT-2's lm_head has shape (50257, 768) — 50257 is not divisible by 16, so
    it must be left in FP32.
    """
    if not isinstance(module, nn.Linear):
        return False
    out_f, in_f = module.weight.shape
    return (out_f % 16 == 0) and (in_f % 16 == 0)


def load_model(device: torch.device | None = None) -> GPT2LMHeadModel:
    device = device or torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
    _conv1d_to_linear(model)
    quantize_(model, NVFP4WeightOnlyConfig(), filter_fn=_nvfp4_filter)
    return model


def benchmark(
    batch_sizes: list[int] = [1, 4, 8, 16],
    seq_lens:    list[int] = [64, 128, 256],
    warmup:      int = 10,
    iterations:  int = 100,
) -> list[dict]:
    device = torch.device("cuda")
    print(f"Loading GPT-2 [{_BACKEND}] (software emulation -- Ada Lovelace) …")
    model = load_model(device)

    results = []
    for bs in batch_sizes:
        for seq_len in seq_lens:
            ids  = torch.randint(0, 50257, (bs, seq_len), dtype=torch.long, device=device)
            mask = torch.ones(bs, seq_len, dtype=torch.long, device=device)

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

            times      = _time_fn(fn, warmup, iterations)
            mean_ms    = float(times.mean())
            throughput = (bs * seq_len) / (mean_ms / 1_000)

            row = {
                "backend":                _BACKEND,
                "batch_size":             bs,
                "seq_len":                seq_len,
                "latency_ms_mean":        round(mean_ms, 3),
                "latency_ms_std":         round(float(times.std()), 3),
                "latency_ms_p50":         round(float(np.percentile(times, 50)), 3),
                "latency_ms_p95":         round(float(np.percentile(times, 95)), 3),
                "latency_ms_p99":         round(float(np.percentile(times, 99)), 3),
                "throughput_tok_per_sec": round(throughput),
                "gpu_memory_mb":          round(peak_mem_mb, 1),
            }
            results.append(row)
            print(
                f"  bs={bs:2d}  seq={seq_len:3d} | "
                f"{mean_ms:7.2f} ms ±{times.std():.2f} | "
                f"{throughput:>8,.0f} tok/s | "
                f"{peak_mem_mb:.0f} MB"
            )

    return results
