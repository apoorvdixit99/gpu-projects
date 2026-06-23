"""Kernel-level profiling with torch.profiler.

Uses a schedule-based profile (wait=1, warmup=1, active=3) so the profiler
reaches steady state before recording, matching the model's real kernel distribution.

Produces per batch size:
  - Chrome/Perfetto trace JSON → results/traces/   (view at ui.perfetto.dev)
  - Top-10 kernel text report  → results/reports/  (sorted by CUDA time)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.profiler as tp
from torch.profiler import ProfilerActivity
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).parent.parent


def _cuda_us(e: object) -> float:
    """Return CUDA time in microseconds for a FunctionEventAvg.

    In PyTorch 2.x the attribute was renamed from cuda_time_total to
    device_time_total on some builds.  Try both names so the code works
    across versions.
    """
    v = getattr(e, "cuda_time_total", None)
    if v is None:
        v = getattr(e, "device_time_total", 0)
    return float(v)


def _load_model() -> torch.nn.Module:
    print("Loading GPT-2 …")
    return GPT2LMHeadModel.from_pretrained("gpt2").eval().half().cuda()


def _make_ids(batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids  = torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long, device="cuda")
    mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")
    return ids, mask


def profile_kernels(
    batch_sizes: list[int],
    seq_len: int,
    traces_dir: Path,
    reports_dir: Path,
    warmup: int = 5,
) -> list[dict]:
    """Profile each batch size; export Chrome traces and top-kernel reports."""
    print("\n=== Kernel Profiling (torch.profiler) ===")
    model = _load_model()
    rows: list[dict] = []

    for bs in batch_sizes:
        ids, mask = _make_ids(bs, seq_len)

        with torch.no_grad():
            for _ in range(warmup):
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        torch.cuda.synchronize()

        # Schedule: wait=1 (skip), warmup=1 (observer stabilise), active=3 (record).
        # Must call prof.step() exactly wait + warmup + active = 5 times.
        schedule    = tp.schedule(wait=1, warmup=1, active=3, repeat=1)
        trace_path  = traces_dir / f"trace_bs{bs}_seq{seq_len}.json"

        with tp.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with torch.no_grad():
                for _ in range(5):
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
                    prof.step()

        prof.export_chrome_trace(str(trace_path))

        averages = prof.key_averages()
        # ProfilerStep* is a synthetic profiler-internal marker, not a real kernel.
        real_events   = [e for e in averages if not e.key.startswith("ProfilerStep")]
        total_cuda_us = sum(_cuda_us(e) for e in real_events)
        top10 = sorted(real_events, key=_cuda_us, reverse=True)[:10]

        lines = [
            f"batch_size={bs}  seq_len={seq_len}",
            f"{'Kernel':<55} {'CUDA ms':>10} {'CPU ms':>10} {'Calls':>7} {'%CUDA':>7}",
            "-" * 91,
        ]
        for e in top10:
            pct = 100.0 * _cuda_us(e) / total_cuda_us if total_cuda_us > 0 else 0.0
            lines.append(
                f"  {e.key:<53} {_cuda_us(e)/1e3:>10.3f}"
                f" {e.cpu_time_total/1e3:>10.3f} {e.count:>7} {pct:>6.1f}%"
            )
        (reports_dir / f"kernels_bs{bs}_seq{seq_len}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        top1 = top10[0]
        pct1 = 100.0 * _cuda_us(top1) / total_cuda_us if total_cuda_us > 0 else 0.0
        print(
            f"  bs={bs:>3}  seq={seq_len}"
            f" | top kernel: {top1.key[:38]:<38}"
            f" | {_cuda_us(top1)/1e3:>7.3f} ms ({pct1:.1f}%)"
        )

        for e in top10:
            pct = 100.0 * _cuda_us(e) / total_cuda_us if total_cuda_us > 0 else 0.0
            rows.append({
                "batch_size":    bs,
                "seq_len":       seq_len,
                "kernel":        e.key,
                "cuda_time_ms":  round(_cuda_us(e) / 1e3, 4),
                "cpu_time_ms":   round(e.cpu_time_total  / 1e3, 4),
                "calls":         e.count,
                "cuda_time_pct": round(pct, 2),
            })

    print(f"  Traces  → results/traces/")
    print(f"  Reports → results/reports/")
    return rows


if __name__ == "__main__":
    for d in (ROOT / "results" / "traces", ROOT / "results" / "reports"):
        d.mkdir(parents=True, exist_ok=True)
    profile_kernels(
        batch_sizes=[1, 2, 4, 8, 16, 32],
        seq_len=128,
        traces_dir=ROOT / "results" / "traces",
        reports_dir=ROOT / "results" / "reports",
        warmup=5,
    )
