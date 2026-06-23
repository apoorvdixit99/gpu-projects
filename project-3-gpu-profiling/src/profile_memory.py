"""GPU memory analysis across batch sizes.

Tracks per-batch-size:
  peak_allocated_mb    — high-water mark of memory in-use by tensors
  peak_reserved_mb     — high-water mark of memory held by the CUDA allocator
                         (includes empty cached blocks; always >= allocated)
  current_allocated_mb — memory still in-use after the forward pass completes
  current_reserved_mb  — memory the allocator still holds after the forward pass
  fragmentation_pct    — (reserved - allocated) / reserved; the allocator cache gap

The gap between allocated and reserved is not wasted — the allocator keeps free
blocks cached to avoid expensive cudaMalloc calls on the next forward pass.
"""

from __future__ import annotations

from pathlib import Path

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


def profile_memory(
    batch_sizes: list[int],
    seq_len: int,
    warmup: int = 5,
) -> list[dict]:
    """Measure GPU memory usage for each batch size."""
    print("\n=== Memory Profiling ===")
    model = _load_model()
    rows: list[dict] = []

    for bs in batch_sizes:
        ids, mask = _make_ids(bs, seq_len)

        # Warm the allocator into its steady-state cache before measuring.
        # reset_peak_memory_stats() is called *after* warmup so the peak
        # reflects only the single measurement pass, not the warmup runs.
        with torch.no_grad():
            for _ in range(warmup):
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        torch.cuda.synchronize()

        peak_alloc_mb       = torch.cuda.max_memory_allocated() / 1024 ** 2
        peak_reserved_mb    = torch.cuda.max_memory_reserved()  / 1024 ** 2
        current_alloc_mb    = torch.cuda.memory_allocated()     / 1024 ** 2
        current_reserved_mb = torch.cuda.memory_reserved()      / 1024 ** 2
        fragmentation_pct   = (
            100.0 * (peak_reserved_mb - current_alloc_mb) / peak_reserved_mb
            if peak_reserved_mb > 0 else 0.0
        )

        rows.append({
            "batch_size":            bs,
            "seq_len":               seq_len,
            "peak_allocated_mb":     round(peak_alloc_mb,       1),
            "peak_reserved_mb":      round(peak_reserved_mb,    1),
            "current_allocated_mb":  round(current_alloc_mb,    1),
            "current_reserved_mb":   round(current_reserved_mb, 1),
            "fragmentation_pct":     round(fragmentation_pct,   1),
        })

        print(
            f"  bs={bs:>3}  seq={seq_len}"
            f" | peak alloc {peak_alloc_mb:>7.1f} MB"
            f" | peak reserved {peak_reserved_mb:>7.1f} MB"
            f" | frag {fragmentation_pct:>5.1f}%"
        )

    return rows


if __name__ == "__main__":
    profile_memory(
        batch_sizes=[1, 2, 4, 8, 16, 32],
        seq_len=128,
        warmup=5,
    )
