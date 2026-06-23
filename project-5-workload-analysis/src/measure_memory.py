"""GPU memory footprint across models and batch sizes.

For each model × batch size:
  - Warm the CUDA allocator with `warmup` forward passes
  - Reset peak memory stats (so warmup allocations are excluded)
  - Single measurement pass
  - Report peak allocated, peak reserved, fragmentation %

peak_allocated  — high-water mark of memory actually in use by tensors
peak_reserved   — high-water mark of memory held by the CUDA allocator
                  (includes empty cached blocks; always >= allocated)
fragmentation   — (reserved - allocated) / reserved; allocator cache gap
"""
from __future__ import annotations

import torch

from models import ModelSpec


def measure_memory(
    specs: list[ModelSpec],
    batch_sizes: list[int],
    warmup: int = 5,
) -> list[dict]:
    """Return per-model per-batch-size GPU memory rows."""
    print("\n=== Memory Footprint (FP16, CUDA) ===")
    rows: list[dict] = []

    for spec in specs:
        print(f"\n  {spec.label}")
        model = spec.load(cuda=True, fp16=True)

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

                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    model(**inputs)
                torch.cuda.synchronize()

            except torch.cuda.OutOfMemoryError:
                print(f"    bs={bs:>3} | OOM — skipped")
                torch.cuda.empty_cache()
                continue

            peak_alloc_mb    = torch.cuda.max_memory_allocated() / 1024 ** 2
            peak_reserved_mb = torch.cuda.max_memory_reserved()  / 1024 ** 2
            frag_pct = (
                100.0 * (peak_reserved_mb - peak_alloc_mb) / peak_reserved_mb
                if peak_reserved_mb > 0 else 0.0
            )

            rows.append({
                "model":              spec.name,
                "label":             spec.label,
                "modality":          spec.modality,
                "batch_size":        bs,
                "peak_allocated_mb": round(peak_alloc_mb,    1),
                "peak_reserved_mb":  round(peak_reserved_mb, 1),
                "fragmentation_pct": round(frag_pct,         1),
            })

            print(
                f"    bs={bs:>3}"
                f" | alloc {peak_alloc_mb:>7.1f} MB"
                f" | reserved {peak_reserved_mb:>7.1f} MB"
                f" | frag {frag_pct:>5.1f}%"
            )

        del model
        torch.cuda.empty_cache()

    return rows


if __name__ == "__main__":
    from models import MODELS
    measure_memory(MODELS, batch_sizes=[1, 4, 8, 16, 32], warmup=5)
