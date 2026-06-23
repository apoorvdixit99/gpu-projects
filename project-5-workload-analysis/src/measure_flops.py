"""FLOPs and parameter count measurement.

For transformer models (GPT-2, DistilGPT-2, BERT-base) the analytical MAC count
from ModelSpec.analytical_macs is used — torchinfo 1.8.0 severely undercounts
3D nn.Linear ops and overcounts Conv1D-based models (see ISSUES.md #2).

For vision models (ResNet-50) torchinfo gives an accurate count via nn.Conv2d.
"""
from __future__ import annotations

import torch
from torchinfo import summary

from models import ModelSpec


def measure_flops(specs: list[ModelSpec]) -> list[dict]:
    """Return per-sample FLOPs and parameter counts for each model."""
    print("\n=== FLOPs & Parameter Count (analytical / torchinfo, CPU FP32, bs=1) ===")
    rows: list[dict] = []

    for spec in specs:
        print(f"  {spec.label} …")
        model = spec.load(cuda=False, fp16=False)

        params = sum(p.numel() for p in model.parameters())

        if spec.analytical_macs is not None:
            macs = spec.analytical_macs
            source = "analytical"
        else:
            raw_inputs    = spec.make_inputs(batch_size=1, device="cpu")
            tensor_inputs = {k: v for k, v in raw_inputs.items() if isinstance(v, torch.Tensor)}
            try:
                stats = summary(model, input_data=tensor_inputs, verbose=0)
                macs  = stats.total_mult_adds
                # torchinfo params double-counts tied weights; use manual count.
                source = "torchinfo"
            except Exception as exc:
                print(f"    WARNING: torchinfo failed ({exc}); param count only")
                macs   = 0
                source = "unavailable"

        flops = 2 * macs
        rows.append({
            "model":    spec.name,
            "label":    spec.label,
            "modality": spec.modality,
            "params_m": round(params / 1e6, 2),
            "macs_g":   round(macs  / 1e9, 3),
            "flops_g":  round(flops / 1e9, 3),
        })

        print(
            f"    params {params/1e6:.1f}M"
            f" | MACs {macs/1e9:.2f}G"
            f" | FLOPs {flops/1e9:.2f}G"
            f"  [{source}]"
        )

        del model

    return rows


if __name__ == "__main__":
    from models import MODELS
    measure_flops(MODELS)
