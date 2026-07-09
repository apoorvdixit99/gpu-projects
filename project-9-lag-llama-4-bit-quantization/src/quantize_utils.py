"""NF4 weight-only quantization for Lag-Llama via bitsandbytes.

Lag-Llama is not a HuggingFace `transformers` model, so there is no
`load_in_4bit=True` shortcut. `bitsandbytes.nn.Linear4bit` is a drop-in
replacement for `torch.nn.Linear` that stores weights in 4-bit NormalFloat
(NF4) and dequantizes them on the fly during the forward matmul, so any
`nn.Linear` submodule -- regardless of which model it lives in -- can be
swapped in place.
"""

from __future__ import annotations

import bitsandbytes as bnb
import torch
import torch.nn as nn


def replace_linear_with_nf4(module: nn.Module, compute_dtype: torch.dtype = torch.float32) -> nn.Module:
    """Recursively replace every `nn.Linear` in `module` with `bnb.nn.Linear4bit` (NF4).

    Must be called on a CPU-resident module; quantization happens the moment
    the returned module (or the replaced submodules) is moved `.to("cuda")`.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            nf4_layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                quant_type="nf4",
            )
            nf4_layer.weight = bnb.nn.Params4bit(
                child.weight.data.clone(),
                requires_grad=False,
                quant_type="nf4",
            )
            if child.bias is not None:
                nf4_layer.bias = nn.Parameter(child.bias.data.clone(), requires_grad=False)
            setattr(module, name, nf4_layer)
        else:
            replace_linear_with_nf4(child, compute_dtype)
    return module
