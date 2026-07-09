"""4-bit weight-only quantization for Lag-Llama, via bitsandbytes (NF4) or torchao (int4).

Lag-Llama is not a HuggingFace `transformers` model, so there is no
`load_in_4bit=True` shortcut for either library. `bitsandbytes.nn.Linear4bit`
is a drop-in replacement for `torch.nn.Linear` that stores weights in 4-bit
NormalFloat (NF4) and dequantizes them on the fly during the forward matmul,
so any `nn.Linear` submodule -- regardless of which model it lives in -- can
be swapped in place. `torchao.quantization.quantize_` takes the same
model-agnostic approach but rewrites the `nn.Linear.weight` tensor in place
as a quantized tensor subclass rather than swapping the module class.
"""

from __future__ import annotations

import bitsandbytes as bnb
import torch
import torch.nn as nn
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
    Int4PackingFormat,
)


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


def quantize_int4_ao(module: nn.Module, group_size: int = 32) -> nn.Module:
    """Quantize eligible `nn.Linear` weights to int4 in place via torchao.

    The `_weight_int4pack_mm` CUDA kernel torchao dispatches to only accepts
    `group_size` in {32, 64, 128, 256} -- torchao's own Python-level shape
    check (`in_features % group_size == 0`) is necessary but not sufficient;
    passing a group_size outside that set raises at the *first forward call*,
    not at quantize_ time. Lag-Llama's transformer width here is 144
    (n_head=9 x n_embd_per_head=16, not a power of 2), and none of the four
    valid group sizes divide 144 -- so of the model's linear layers, only
    `mlp.c_proj` (in_features=512) is actually eligible; every 144-wide
    projection (q_proj, kv_proj, attn.c_proj, c_fc1, c_fc2) and `wte`
    (in_features=92) are silently left unquantized by `quantize_` (logged,
    not raised). This is a real architecture/kernel mismatch, not a bug: NF4
    via bitsandbytes has no such shape restriction and quantizes all of them
    (see replace_linear_with_nf4 above). The distribution-head output layers
    (`out_features=1`) are excluded outright regardless: too few output rows
    for the tile packing.
    """

    def _filter(m: nn.Module, _name: str) -> bool:
        return (
            isinstance(m, nn.Linear)
            and m.out_features >= 8
            and m.in_features % group_size == 0
        )

    config = Int4WeightOnlyConfig(
        group_size=group_size, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D
    )
    quantize_(module, config, filter_fn=_filter)
    return module
