"""Model registry — specs and factory functions for all benchmark models.

Each ModelSpec knows how to:
  - load itself (CPU FP32 for FLOPs counting; CUDA FP16 for inference)
  - produce dummy inputs for a given batch size and device

analytical_macs
---------------
torchinfo 1.8.0 undercounts MACs for nn.Linear when the input is 3D (batch, seq,
features): it computes out_features × in_features instead of seq × out_features ×
in_features, giving ~seq_len× too few MACs for transformer models.  CNNs (ResNet-50)
use nn.Conv2d and are not affected.  Where set, analytical_macs bypasses torchinfo.
See ISSUES.md #2 for the full investigation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

SEQ_LEN    = 128
IMAGE_SIZE = 224


@dataclass
class ModelSpec:
    name: str
    label: str
    modality: str     # "nlp" | "vision"
    seq_len: int      # tokens for NLP; 0 for vision
    image_size: int   # pixels for vision; 0 for NLP

    loader_fn: Callable[[], nn.Module] = field(repr=False)
    inputs_fn: Callable[[int, str], dict[str, torch.Tensor]] = field(repr=False)

    # Analytical MACs per sample (bs=1).  None → use torchinfo.
    analytical_macs: int | None = field(default=None)

    def load(self, *, cuda: bool = True, fp16: bool = True) -> nn.Module:
        model = self.loader_fn().eval()
        if cuda:
            model = model.cuda()
        if fp16:
            model = model.half()
        return model

    def make_inputs(self, batch_size: int, device: str = "cuda") -> dict[str, torch.Tensor]:
        return self.inputs_fn(batch_size, device)

    @property
    def throughput_unit(self) -> str:
        return "tok/s" if self.modality == "nlp" else "img/s"

    @property
    def throughput_scale(self) -> int:
        """Tokens or images produced per forward pass."""
        return self.seq_len if self.modality == "nlp" else 1


# ── Analytical FLOPs helper ──────────────────────────────────────────────────

def _transformer_macs(
    seq_len: int,
    hidden: int,
    ffn_dim: int,
    num_layers: int,
    vocab_size: int | None = None,
    has_pooler: bool = False,
) -> int:
    """Analytical MAC count for one transformer forward pass (batch_size=1).

    Per layer:
      QKV projections : 3 × seq × hidden²
      Attention scores : seq² × hidden
      Attention × value: seq² × hidden
      Output projection: seq × hidden²
      FFN (2 linear)   : 2 × seq × hidden × ffn_dim
    Optional:
      LM head  : seq × hidden × vocab_size
      Pooler   : hidden × hidden  (one dense on the CLS token)
    """
    per_layer = (
        4 * seq_len * hidden * hidden          # QKV + O
        + 2 * seq_len * seq_len * hidden       # attn scores + weighted sum
        + 2 * seq_len * hidden * ffn_dim       # FFN1 + FFN2
    )
    total = num_layers * per_layer
    if vocab_size is not None:
        total += seq_len * hidden * vocab_size
    if has_pooler:
        total += hidden * hidden
    return total


# ── Input factories ──────────────────────────────────────────────────────────

def _gpt2_inputs(seq_len: int, vocab_size: int = 50257):
    def fn(batch_size: int, device: str) -> dict:
        ids  = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        return {"input_ids": ids, "attention_mask": mask, "use_cache": False}
    return fn


def _bert_inputs(seq_len: int, vocab_size: int = 30522):
    def fn(batch_size: int, device: str) -> dict:
        ids            = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
        mask           = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        return {"input_ids": ids, "attention_mask": mask, "token_type_ids": token_type_ids}
    return fn


def _vision_inputs(image_size: int):
    def fn(batch_size: int, device: str) -> dict:
        return {"x": torch.randn(batch_size, 3, image_size, image_size, device=device)}
    return fn


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_gpt2() -> nn.Module:
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained("gpt2")


def _load_distilgpt2() -> nn.Module:
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained("distilgpt2")


def _load_bert_base() -> nn.Module:
    from transformers import BertModel
    # eager: force non-fused attention so CUDA-event timing sees the same ops
    # that the analytical FLOPs formula accounts for (see ISSUES.md #2).
    return BertModel.from_pretrained("bert-base-uncased", attn_implementation="eager")


def _load_resnet50() -> nn.Module:
    import torchvision.models as tvm
    try:
        return tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
    except AttributeError:
        return tvm.resnet50(pretrained=True)  # torchvision < 0.13


# ── Registry ─────────────────────────────────────────────────────────────────

MODELS: list[ModelSpec] = [
    ModelSpec(
        name="gpt2",
        label="GPT-2 (124M)",
        modality="nlp",
        seq_len=SEQ_LEN,
        image_size=0,
        loader_fn=_load_gpt2,
        inputs_fn=_gpt2_inputs(SEQ_LEN),
        # GPT-2: 12 layers (h=768, f=3072) + LM head (vocab=50257)
        analytical_macs=_transformer_macs(SEQ_LEN, 768, 3072, 12, vocab_size=50257),
    ),
    ModelSpec(
        name="distilgpt2",
        label="DistilGPT-2 (82M)",
        modality="nlp",
        seq_len=SEQ_LEN,
        image_size=0,
        loader_fn=_load_distilgpt2,
        inputs_fn=_gpt2_inputs(SEQ_LEN),
        # DistilGPT-2: 6 layers (h=768, f=3072) + LM head (vocab=50257)
        analytical_macs=_transformer_macs(SEQ_LEN, 768, 3072, 6, vocab_size=50257),
    ),
    ModelSpec(
        name="bert_base",
        label="BERT-base (110M)",
        modality="nlp",
        seq_len=SEQ_LEN,
        image_size=0,
        loader_fn=_load_bert_base,
        inputs_fn=_bert_inputs(SEQ_LEN),
        # BERT-base: 12 layers (h=768, f=3072) + pooler dense
        analytical_macs=_transformer_macs(SEQ_LEN, 768, 3072, 12, has_pooler=True),
    ),
    ModelSpec(
        name="resnet50",
        label="ResNet-50 (25M)",
        modality="vision",
        seq_len=0,
        image_size=IMAGE_SIZE,
        loader_fn=_load_resnet50,
        inputs_fn=_vision_inputs(IMAGE_SIZE),
        # ResNet-50: torchinfo is accurate for nn.Conv2d — no override needed
        analytical_macs=None,
    ),
]

MODEL_MAP: dict[str, ModelSpec] = {m.name: m for m in MODELS}
