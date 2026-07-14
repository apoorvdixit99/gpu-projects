"""Export GPT-2 to ONNX with dynamic batch and sequence-length axes."""

import os
from pathlib import Path

import onnx
import torch
from torch.export import Dim
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


class _GPT2Wrapper(torch.nn.Module):
    """Strips the HuggingFace output dataclass so ONNX export sees a plain tensor."""

    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits


def export(output_path: str | None = None, opset: int = 18, fp16: bool = False) -> str:
    """Download GPT-2, wrap it, and export to ONNX. Returns the saved file path."""
    if output_path is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tag = "fp16" if fp16 else "fp32"
        output_path = str(MODELS_DIR / f"gpt2_{tag}.onnx")

    print(f"Loading GPT-2 (gpt2) from HuggingFace [{'FP16' if fp16 else 'FP32'}] …")
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    if fp16:
        base = base.half()
    model = _GPT2Wrapper(base).cuda().eval()

    dummy_ids = torch.randint(0, 50257, (1, 128), dtype=torch.long, device="cuda")
    dummy_mask = torch.ones(1, 128, dtype=torch.long, device="cuda")

    print(f"Exporting to {output_path} (opset {opset}) …")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            output_path,
            opset_version=opset,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_shapes={
                # Separate Dim instances per tensor — sharing the same instance
                # causes the exporter to warn that the second axis name is unused.
                "input_ids":      {0: Dim("batch_size", min=1, max=64), 1: Dim("seq_len", min=1, max=1024)},
                "attention_mask": {0: Dim("batch_size", min=1, max=64), 1: Dim("seq_len", min=1, max=1024)},
            },
            do_constant_folding=True,
        )

    onnx.checker.check_model(onnx.load(output_path))
    size_mb = os.path.getsize(output_path) / 1024 ** 2
    print(f"ONNX model verified — {size_mb:.1f} MB saved at {output_path}")
    return output_path


if __name__ == "__main__":
    export()
