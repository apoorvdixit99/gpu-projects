"""Build a Lag-Llama zero-shot predictor in FP32 or NF4 (bitsandbytes).

Lag-Llama ships as a single Lightning checkpoint (`lag-llama.ckpt`) whose
`hyper_parameters` blob records the exact architecture it was pretrained
with (n_layer, n_head, lags_seq, ...). We read those back out and feed them
into `LagLlamaEstimator` so the reconstructed model matches the checkpoint
weights -- this is the same pattern used in the project's official Colab
demo (`get_lag_llama_predictions`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
VENDOR_DIR = ROOT / "vendor" / "lag-llama"
CKPT_PATH = ROOT / "checkpoints" / "lag-llama.ckpt"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from lag_llama.gluon.estimator import LagLlamaEstimator  # noqa: E402

from quantize_utils import replace_linear_with_nf4  # noqa: E402

PRECISIONS = ("fp32", "nf4")

# Two problems with loading lag-llama.ckpt directly, both patched here:
#
# 1. PyTorch >=2.6 defaults `torch.load(weights_only=...)` to True, which rejects
#    the gluonts distribution classes pickled into the checkpoint.
#
# 2. lag-llama.ckpt is a full Lightning training checkpoint -- besides the model
#    weights it also carries optimizer state (Adam moment buffers, ~2-3x model
#    size) and other training bookkeeping. Lightning's internal loader
#    (`LagLlamaLightningModule.load_from_checkpoint`, called inside
#    estimator.create_lightning_module()) passes `map_location=self.device`
#    straight through to `torch.load`, so all of that unused training state gets
#    materialized on the GPU too, transiently, every single time a predictor is
#    built. Across the repeated build_predictor() calls in a benchmark sweep,
#    that was enough to exhaust and fragment the 12GB card (see OOM warnings in
#    ISSUES.md). Forcing every load onto CPU here and moving only the actual
#    model to `device` ourselves (see build_predictor below) avoids it.
#
# Neither override is exposed as a passthrough kwarg by Lightning, so both are
# patched at the torch.load source. Safe: the checkpoint is from the official
# time-series-foundation-models repo, and cpu-loading is strictly the more
# conservative choice.
_torch_load = torch.load


def _load_full(*args, **kwargs):
    kwargs["weights_only"] = False
    kwargs["map_location"] = "cpu"
    return _torch_load(*args, **kwargs)


torch.load = _load_full


def build_predictor(
    precision: str,
    context_length: int = 32,
    prediction_length: int = 24,
    num_parallel_samples: int = 100,
    device: torch.device | None = None,
):
    """Return (predictor, estimator) for the requested precision."""
    assert precision in PRECISIONS, f"precision must be one of {PRECISIONS}"
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"{CKPT_PATH} not found -- run setup_lagllama.ps1 first to download the checkpoint."
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only the small `model_kwargs` hyperparameter dict is needed here; see the
    # torch.load patch above for why this (and every other load of this
    # checkpoint) stays on CPU.
    ckpt = torch.load(str(CKPT_PATH))
    trained_kwargs = ckpt["hyper_parameters"]["model_kwargs"]
    del ckpt

    use_rope_scaling = context_length > trained_kwargs["context_length"]
    rope_scaling_args = (
        {
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / trained_kwargs["context_length"]),
        }
        if use_rope_scaling
        else None
    )

    estimator = LagLlamaEstimator(
        ckpt_path=str(CKPT_PATH),
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=trained_kwargs["input_size"],
        n_layer=trained_kwargs["n_layer"],
        n_embd_per_head=trained_kwargs["n_embd_per_head"],
        n_head=trained_kwargs["n_head"],
        scaling=trained_kwargs["scaling"],
        time_feat=trained_kwargs["time_feat"],
        rope_scaling=rope_scaling_args,
        batch_size=16,
        num_parallel_samples=num_parallel_samples,
        device=device,
    )

    # create_lightning_module() now always loads onto CPU (see the torch.load
    # patch above), so the model is moved to `device` here explicitly -- after
    # NF4 quantization, if requested, since Linear4bit quantizes on the .to(device)
    # call and must not be quantized from an already-cuda tensor.
    lightning_module = estimator.create_lightning_module()
    if precision == "nf4":
        lightning_module.model = replace_linear_with_nf4(lightning_module.model)
    lightning_module.model.to(device)

    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    return predictor, estimator
