"""Build a Lag-Llama zero-shot predictor in FP32, NF4 (bitsandbytes), or int4-ao (torchao).

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

from quantize_utils import quantize_int4_ao, replace_linear_with_nf4  # noqa: E402

PRECISIONS = ("fp32", "nf4", "int4-ao")

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


def _cast_floats(obj, dtype: torch.dtype):
    """Recursively cast floating-point tensors in a (possibly nested) tuple/list/dict."""
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: _cast_floats(v, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cast_floats(v, dtype) for v in obj)
    return obj


def _bridge_bf16_io(lightning_module) -> None:
    """Bridge the fp32/bf16 boundary for the int4-ao (torchao) path.

    The model's weights are bf16 (required by torchao's tile-packed int4
    kernel), but two things must stay fp32: gluonts' `past_target` input (its
    `robust` scaler calls `torch.nanquantile`, which bf16 doesn't support) and
    the final output (gluonts' `predict_to_numpy` calls `.cpu().numpy()`,
    which bf16 doesn't support either). So the cast to bf16 happens exactly at
    `transformer.wte`, the model's actual entry point *after* scaling -- not
    at the outer `LightningModule.forward()` -- and the cast back to fp32
    happens on the overall output.
    """

    def _wte_pre_hook(_module, args, kwargs):
        return _cast_floats(args, torch.bfloat16), _cast_floats(kwargs, torch.bfloat16)

    def _output_post_hook(_module, _args, _kwargs, output):
        return _cast_floats(output, torch.float32)

    lightning_module.model.transformer.wte.register_forward_pre_hook(_wte_pre_hook, with_kwargs=True)
    lightning_module.register_forward_hook(_output_post_hook, with_kwargs=True)


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
    # patch above), so the model is moved to `device` here explicitly. The two
    # quantization libraries want the opposite ordering relative to that move:
    # bitsandbytes' Linear4bit quantizes NF4 the moment its Params4bit is moved
    # onto CUDA, so it must be applied before .to(device); torchao's tinygemm
    # int4 packing instead expects to run on an already-CUDA-resident model, so
    # it must be applied after.
    lightning_module = estimator.create_lightning_module()
    if precision == "nf4":
        lightning_module.model = replace_linear_with_nf4(lightning_module.model)
        lightning_module.model.to(device)
    elif precision == "int4-ao":
        # torchao's tile-packed int4 kernel only supports bfloat16 weights.
        lightning_module.model.to(device).to(torch.bfloat16)
        quantize_int4_ao(lightning_module.model)
        _bridge_bf16_io(lightning_module)
    else:
        lightning_module.model.to(device)

    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    return predictor, estimator
