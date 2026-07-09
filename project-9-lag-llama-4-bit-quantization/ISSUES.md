# Known Issues & Fixes

Issues encountered during setup and first run, in chronological order.

## 1. Installing project deps into the shared `.venv` downgraded numpy/pandas for every other project

`gluonts<=0.14.4` (pinned because the lag-llama repo's `LagLlamaEstimator` targets that API) requires `numpy~=1.16` and `pandas<2.2`. Running `pip install -r requirements.txt` in the shared `../.venv` silently downgraded `numpy` 2.4.6 -> 1.26.4, `pandas` 3.0.3 -> 2.1.4, and `fsspec`/`packaging`, which are pinned newer for Projects 1 and 3-8.

**Fix:** give this project its own venv, `../.venv3`, exactly like Project 2. `setup_lagllama.ps1` now creates and installs into `.venv3` instead of the shared environment. (The shared `.venv` was restored to its pinned versions afterward.)

## 2. `torch.load` fails on `lag-llama.ckpt` with `WeightsUnpickler error`

PyTorch >=2.6 changed the default of `torch.load(weights_only=...)` to `True`. Lightning's internal checkpoint loader (invoked inside `LagLlamaEstimator.create_lightning_module()`) calls `torch.load` without a `weights_only` argument, so it inherits the new default and rejects the pickled `gluonts.torch.distributions.studentT.StudentTOutput` class in the checkpoint.

**Fix:** `src/load_model.py` monkeypatches `torch.load` to default `weights_only=False` before any checkpoint loading happens. Safe here because the checkpoint is from the official `time-series-foundation-models` HuggingFace repo.

## 3. `CUDACachingAllocator ... memory allocation failed with OOM` during accuracy evaluation

During `evaluate_accuracy.py`, occasional `[W...] CUDACachingAllocator.cpp:3933] memory allocation failed with OOM` warnings appeared, more often on datasets with longer `prediction_length` (worst on `m4_hourly`). These were **not** fatal -- PyTorch's allocator retried after freeing its own unused cache and every result printed correctly afterward, run exit code 0 throughout.

Initially misdiagnosed as `build_predictor()` transiently double-loading the full checkpoint onto the GPU (once via a manual `torch.load` just to read hyperparameters, once via Lightning's internal loader) -- that was real and worth fixing regardless (see `src/load_model.py`, both loads now forced to CPU with the model moved to `device` explicitly afterward), but a follow-up full run with that fix still showed the same warnings, slightly more often. The actual driver: `LagLlamaEstimator` was built with `batch_size=64` and `num_parallel_samples=100`, so accuracy evaluation samples up to 6,400 forecast trajectories in parallel per batch -- on a 12GB card, combined with several sequential predictor rebuilds in one process (never resetting the CUDA context), that's close to the memory ceiling, and the requested allocation size scales with each dataset's prediction length.

**Fix:** lowered `batch_size` from 64 to 16 in `src/load_model.py`'s `LagLlamaEstimator` construction. Accuracy evaluation runs somewhat slower (more, smaller batches) but latency/throughput benchmarks are unaffected (`bench_latency.py` doesn't use this batch size). If the warnings reappear on a smaller GPU, lower it further.

## 4. torchao int4 quantization: `ImportError: Requires mslk >= 1.0.0`

`Int4WeightOnlyConfig`'s default `int4_packing_format` (`PLAIN`, torchao's newer "version 2" tensor subclass path) depends on an internal `mslk` package that isn't a normal pip dependency and wasn't installed.

**Fix:** use `int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D` instead (`src/quantize_utils.py`), which dispatches to PyTorch's built-in `torch.ops.aten._weight_int4pack_mm` and has no extra dependency.

## 5. torchao int4: `Only bfloat16 is supported for Int4TilePackedTo4dTensor, got torch.float32`

The tile-packed kernel requires bf16 weights; Lag-Llama's checkpoint loads as FP32.

**Fix:** cast the whole model to `bfloat16` before calling `quantize_()` (`src/load_model.py`). This has a knock-on effect covered in issues 6 and 7 below, and turned out to reduce peak memory more than expected -- see the README's "int4-ao" section.

## 6. torchao int4: `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`

Once the model is bf16, GluonTS's data pipeline still feeds it FP32 tensors (`past_target`, `past_observed_values`, etc.), so the first linear layer (`transformer.wte`) sees a dtype mismatch.

**First attempt (wrong):** a forward pre-hook on the whole `LightningModule` casting every incoming float tensor to bf16. This broke issue 7 below -- casting *before* the scaler runs is too early.

**Fix:** a forward pre-hook on `transformer.wte` specifically -- the model's actual entry point *after* Lag-Llama's own scaling/lag-feature construction -- casts only the already-scaled features to bf16. Everything upstream of that (the scaler) stays FP32; everything downstream (the transformer) is consistently bf16.

## 7. torchao int4: `RuntimeError: quantile() input tensor must be either float or double dtype`

Surfaced while chasing issue 6: Lag-Llama's `robust` scaler (`gluon_utils/scalers/robust_scaler.py`) calls `torch.nanquantile` on `past_target`, which does not support bf16 -- confirming the cast has to happen after scaling, not before (see fix for issue 6).

## 8. torchao int4: `RuntimeError: Expected qGroupSize == 32 || 64 || 128 || 256 to be true`

torchao's own Python-level check (`in_features % group_size == 0`) is necessary but not sufficient -- it let `group_size=16` through because it divides Lag-Llama's 144-wide attention/MLP projections, but the actual CUDA kernel (`_weight_int4pack_mm`) only accepts group_size in `{32, 64, 128, 256}` and raises at the *first forward call*, not at `quantize_()` time.

**Fix:** use `group_size=32` (the smallest valid size). Since none of `{32, 64, 128, 256}` divide 144, this means only `mlp.c_proj` (`in_features=512`, divisible by all of them) actually gets quantized to int4 across the whole model -- every 144-wide projection and `wte` (`in_features=92`) stay bf16. `quantize_()` skips them silently rather than raising, which is what makes this degrade gracefully instead of crashing. This is a genuine architecture/kernel mismatch (Lag-Llama's `n_head=9` gives an unusual, non-power-of-2 width), not a workaround to "fix" further -- see the README's "int4-ao" section for why it's still a meaningful comparison point against NF4 (which has no such shape restriction).
