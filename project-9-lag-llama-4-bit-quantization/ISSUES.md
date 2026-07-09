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
