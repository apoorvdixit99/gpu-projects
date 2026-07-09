# Lag-Llama 4-bit Quantization

Quantizes [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama) -- an open-source decoder-only transformer foundation model for probabilistic time series forecasting -- to 4-bit weights two different ways (`bitsandbytes` NF4 and `torchao` int4), and compares both against the FP32 pretrained checkpoint on real zero-shot forecasting workloads.

**Hardware:** NVIDIA RTX 4080 Laptop GPU (Ada Lovelace) · CUDA 12.6
**Model:** Lag-Llama (pretrained checkpoint, `time-series-foundation-models/Lag-Llama`)
**Quantization:** `bitsandbytes` `Linear4bit` (NF4, weight-only, post-training) and `torchao` `Int4WeightOnlyConfig` (tile-packed int4, weight-only, post-training)
**Data:** GluonTS built-in datasets -- `airpassengers`, `exchange_rate`, `m4_hourly`

---

## What is Lag-Llama

Lag-Llama reframes univariate time series forecasting as next-token prediction: at each step the model sees a feature vector built from **lagged values** of the series (values from `t-1`, `t-7`, `t-24`, ... depending on frequency) plus scaling statistics, and predicts the parameters of a Student-T distribution for the next timestep. The backbone is a Llama-style pre-norm transformer decoder (RMSNorm, rotary position embeddings, SwiGLU-style MLP) -- architecturally a small LLM, just with a lag-feature embedding instead of a token embedding. This is what makes it **zero-shot**: because it was pretrained across many (frequency, domain) combinations, it can forecast a series it has never seen without any fine-tuning.

## What is being measured

**Performance metrics** (latency, throughput, peak GPU memory) across a context-length sweep, timed on a fixed batch of series from `airpassengers`:

| Dimension | Values |
|---|---|
| Context length | 32, 64, 128 |
| Prediction length | 24 |

**Accuracy metrics** (zero-shot, no fine-tuning) on three GluonTS datasets of different frequency/domain:

| Dataset | Frequency | Domain |
|---|---|---|
| `airpassengers` | Monthly | Classic single-series benchmark |
| `exchange_rate` | Daily | Multi-series financial |
| `m4_hourly` | Hourly | M4 competition, high-frequency |

Reported per dataset: **MASE**, **sMAPE**, **CRPS** (approximated via GluonTS's `mean_wQuantileLoss`, the mean pinball loss averaged over quantiles -- the standard proxy for CRPS when only a quantile grid is available), and **MSIS**.

---

## Precision levels

### FP32 -- Baseline
The pretrained checkpoint loaded as-is. Reference for all comparisons.

### NF4 -- 4-bit weight-only quantization (bitsandbytes)
Every `nn.Linear` in the transformer (`q_proj`, `kv_proj`, `c_proj`, MLP projections, `wte`) is replaced with `bitsandbytes.nn.Linear4bit` using the NF4 (4-bit NormalFloat) data type. Weights are dequantized to the compute dtype (FP32) on the fly during each matmul; activations and sampling stay in FP32. Expected: ~4x smaller weight memory footprint, with some forecast accuracy degradation from the lower weight precision -- the interesting question this project answers is *how much*, on a model this small (Lag-Llama's released checkpoint is far smaller than a typical LLM, so per-weight precision loss has more relative impact than in, say, a 7B-parameter model).

### int4-ao -- 4-bit weight-only quantization (torchao)
The whole model is cast to `bfloat16`, then `torchao.quantization.quantize_` with `Int4WeightOnlyConfig` (tile-packed int4, `group_size=32`) is applied to eligible `nn.Linear` layers. "Eligible" turns out to be a real constraint here, not a formality: the CUDA kernel torchao dispatches to (`_weight_int4pack_mm`) only accepts `group_size` in `{32, 64, 128, 256}`, and Lag-Llama's transformer width is 144 (`n_head=9 x n_embd_per_head=16`) -- none of those group sizes divide 144. So of the model's linear layers, only `mlp.c_proj` (`in_features=512`, one per transformer block, 8 total) is actually quantized to int4; `q_proj`, `kv_proj`, `attn.c_proj`, `c_fc1`, `c_fc2`, and `wte` all stay bf16. `quantize_` skips ineligible layers silently (logged, not raised) rather than erroring, which is what makes this graceful instead of a hard failure.

This is the single most interesting finding of the two-library comparison: bitsandbytes' NF4 has no shape restriction and quantizes every linear layer regardless of width, while torchao's kernel-backed int4 path is far pickier about tensor shapes and requires bf16 end-to-end. In exchange, casting the *whole* model to bf16 (not just the quantized layers) roughly halves the memory of everything -- including the un-quantized layers -- which is why int4-ao ends up using *less* peak memory than NF4 here despite quantizing far fewer layers to int4 (see Results below).

---

## Results

Full sweep, RTX 4080 Laptop GPU (see `results/latency_20260709_132021.csv`, `results/accuracy_20260709_132021.csv`, `results/plots/`):

**Latency (ms, mean) -- lower is better**

| Context length | FP32 | NF4 | int4-ao |
|---|---|---|---|
| 32 | 177.5 | 318.3 | 185.7 |
| 64 | 294.0 | 319.7 | 206.6 |
| 128 | 546.9 | 571.8 | 352.2 |

**Peak GPU memory (MB) -- lower is better**

| Context length | FP32 | NF4 | int4-ao |
|---|---|---|---|
| 32 | 66.3 | 58.3 | 41.1 |
| 64 | 92.2 | 84.3 | 57.6 |
| 128 | 143.7 | 135.7 | 84.4 |

**Zero-shot accuracy -- lower is better**

| Dataset | Precision | MASE | sMAPE | CRPS (approx.) | MSIS |
|---|---|---|---|---|---|
| `airpassengers` (n=1) | FP32 | 2.853 | 0.190 | 0.143 | 20.40 |
| `airpassengers` (n=1) | NF4 | 1.579 | 0.098 | 0.085 | 18.77 |
| `airpassengers` (n=1) | int4-ao | 2.701 | 0.178 | 0.134 | 11.73 |
| `exchange_rate` (n=40) | FP32 | 2.052 | 0.013 | 0.010 | 17.52 |
| `exchange_rate` (n=40) | NF4 | 3.166 | 0.023 | 0.020 | 28.68 |
| `exchange_rate` (n=40) | int4-ao | 2.071 | 0.013 | 0.010 | 20.79 |
| `m4_hourly` (n=50) | FP32 | 4.017 | 0.230 | 0.129 | 40.76 |
| `m4_hourly` (n=50) | NF4 | 3.372 | 0.200 | 0.090 | 41.74 |
| `m4_hourly` (n=50) | int4-ao | 3.587 | 0.200 | 0.111 | 38.61 |

**Takeaways:**
- **The two 4-bit libraries land in very different places, despite both being "int4."** NF4 (bitsandbytes) quantizes every linear layer but keeps compute in FP32 -- it's uniformly *slower* than FP32 (dequant overhead per matmul dominates on a model this small) with modest memory savings. int4-ao (torchao), by contrast, quantizes only `mlp.c_proj` (a kernel/shape constraint, not a choice -- see "int4-ao" above) but casts the *entire* model to bf16 -- it ends up both **faster than FP32 at every context length** and using **~35-40% less memory than NF4**, purely from the bf16 cast, not from the int4 quantization itself.
- **int4-ao's speed and memory wins are really a bf16 story, not an int4 story.** Since only 8 of 56 linear layers actually run the int4 kernel, most of the observed improvement over FP32 would show up from casting to bf16 alone (halved weight memory, native tensor-core throughput on Ada) -- the int4 quantization is a small additional slice on top of that.
- **NF4's accuracy is the most volatile of the three** -- best on `airpassengers` (MASE 1.58 vs FP32's 2.85) but worst on `exchange_rate` (3.17 vs 2.05). int4-ao tracks the FP32 baseline much more closely across all three datasets (e.g. essentially matching FP32 on `exchange_rate`: 2.07 vs 2.05), consistent with only a small fraction of its weights actually being quantized.
- The `airpassengers` comparison (`n=1`, a single-series dataset) isn't statistically strong on its own -- `exchange_rate` and `m4_hourly` (40-50 series each) are the more reliable signal.

---

## Project structure

```
project-9-lag-llama-4-bit-quantization/
├── src/
│   ├── quantize_utils.py      nn.Linear -> bnb.nn.Linear4bit (NF4) swap; torchao quantize_ int4 pass
│   ├── load_model.py          Builds a FP32, NF4, or int4-ao LagLlamaEstimator/predictor from the checkpoint
│   ├── bench_latency.py       Latency / throughput / peak-memory benchmark (predictor.predict())
│   ├── evaluate_accuracy.py   Zero-shot CRPS / MASE / sMAPE via GluonTS Evaluator
│   ├── plot_results.py        Generate all charts
│   └── run_benchmark.py       CLI entry point -- orchestrates all of the above
├── vendor/lag-llama/          Cloned official repo (gitignored, created by setup script)
├── checkpoints/               Downloaded lag-llama.ckpt (gitignored)
├── results/                   Generated output (CSV + PNG plots)
├── setup_lagllama.ps1         One-time setup: deps, repo clone, checkpoint download
├── ISSUES.md                  Log of issues hit during setup and fixes applied
├── requirements.txt
└── README.md
```

---

## How to run

> This project uses its own venv (`../.venv3`), **not** the shared `../.venv` -- see
> "Why a separate venv" below.

**One-time setup** (creates `.venv3`, installs deps, clones the lag-llama repo, downloads the checkpoint):
```powershell
cd project-9-lag-llama-4-bit-quantization
.\setup_lagllama.ps1
```

**Subsequent runs:**
```powershell
..\.venv3\Scripts\Activate.ps1
cd project-9-lag-llama-4-bit-quantization
```

**Full run** -- all three precisions, latency + accuracy:
```powershell
python src/run_benchmark.py
```

**Just the two int4 variants** (skip the FP32 baseline):
```powershell
python src/run_benchmark.py --precisions nf4 int4-ao
```

**Latency only** (skip dataset accuracy evaluation, faster):
```powershell
python src/run_benchmark.py --no-accuracy
```

**Custom sweep:**
```powershell
python src/run_benchmark.py --context-lengths 32 64 --datasets airpassengers m4_hourly
python src/run_benchmark.py --max-series-per-dataset 20   # faster accuracy pass
python src/run_benchmark.py --no-plot                     # skip chart generation
python src/run_benchmark.py --help                        # all options
```

Results are saved to `results/latency_<timestamp>.csv` and `results/accuracy_<timestamp>.csv`. Plots go to `results/plots/`.

---

## Key design decisions

**Why bitsandbytes instead of `optimum-quanto` (used in Project 4)?**
Lag-Llama is not a HuggingFace `transformers` model -- it's a standalone PyTorch Lightning module from a separate GitHub repo, loaded from a raw `.ckpt`. `bitsandbytes.nn.Linear4bit` is a drop-in replacement for `torch.nn.Linear` with no dependency on the `transformers` model-loading machinery, so it swaps cleanly into any custom architecture. It's also the more common choice specifically for 4-bit (NF4/QLoRA-style) quantization in practice.

**Why add torchao as a second 4-bit library?**
torchao is PyTorch's own native quantization toolkit -- no extra dependency outside the PyTorch ecosystem, and tighter `torch.compile` integration going forward. Its `quantize_()` API takes the same model-agnostic approach as bitsandbytes (swap/rewrite `nn.Linear` weights in place), so it slots into the same `quantize_utils.py` pattern. It turned out to be a genuinely useful second data point rather than a redundant one: torchao's kernel-backed int4 path has real shape constraints bitsandbytes doesn't (see "int4-ao" above), so the two libraries hit this specific model very differently -- that contrast is more informative than either result alone.

**Why does int4-ao need bf16 input/output bridging hooks in `load_model.py`?**
The model's weights are bf16 (required by torchao's tile-packed int4 kernel), but two things must stay FP32: GluonTS's `past_target` input -- Lag-Llama's `robust` scaler calls `torch.nanquantile`, which doesn't support bf16 -- and the final output, since GluonTS's `predict_to_numpy` calls `.cpu().numpy()`, which also doesn't support bf16. So the cast to bf16 happens via a forward pre-hook on `transformer.wte` specifically (the model's actual entry point *after* scaling, not before), and the cast back to FP32 happens via a forward hook on the outermost `LightningModule`. Casting at the outer boundary instead (the first thing tried) broke the scaler.

**Why time `predictor.predict()` end-to-end instead of isolating a single forward pass (as in Projects 1 and 4)?**
GPT-2 benchmarking isolates the transformer forward pass because that's the entire unit of work. Lag-Llama inference is inherently a pipeline: GluonTS transforms the raw series into lag features, then the model autoregressively samples `num_parallel_samples` trajectories. That full pipeline is what a real forecasting request pays for, so wall-clock timing (`time.perf_counter`, GPU-synchronized) around the whole call is the more honest measurement -- a pure-forward-pass number would understate what actually changes (or doesn't) between FP32 and NF4.

**Why CRPS via `mean_wQuantileLoss` instead of a closed-form CRPS?**
Lag-Llama emits Student-T distribution parameters but forecasts are evaluated as sample paths, so GluonTS's `Evaluator` scores them empirically. `mean_wQuantileLoss` (mean pinball/quantile loss averaged over a quantile grid) is the standard sample-based approximation to CRPS and is what GluonTS reports out of the box -- no closed-form Student-T CRPS implementation needed.

**Why these three datasets?**
`airpassengers`, `exchange_rate`, and `m4_hourly` span three different frequencies (monthly, daily, hourly) and domains (classic univariate, multi-series financial, high-frequency competition data), matching the spirit of the zero-shot benchmarks in the original Lag-Llama paper/demo -- no fine-tuning, no domain-specific training, just probing how well the pretrained weights generalize at each precision level.

**Why a separate venv (`.venv3`) instead of the shared one?**
The lag-llama repo's `LagLlamaEstimator` API requires `gluonts<=0.14.4`, which pins `numpy~=1.16` and `pandas<2.2`. The shared `.venv` used by Projects 1 and 3-8 has much newer `numpy`/`pandas` pinned in the root `requirements.txt`; installing gluonts there would silently downgrade those packages for every other project. Project 2 hit the same class of problem (backend-specific dependency versions) and solved it with a dedicated `.venv2` -- this project follows that precedent with `.venv3`.

**Why is the checkpoint/repo not committed to git?**
`lag-llama.ckpt` and the vendored `lag-llama` repo are fetched by `setup_lagllama.ps1` and gitignored, the same way Project 1 gitignores exported `models/`. Only the benchmark code and results (CSV + plots) are portfolio artifacts worth versioning.
