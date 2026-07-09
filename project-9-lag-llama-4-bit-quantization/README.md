# Lag-Llama 4-bit Quantization

Quantizes [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama) -- an open-source decoder-only transformer foundation model for probabilistic time series forecasting -- to 4-bit NF4 weights via `bitsandbytes`, and compares it against the FP32 pretrained checkpoint on real zero-shot forecasting workloads.

**Hardware:** NVIDIA RTX 4080 Laptop GPU (Ada Lovelace) · CUDA 12.6
**Model:** Lag-Llama (pretrained checkpoint, `time-series-foundation-models/Lag-Llama`)
**Quantization:** `bitsandbytes` `Linear4bit` (NF4, weight-only, post-training)
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

### NF4 -- 4-bit weight-only quantization
Every `nn.Linear` in the transformer (`q_proj`, `kv_proj`, `c_proj`, MLP projections, `wte`) is replaced with `bitsandbytes.nn.Linear4bit` using the NF4 (4-bit NormalFloat) data type. Weights are dequantized to the compute dtype on the fly during each matmul; activations and sampling stay in FP32. Expected: ~4x smaller weight memory footprint, with some forecast accuracy degradation from the lower weight precision -- the interesting question this project answers is *how much*, on a model this small (Lag-Llama's released checkpoint is far smaller than a typical LLM, so per-weight precision loss has more relative impact than in, say, a 7B-parameter model).

---

## Results

Full sweep, RTX 4080 Laptop GPU (see `results/latency_20260709_114623.csv`, `results/accuracy_20260709_114623.csv`, `results/plots/`):

**Latency (ms, mean) -- lower is better**

| Context length | FP32 | NF4 |
|---|---|---|
| 32 | 205.99 | 306.34 |
| 64 | 310.63 | 323.56 |
| 128 | 564.18 | 572.66 |

**Peak GPU memory (MB) -- lower is better**

| Context length | FP32 | NF4 |
|---|---|---|
| 32 | 66.3 | 58.3 |
| 64 | 92.2 | 84.3 |
| 128 | 143.7 | 135.7 |

**Zero-shot accuracy -- lower is better**

| Dataset | Precision | MASE | sMAPE | CRPS (approx.) | MSIS |
|---|---|---|---|---|---|
| `airpassengers` (n=1) | FP32 | 2.653 | 0.174 | 0.137 | 15.67 |
| `airpassengers` (n=1) | NF4 | 1.472 | 0.091 | 0.086 | 19.40 |
| `exchange_rate` (n=40) | FP32 | 2.046 | 0.013 | 0.010 | 19.67 |
| `exchange_rate` (n=40) | NF4 | 3.065 | 0.022 | 0.019 | 29.00 |
| `m4_hourly` (n=50) | FP32 | 3.957 | 0.225 | 0.131 | 41.16 |
| `m4_hourly` (n=50) | NF4 | 3.357 | 0.201 | 0.092 | 42.83 |

**Takeaways:**
- **NF4 is consistently slower, not faster.** Lag-Llama's released checkpoint is small, so weight-only dequantization overhead per matmul dominates rather than being offset by memory-bandwidth savings -- the same effect Project 4 observed quantizing GPT-2 to INT4.
- **Memory savings are modest** (~8-12%, growing with context length). NF4 only shrinks the linear-layer weights; activations and the KV/lag context (which scale with context length) are unaffected and make up a large share of peak memory on a model this small.
- **Accuracy impact is mixed, not uniformly worse.** NF4 improves MASE/CRPS on `airpassengers` and `m4_hourly` but degrades on `exchange_rate`. The `airpassengers` comparison (`n=1`, it's a single-series dataset) isn't statistically strong on its own -- `exchange_rate` and `m4_hourly` (40-50 series each) are the more reliable signal, and there the picture is genuinely mixed rather than a clean "NF4 always loses" story.

---

## Project structure

```
project-9-lag-llama-4-bit-quantization/
├── src/
│   ├── quantize_utils.py      Recursive nn.Linear -> bnb.nn.Linear4bit (NF4) swap
│   ├── load_model.py          Builds a FP32 or NF4 LagLlamaEstimator/predictor from the checkpoint
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

**Full run** -- both precisions, latency + accuracy:
```powershell
python src/run_benchmark.py
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
