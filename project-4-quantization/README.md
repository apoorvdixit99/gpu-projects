# Quantization & Precision Optimization

Benchmarks GPT-2 (124M) inference across four precision levels to isolate the pure effect of quantization on speed, memory, and accuracy.

**Hardware:** NVIDIA RTX 4080 Laptop GPU (Ada Lovelace) · CUDA 12.6  
**Model:** GPT-2 124M (HuggingFace `gpt2`)  
**Quantization:** `optimum-quanto` (weight-only, post-training)

---

## What is being measured

Each precision level runs a single forward pass across a sweep of batch sizes and sequence lengths:

| Dimension | Values |
|---|---|
| Batch size | 1, 4, 8, 16 |
| Sequence length | 64, 128, 256 |

**Performance metrics (per configuration):**
- Latency — mean, std, p50, p95, p99 (ms), measured with CUDA events
- Throughput — tokens per second (`batch_size × seq_len / latency`)
- Peak GPU memory — MB allocated during inference

**Accuracy metric:**
- Perplexity on a fixed 20-sentence synthetic corpus (lower = better)
- Measured once per precision level with the same corpus for all four

---

## Precision levels

### FP32 — Baseline
Standard PyTorch inference with 32-bit floating point weights and activations. Reference for all speedup and accuracy comparisons.

### FP16 — Half precision
Model cast to `float16` via `.half()`. Full FP16 compute on Ada Lovelace tensor cores. Expected to be ~1.5–2x faster than FP32 with negligible accuracy loss.

### INT8 — Weight-only quantization
Weights quantized to 8-bit integers via `optimum-quanto` (`qint8`). Activations remain in FP32. Weights are dequantized to FP32 at runtime during each matrix multiplication. Memory savings: ~2x vs FP32. Latency: may be comparable to or slightly slower than FP16 at small batch sizes due to dequantization overhead; the benefit shows at larger batches and memory-constrained scenarios.

### INT4 — Aggressive weight quantization
Weights quantized to 4-bit integers via `optimum-quanto` (`qint4`). Activations remain in FP32. Memory savings: ~4x vs FP32. Accuracy degradation is more pronounced than INT8 — expect a measurable perplexity increase, especially for smaller models like GPT-2 where weight precision matters more than in larger models.

---

## Project structure

```
project-4-quantization/
├── src/
│   ├── corpus.py              Fixed 20-sentence corpus for perplexity
│   ├── bench_fp32.py          FP32 PyTorch baseline
│   ├── bench_fp16.py          FP16 PyTorch baseline
│   ├── bench_int8.py          INT8 weight quantization via optimum-quanto
│   ├── bench_int4.py          INT4 weight quantization via optimum-quanto
│   ├── measure_perplexity.py  Perplexity on the fixed corpus
│   ├── plot_results.py        Generate all five charts
│   └── run_benchmark.py       CLI entry point -- orchestrates all of the above
├── results/                   Generated output (gitignored)
│   ├── benchmark_*.csv        Latency / throughput / memory for every config
│   ├── perplexity_*.csv       Perplexity per precision level
│   └── plots/                 PNG charts (five files)
├── ISSUES.md                  Log of issues hit during setup and fixes applied
├── requirements.txt
└── README.md
```

---

## How to run

> Activate the shared venv from the `Nvidia/` parent directory first:
> ```powershell
> .venv\Scripts\Activate.ps1
> cd project-4-quantization
> ```

**Install the quantization dependency** (first time only):
```powershell
pip install optimum-quanto
```

**Full run** — all four precisions, benchmark + perplexity:
```powershell
python src/run_benchmark.py
```

**Subset of precisions:**
```powershell
python src/run_benchmark.py --precisions fp32 fp16        # baselines only
python src/run_benchmark.py --precisions int8 int4        # quantized only
```

**Custom sweep:**
```powershell
python src/run_benchmark.py --batch-sizes 1 8 32 --seq-lens 128 256
python src/run_benchmark.py --iterations 200              # more samples
python src/run_benchmark.py --no-perplexity               # skip accuracy measurement
python src/run_benchmark.py --no-plot                     # skip chart generation
python src/run_benchmark.py --help                        # all options
```

Results are saved to `results/benchmark_<timestamp>.csv` and `results/perplexity_<timestamp>.csv`. Plots go to `results/plots/`.

---

## Key design decisions

**Why weight-only quantization instead of full quantization (weights + activations)?**  
Full quantization requires calibration data to determine activation ranges, and activating INT8/INT4 compute paths in PyTorch on GPU requires hardware-specific kernels (e.g., TensorRT). `optimum-quanto` weight-only quantization is post-training, requires no calibration, and runs transparently on any CUDA GPU. This isolates the precision effect cleanly. TensorRT INT8 with calibration is deferred to a potential Project 4.5.

**Why does INT8/INT4 latency sometimes exceed FP16?**  
Weight-only quantization dequantizes weights to FP32 at each matrix multiply. This adds an overhead that is constant per layer and does not scale with batch size. At small batch sizes the dequantization cost dominates, making INT8/INT4 slower than FP16. At larger batch sizes the memory bandwidth savings outweigh the overhead.

**Why perplexity and not a task accuracy metric?**  
GPT-2 is a language model, not a classifier, so task accuracy requires a downstream evaluation harness. Perplexity directly measures the model's output distribution against the FP32 reference using a fixed corpus — no external dataset, no task-specific fine-tuning. It gives a clean, self-contained accuracy signal that scales with quantization severity.

**Why a fixed synthetic corpus instead of WikiText-2?**  
WikiText-2 requires downloading ~1 MB of data and tokenizing thousands of sentences. A hardcoded 20-sentence corpus removes the external dependency, runs in under a second, and is sufficient to detect the perplexity gap between FP32 and INT4. The absolute perplexity values are not comparable to published WikiText-2 benchmarks, but the relative degradation across precision levels is meaningful.

**Why `use_cache=False` during benchmarking?**  
Disabling the KV cache ensures all four precisions perform identical computation — a full attention pass over all input tokens. With caching enabled, subsequent tokens would compute attention over a growing KV state, making latency comparisons across precision levels dependent on sequence position rather than just precision.
