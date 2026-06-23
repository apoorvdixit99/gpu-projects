# Deep Learning Workload Performance Analysis

Compares four production deep learning models across FLOPs, memory footprint, latency, throughput, and GPU utilization (MFU) to understand how architecture and modality affect hardware efficiency.

**Hardware:** NVIDIA RTX 4080 Laptop GPU (Ada Lovelace) · CUDA 12.6  
**Models:** GPT-2 (124M), DistilGPT-2 (82M), BERT-base (110M), ResNet-50 (25M)

---

## What is being measured

All models run FP16 inference on CUDA. NLP models use a fixed sequence length of 128 tokens; ResNet-50 uses 224×224 images.

| Dimension    | Values                    |
|--------------|---------------------------|
| Batch size   | 1, 4, 8, 16, 32           |
| NLP seq len  | 128 tokens (fixed)        |
| Vision input | 224×224 RGB (fixed)       |
| Precision    | FP16 (weights + activations) |

**Per-model metrics:**

| Module                | Metric             | Description |
|-----------------------|--------------------|-------------|
| `measure_flops.py`    | MACs (G)           | Multiply-accumulate ops per sample |
| `measure_flops.py`    | FLOPs (G)          | 2 × MACs (standard convention) |
| `measure_flops.py`    | Parameters (M)     | Total trainable parameter count |
| `measure_latency.py`  | Latency mean (ms)  | Mean forward-pass time via CUDA events |
| `measure_latency.py`  | Latency std (ms)   | Run-to-run variance |
| `measure_latency.py`  | Throughput         | Tokens/s (NLP) or images/s (vision) |
| `measure_latency.py`  | MFU (%)            | Model FLOP Utilization vs peak FP16 TFLOPS |
| `measure_memory.py`   | Peak allocated (MB) | Tensor memory high-water mark |
| `measure_memory.py`   | Peak reserved (MB)  | CUDA allocator cache high-water mark |
| `measure_memory.py`   | Fragmentation (%)   | Cached-but-unused allocator fraction |

---

## Analysis modules

### `measure_flops.py` — FLOPs and parameter count
Counts multiply-accumulate operations (MACs) per sample at batch size 1 on CPU in FP32.
For transformer models (GPT-2, DistilGPT-2, BERT-base), an **analytical formula** is
used instead of torchinfo — torchinfo 1.8.0 has a bug where `nn.Linear` with a 3D input
`(batch, seq, features)` ignores the sequence dimension, giving counts that are
`seq_len`× too low for BERT and `~10×` too high for GPT-2's `Conv1D`-based layers.
See ISSUES.md #2 for the full investigation. For ResNet-50, torchinfo's `nn.Conv2d`
counting is accurate and is used directly.

Analytical formula per transformer layer (seq=S, hidden=h, ffn=f):
```
QKV projections  : 3 × S × h²
Attention scores : S² × h
Attention × value: S² × h
Output projection: S × h²
FFN (two linears): 2 × S × h × f
─────────────────────────────────
Per layer total  : 4Sh² + 2S²h + 2Shf
```

### `measure_latency.py` — Latency, throughput, and MFU
Times each forward pass with `torch.cuda.Event` pairs (start/end), which measure only
GPU kernel execution time and exclude Python dispatch overhead. Each batch size runs
10 warmup iterations (excluded) followed by 50 timed iterations.

MFU is computed as:
```
MFU = (batch_size × FLOPs_per_sample) / (latency_s × peak_fp16_tflops × 1e12)
```

The default peak is set to **74.4 TFLOPS** — the dense (non-sparse) FP16 Tensor Core
throughput of the RTX 4080 Laptop GPU. If you run on a different GPU, pass the correct
value via `--peak-fp16-tflops`. Common reference values:

| GPU                    | Dense FP16 TFLOPS |
|------------------------|-------------------|
| RTX 4080 Laptop GPU    | 74.4              |
| RTX 4080 (desktop)     | 121.9             |
| RTX 4090               | 165.2             |
| A100 SXM4 80GB         | 312.0             |

### `measure_memory.py` — GPU memory footprint
Runs warmup passes to fill the CUDA allocator cache, then resets peak stats and takes
a single measurement pass. Reports both `max_memory_allocated` (true tensor usage) and
`max_memory_reserved` (allocator cache, always ≥ allocated). The gap between the two
is memory the allocator holds to avoid repeated `cudaMalloc` calls.

---

## Project structure

```
project-5-workload-analysis/
├── src/
│   ├── run_analysis.py      CLI entry point — orchestrates all three modules
│   ├── models.py            Model registry: ModelSpec dataclass + loaders + input factories
│   ├── measure_flops.py     FLOPs and parameter count (analytical + torchinfo)
│   ├── measure_latency.py   Latency, throughput, and MFU via CUDA events
│   ├── measure_memory.py    GPU memory footprint per model × batch size
│   └── plot_results.py      Five output charts (NVIDIA green palette)
├── results/                 Generated output (gitignored)
│   ├── flops_*.csv          Architecture summary per model
│   ├── latency_*.csv        Latency / throughput / MFU per model × batch size
│   ├── memory_*.csv         Memory stats per model × batch size
│   └── plots/               PNG charts (five files)
├── ISSUES.md                Issues hit during development and fixes applied
├── requirements.txt
└── README.md
```

---

## How to run

> Activate the shared venv from the `Nvidia/` parent directory first:
> ```powershell
> .venv\Scripts\Activate.ps1
> cd project-5-workload-analysis
> ```

**Full analysis — all four models, all batch sizes:**
```powershell
python src/run_analysis.py
```

**Custom model selection or batch sizes:**
```powershell
python src/run_analysis.py --models gpt2 distilgpt2
python src/run_analysis.py --batch-sizes 1 8 32
```

**Override GPU peak TFLOPS** (default is 74.4 for RTX 4080 Laptop GPU):
```powershell
python src/run_analysis.py --peak-fp16-tflops 121.9   # desktop RTX 4080
python src/run_analysis.py --peak-fp16-tflops 165.2   # RTX 4090
```

**Skip plot generation:**
```powershell
python src/run_analysis.py --no-plot
```

**Run a single module directly:**
```powershell
python src/measure_flops.py
python src/measure_latency.py
python src/measure_memory.py
```

Results are saved to `results/` with timestamps. Plots are written to `results/plots/`.

---

## Key design decisions

**Why analytical FLOPs for transformers instead of torchinfo?**  
torchinfo 1.8.0 has a bug where `nn.Linear` with a 3D input `(batch, seq, features)`
computes MACs as `out_features × in_features` rather than
`seq × out_features × in_features`. This gives counts that are `seq_len`× too low for
BERT-base (0.11G vs 11.2G expected) and — through the different `Conv1D` code path —
anomalously high for GPT-2. The analytical formula is derived directly from each
model's architecture hyperparameters and is accurate regardless of library version.

**Why CUDA events instead of `time.perf_counter()` for latency?**  
CUDA kernels execute asynchronously. `time.perf_counter()` measures wall time including
Python dispatch overhead and CPU–GPU synchronization gaps. CUDA event pairs record
only the time the GPU spent executing kernels, giving a clean measure of device
throughput independent of Python overhead.

**Why `reset_peak_memory_stats()` after warmup but before measurement?**  
The CUDA allocator accumulates peak stats across all allocations since process start.
Resetting after warmup means the reported peak reflects only the single measurement
pass. Without the reset, warmup allocations inflate the numbers.

**Why `attn_implementation="eager"` for BERT?**  
Transformers 5.x enables `scaled_dot_product_attention` (SDPA) by default on PyTorch
≥ 2.0. SDPA fuses the QK matmul, softmax, and AV matmul into a single opaque kernel.
Forcing eager attention ensures the inference timing uses the same explicit-matmul
path that the analytical FLOPs formula accounts for, keeping the MFU ratio
interpretable.

**Why separate `peak_allocated` and `peak_reserved` in memory measurement?**  
`max_memory_allocated` is the true high-water mark of memory used by live tensors.
`max_memory_reserved` includes empty blocks the allocator holds cached to avoid
`cudaMalloc` on the next call. Reporting only `reserved` inflates the apparent cost
at small batch sizes where the allocator pre-caches aggressively.
