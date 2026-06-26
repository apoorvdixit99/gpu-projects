# LLM Inference Optimization

Benchmarks GPT-2 (124M) inference across three runtimes to measure the real-world impact of switching from a plain PyTorch model to an optimized inference backend.

**Hardware:** NVIDIA RTX 4080 Laptop GPU (Ada Lovelace) · CUDA 12.6  
**Model:** GPT-2 124M (HuggingFace `gpt2`)

---

## What is being measured

Each backend runs a single forward pass (the prefill phase — processing input tokens and producing logits) across a sweep of batch sizes and sequence lengths:

| Dimension | Values |
|---|---|
| Batch size | 1, 4, 8, 16 |
| Sequence length | 64, 128, 256 |

**Metrics collected per configuration:**
- Latency — mean, std, p50, p95, p99 (ms), measured with CUDA events
- Throughput — tokens per second (`batch_size × seq_len / latency`)
- Peak GPU memory — MB allocated during inference

---

## Backends

### PyTorch FP16
The reference baseline. GPT-2 loaded from HuggingFace, moved to GPU, and cast to `float16`. Inference runs with `torch.no_grad()` and `use_cache=False` to keep the comparison fair (single forward pass, no KV cache).

### PyTorch FP16 + SDPA
Same as the baseline but loaded with `attn_implementation="sdpa"`, which routes every attention layer through `torch.nn.functional.scaled_dot_product_attention`. On Ada hardware PyTorch selects Flash Attention, fusing the full QK^T → softmax → V operation into a single memory-efficient kernel. No model export or compilation needed — it's a drop-in swap over the baseline.

### ONNX Runtime (CUDAExecutionProvider)
The model is exported to ONNX opset 18 using PyTorch's dynamo-based exporter, then loaded into an ONNX Runtime `InferenceSession` with `CUDAExecutionProvider`. ORT applies its own graph optimizations (op fusion, constant folding) independently of TensorRT.

### TensorRT FP16
The ONNX model is compiled into a TensorRT engine at startup using `build_trt.py`. The engine is built with an optimization profile covering the full benchmark sweep (min/opt/max shapes), allowing TRT to tune kernels for the expected input range. At inference time, device buffers are allocated as PyTorch CUDA tensors and their raw pointers are passed directly to the TRT execution context — no pycuda or cuda-python dependency needed.

---

## Project structure

```
project-1-llm-inference-optimization/
├── src/
│   ├── export_onnx.py       Export GPT-2 to ONNX (opset 18, dynamic shapes)
│   ├── build_trt.py         Compile ONNX → TensorRT engine
│   ├── bench_pytorch.py     PyTorch FP16 benchmark
│   ├── bench_onnx.py        ONNX Runtime benchmark
│   ├── bench_tensorrt.py    TensorRT benchmark
│   ├── plot_results.py      Generate latency / throughput / memory charts
│   └── run_benchmark.py     CLI entry point — orchestrates all of the above
├── models/                  Generated model files (gitignored)
│   ├── gpt2.onnx            ONNX graph (weights stored in gpt2.onnx.data)
│   └── gpt2_fp16.trt        Compiled TRT engine
├── results/                 Benchmark output (gitignored)
│   ├── benchmark_*.csv      Raw numbers for every (backend, batch, seq_len)
│   └── plots/               PNG charts — latency, throughput, memory
├── ISSUES.md                Running log of issues hit during setup and fixes applied
└── README.md
```

---

## How to run

> Activate the shared venv from the `Nvidia/` parent directory first:
> ```powershell
> .venv\Scripts\Activate.ps1
> cd project-1-llm-inference-optimization
> ```

**First run** — exports the model, builds the TRT engine, then benchmarks all three backends:
```powershell
python src/run_benchmark.py --export
```

**Subsequent runs** — models are cached in `models/`, so the export and build steps are skipped:
```powershell
python src/run_benchmark.py
```

**Partial runs:**
```powershell
python src/run_benchmark.py --backends pytorch onnx   # skip TRT
python src/run_benchmark.py --backends pytorch        # baseline only
python src/run_benchmark.py --iterations 200          # more samples
python src/run_benchmark.py --help                    # all options
```

Results are saved to `results/benchmark_<timestamp>.csv` and plots to `results/plots/`.

---

## Key design decisions

**Why `use_cache=False`?**  
Disabling the KV cache means all three backends do exactly the same computation — a full attention pass over all input tokens. With caching enabled, PyTorch and TRT would diverge in how they handle the cached state, making latency comparisons meaningless.

**Why CUDA events for timing?**  
`time.perf_counter()` measures wall-clock time including Python overhead and CPU↔GPU scheduling gaps. CUDA events are inserted directly into the GPU command stream and measure only the time the GPU spent on the operation.

**Why PyTorch tensors as TRT device buffers?**  
TRT's execution context accepts raw CUDA memory pointers (`data_ptr()`). Allocating buffers as PyTorch tensors avoids any dependency on `pycuda` or `cuda-python` while still giving TRT direct access to device memory.

**Why a dedicated CUDA stream for TRT?**  
Passing CUDA stream 0 (the default) to `execute_async_v3` forces TRT to insert extra `cudaStreamSynchronize` calls for safety. A dedicated `torch.cuda.Stream()` eliminates that overhead and ensures CUDA event timestamps bracket only the TRT kernel work.
