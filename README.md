# ML Infrastructure and Systems Engineering Portfolio

A collection of projects demonstrating GPU-accelerated ML inference, optimization, and systems programming.

**Environment:** Python 3.11 · CUDA 12.6 · Windows 11

---

## Setup

Projects 1, 3–7 share `.venv`. Project 2 uses a separate `.venv2` (WSL2, Docker).

**Projects 1, 3–6 (Windows PowerShell):**
```powershell
cd path\to\Nvidia
.\setup.ps1
.venv\Scripts\Activate.ps1
```

**Project 2 (WSL2):**
```bash
cd /mnt/c/Users/apoor/Desktop/projects/Nvidia/project-2-inference-server
bash setup_wsl.sh
source /mnt/c/Users/apoor/Desktop/projects/Nvidia/.venv2/bin/activate
```

---

## Projects

### [Project 1 — LLM Inference Optimization](project-1-llm-inference-optimization/)

Benchmark GPT-2 (124M) inference across four backends and measure the real-world impact of runtime and precision choices.

| Backend | Precision | Notes |
|---|---|---|
| PyTorch | FP32 | Baseline |
| PyTorch | FP16 | Tensor core path |
| PyTorch + SDPA | FP16 | Flash Attention via `attn_implementation="sdpa"` |
| PyTorch + torch.compile | FP16 | CUDA graphs via `backend="cudagraphs"` (Triton-free) |
| ONNX Runtime | FP16 | CUDAExecutionProvider |
| TensorRT | FP16 | Compiled engine, dynamic batch |

**Metrics collected:** latency (mean / p50 / p95 / p99), throughput (tokens/s), peak GPU memory

**Sweep:** batch sizes `[1, 4, 8, 16]` × sequence lengths `[64, 128, 256]`

**Skills:** ONNX export, TensorRT engine build, CUDA-event timing, matplotlib benchmarking charts

```powershell
cd project-1-llm-inference-optimization
python src/run_benchmark.py --export   # first run: exports ONNX + builds TRT engine
python src/run_benchmark.py            # subsequent runs reuse cached models
```

Results land in `project-1-llm-inference-optimization/results/` as CSV + PNG plots.

---

### [Project 2 — High-Throughput LLM Inference Server](project-2-inference-server/)

Benchmark five production-style serving backends for GPT-2 (FP16) on a single GPU, comparing latency, throughput, and GPU utilization.

| Backend | Description | Port |
|---|---|---|
| FastAPI + HuggingFace | Async baseline using `transformers.generate()` | 8000 |
| vLLM | PagedAttention-based continuous batching (Docker) | 8001 |
| Triton Inference Server | NVIDIA Triton Python backend (Docker) | 8002 |
| SGLang | RadixAttention-based inference server (Docker) | 8003 |
| TensorRT-LLM | Compiled TRT engines via TRT-LLM Python API (Docker) | 8004 |

**Results (single-client sequential · 50 requests · GPT-2 FP16 · 50 new tokens):**

| Server | p50 (ms) | Throughput (req/s) | GPU Util |
|--------|----------|--------------------|----------|
| FastAPI | 256.7 | 3.89 | 38.2% |
| Triton | 258.8 | 3.85 | 38.1% |
| vLLM | 85.0 | 11.56 | 62.2% |
| TRT-LLM | 84.9 | 11.73 | 75.2% |
| **SGLang** | **80.1** | **12.43** | **73.6%** |

**Skills:** FastAPI async serving, vLLM PagedAttention, SGLang RadixAttention, NVIDIA Triton Python backend, TensorRT-LLM engine compilation, Docker + NVIDIA Container Toolkit, WSL2 GPU passthrough, `tritonclient`, HTTP benchmarking

```bash
# FastAPI (venv)
uvicorn servers.fastapi_hf.server:app --port 8000

# vLLM / SGLang / Triton / TRT-LLM (all Docker)
bash servers/vllm_server/run.sh
bash servers/sglang_server/run.sh
bash servers/triton_server/run.sh
bash servers/trt_llm_server/run.sh

# Benchmark
python benchmark/benchmark.py --server fastapi   # or vllm / triton / sglang / trtllm
```

Results land in `project-2-inference-server/results/` as CSV + PNG plots.

---

### [Project 3 — GPU Profiling & Bottleneck Analysis](project-3-gpu-profiling/)

Profile GPT-2 (124M) PyTorch inference to identify where time is actually spent — kernel-level, CPU/GPU overlap, and memory.

| Tool | What it captures |
|---|---|
| `torch.profiler` | Per-kernel CUDA time, call counts, Chrome trace |
| CUDA events + `perf_counter` | GPU kernel time vs wall time, CPU overhead |
| `torch.cuda.memory_stats` | Peak allocated, peak reserved, fragmentation % |
| Nsight Systems 2024.4.2 | Full CUDA kernel timeline with NVTX range annotations |

**Sweep:** batch sizes `[1, 2, 4, 8, 16, 32]` × sequence length `128`

**Skills:** `torch.profiler` scheduling, CUDA event timing, NVTX annotation, Nsight Systems timeline analysis, bottleneck classification

```powershell
cd project-3-gpu-profiling
python src/run_profiler.py             # full run: kernel breakdown + bottlenecks + memory + plots
.\nsight\run_nsys.ps1                  # Nsight Systems timeline (admin PowerShell required)
```

Results land in `project-3-gpu-profiling/results/` as CSV + PNG plots + Chrome traces.

---

### [Project 4 — Quantization & Precision Optimization](project-4-quantization/)

Benchmark GPT-2 (124M) across four precision levels using pure PyTorch + `optimum-quanto` to isolate the effect of quantization on speed, memory, and accuracy — without backend variables (no TensorRT, no ONNX).

| Precision | Framework | Notes |
|---|---|---|
| FP32 | PyTorch | Baseline |
| FP16 | PyTorch `.half()` | Tensor core path |
| INT8 | optimum-quanto `qint8` | Weight-only, no calibration |
| INT4 | optimum-quanto `qint4` | Weight-only, no calibration |

**Metrics collected:** latency (mean / p50 / p95 / p99), throughput (tokens/s), peak GPU memory, perplexity on a fixed 20-sentence corpus

**Sweep:** batch sizes `[1, 4, 8, 16]` × sequence lengths `[64, 128, 256]`

**Skills:** post-training quantization, `optimum-quanto`, CUDA-event timing, perplexity evaluation, precision vs accuracy tradeoff analysis

```powershell
cd project-4-quantization
pip install optimum-quanto        # first time only
python src/run_benchmark.py       # full run: all four precisions + perplexity + plots
python src/run_benchmark.py --precisions fp32 fp16   # baselines only
python src/run_benchmark.py --no-perplexity          # latency/memory only
```

Results land in `project-4-quantization/results/` as CSV + PNG plots (latency, throughput, memory, speedup vs FP32, perplexity bar chart).

---

### [Project 5 — Deep Learning Workload Performance Analysis](project-5-workload-analysis/)

Compare four models across modalities (NLP and vision) to understand how architecture drives hardware efficiency — FLOPs, memory, latency, throughput, and GPU utilization (MFU).

| Model | Params | Modality |
|---|---|---|
| GPT-2 | 124M | NLP (decoder) |
| DistilGPT-2 | 82M | NLP (decoder) |
| BERT-base | 110M | NLP (encoder) |
| ResNet-50 | 25M | Vision (CNN) |

**Metrics collected:** FLOPs (analytical for transformers, torchinfo for CNNs), parameters, latency (mean / std / p50 / p95 / p99), throughput (tok/s or img/s), MFU (%), peak GPU memory (allocated + reserved), fragmentation %

**Sweep:** batch sizes `[1, 4, 8, 16, 32]` · NLP at seq=128 · Vision at 224×224

**Skills:** analytical FLOPs derivation, CUDA-event timing, MFU measurement, cross-modality architecture comparison, torchinfo

```powershell
cd project-5-workload-analysis
python src/run_analysis.py                              # full run: all models + plots
python src/run_analysis.py --models gpt2 bert_base     # subset of models
python src/run_analysis.py --peak-fp16-tflops 121.9    # override GPU peak (default: 74.4 for RTX 4080 Laptop)
```

Results land in `project-5-workload-analysis/results/` as CSV + PNG plots (architecture overview, latency sweep, throughput sweep, memory sweep, MFU sweep).

---

### [Project 6 — CUDA Kernel Optimization](project-6-cuda-kernel-optimization/)

Custom CUDA kernels written from scratch, compiled as a PyTorch extension, and benchmarked against CPU baselines and cuBLAS.

| Kernel | Variants | Key technique |
|---|---|---|
| Vector Addition | naive · float4 grid-stride | 128-bit loads, SM saturation |
| Matrix Multiplication | naive · tiled · cuBLAS | Shared memory tiling (TILE=16) |
| Parallel Reduction | naive · sequential · warp-shuffle | `__shfl_down_sync`, zero bank conflicts |

**Skills:** CUDA C++, `torch.utils.cpp_extension`, shared memory, warp-level primitives, memory bandwidth analysis, Nsight Systems / Nsight Compute profiling

```powershell
cd project-6-cuda-kernel-optimization
python src/run_benchmark.py           # all kernels, all sizes
python src/run_benchmark.py --no-cpu  # skip CPU baselines (faster)
.\nsight\run_nsys.ps1                 # Nsight Systems timeline
.\nsight\run_ncu.ps1                  # Nsight Compute per-kernel metrics
```

Results land in `project-6-cuda-kernel-optimization/results/` as CSV + PNG plots + Nsight reports.

---

### [Project 7 — Micrograd](project-7-micrograd/)

A from-scratch scalar-valued autograd engine and neural network library, following Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

| Component | File | Description |
|---|---|---|
| Autograd engine | `src/engine.py` | `Value` — scalar node with `.backward()` via reverse topo sort |
| Neural net primitives | `src/nn.py` | `Neuron`, `Layer`, `MLP` built entirely from `Value` arithmetic |
| Walkthrough notebook | `micrograd.ipynb` | Engine derivation, computation graph viz, MLP training loop |

**Skills:** reverse-mode automatic differentiation, computation graph construction, chain rule, manual SGD, scalar backprop

```powershell
cd project-7-micrograd
jupyter notebook micrograd.ipynb
```
