# High-Throughput LLM Inference Server

Benchmarks five production-style serving backends for GPT-2 (FP16) on a single GPU, comparing latency, throughput, and GPU utilization across a plain HuggingFace server, two PagedAttention engines, NVIDIA's production inference runtime, and a compiled TensorRT engine.

**Hardware:** NVIDIA RTX 4080 Laptop GPU (Ada Lovelace) · CUDA 12.6 (driver 560.76)  
**Model:** GPT-2 124M (HuggingFace `gpt2`, FP16)  
**Environment:** WSL2 (Ubuntu) on Windows 11

---

## Backends

| Server | Description | Port | Runtime |
|--------|-------------|------|---------|
| **FastAPI + HuggingFace** | Async baseline using `transformers.generate()` | 8000 | Python venv |
| **vLLM** | PagedAttention-based continuous batching server | 8001 | Docker |
| **Triton Inference Server** | NVIDIA Triton with Python backend | 8002 | Docker |
| **SGLang** | RadixAttention-based inference server | 8003 | Docker |
| **TensorRT-LLM** | Compiled TRT engines via TRT-LLM Python API | 8004 | Docker |

---

## Results

Single-client sequential benchmark · 5 warmup + 50 measured requests · prompt: `"The future of artificial intelligence is"` · `max_new_tokens=50`

| Server | p50 (ms) | p95 (ms) | Throughput (req/s) | GPU Util |
|--------|----------|----------|--------------------|----------|
| FastAPI | 256.7 | 268.6 | 3.89 | 38.2% |
| Triton | 258.8 | 268.4 | 3.85 | 38.1% |
| vLLM | 85.0 | 93.8 | 11.56 | 62.2% |
| **SGLang** | **80.1** | **83.6** | **12.43** | **73.6%** |
| TRT-LLM | — | — | — | — |

### Key findings

**SGLang is the fastest** at 80ms p50 and 12.43 req/s — edging out vLLM (85ms, 11.56 req/s) by about 7%. The difference is visible in GPU utilization: SGLang keeps the GPU at 73.6% vs vLLM's 62.2%, suggesting its RadixAttention KV cache reuse is squeezing more compute out of each request.

**vLLM is a close second**, roughly 3× faster than FastAPI and Triton. Both SGLang and vLLM benefit from continuous batching and efficient KV cache management — neither is scheduling idle GPU time between requests the way FastAPI does.

**FastAPI ≈ Triton** at ~257ms p50 and ~3.87 req/s. Both process requests one at a time with no batching. The GPU fires for the generation window then sits idle waiting for the next HTTP round-trip, which is why utilization is only 38%.

**GPU utilization is the signal:** 38% (FastAPI/Triton) vs 62% (vLLM) vs 73% (SGLang) tells you how efficiently each framework schedules work onto the GPU.

> Note: single-client sequential benchmark. The gap between optimized engines and naive serving compounds sharply under concurrent load — at 32 clients, vLLM and SGLang's throughput advantage would be far larger.

> TRT-LLM results pending — first run compiles GPT-2 into TRT engines (~2-3 min).

---

## Project Structure

```
project-2-inference-server/
├── servers/
│   ├── fastapi_hf/
│   │   └── server.py              # HuggingFace + FastAPI server
│   ├── vllm_server/
│   │   └── run.sh                 # docker run vllm/vllm-openai:v0.6.6.post1
│   ├── triton_server/
│   │   ├── Dockerfile             # tritonserver:23.10-py3 + torch cu124
│   │   ├── run.sh                 # build image + run container
│   │   └── model_repo/gpt2/
│   │       ├── config.pbtxt       # Triton Python backend config
│   │       └── 1/model.py         # GPT-2 generation backend
│   ├── sglang_server/
│   │   └── run.sh                 # docker run lmsysorg/sglang:v0.4.6.post1-cu124
│   └── trt_llm_server/
│       ├── Dockerfile             # tritonserver:24.08-trtllm-python-py3 + fastapi
│       ├── server.py              # FastAPI wrapper using tensorrt_llm.LLM
│       └── run.sh                 # build image + run container
├── benchmark/
│   └── benchmark.py              # p50/p95/throughput/GPU util
├── results/                       # CSV + plots (git-ignored)
├── requirements.txt
└── setup_wsl.sh
```

---

## Requirements

- WSL2 (Ubuntu) with NVIDIA GPU passthrough
- CUDA 12.6 driver (Windows driver 560.76+)
- Python 3.12 with `.venv2` (see `setup_wsl.sh`)
- Docker Engine with NVIDIA Container Toolkit

---

## Setup

```bash
# 1. Verify GPU access in WSL2
nvidia-smi

# 2. Create venv and install dependencies
bash setup_wsl.sh

# 3. Activate venv (required before FastAPI server and benchmark)
source /mnt/c/Users/apoor/Desktop/projects/Nvidia/.venv2/bin/activate

# 4. Verify Docker has GPU access
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

---

## Running the Servers

Each server runs in its own terminal. Stop with `Ctrl+C` before starting the next — all servers need full GPU access.

### FastAPI (HuggingFace)

```bash
source /mnt/c/Users/apoor/Desktop/projects/Nvidia/.venv2/bin/activate
cd /mnt/c/Users/apoor/Desktop/projects/Nvidia/project-2-inference-server
uvicorn servers.fastapi_hf.server:app --port 8000
```

Ready when you see: `Application startup complete.`

### vLLM

```bash
bash servers/vllm_server/run.sh
```

Pulls `vllm/vllm-openai:v0.6.6.post1` on first run. Ready when you see: `Application startup complete.`

### Triton Inference Server

```bash
bash servers/triton_server/run.sh
```

Builds a custom image on first run (~5 min). Ready when you see: `Started HTTPService at 0.0.0.0:8000`

### SGLang

```bash
bash servers/sglang_server/run.sh
```

Pulls `lmsysorg/sglang:v0.4.6.post1-cu124` on first run (~10 GB). Ready when you see: `The server is fired up and ready to roll!`

### TensorRT-LLM

```bash
bash servers/trt_llm_server/run.sh
```

Builds a custom image on first run (~5 min), then compiles GPT-2 TRT engines on first startup (~2-3 min, cached after). Ready when you see: `TRT-LLM engine ready.`

---

## Running the Benchmark

```bash
source /mnt/c/Users/apoor/Desktop/projects/Nvidia/.venv2/bin/activate
cd /mnt/c/Users/apoor/Desktop/projects/Nvidia/project-2-inference-server

python benchmark/benchmark.py --server fastapi
python benchmark/benchmark.py --server vllm
python benchmark/benchmark.py --server triton
python benchmark/benchmark.py --server sglang
python benchmark/benchmark.py --server trtllm
```

Results are saved to `results/benchmark_results.csv`. Plots are generated when two or more servers are benchmarked together (`--server all`).

---

## Key Design Decisions

**Why sequential single-client benchmarking?**  
This isolates per-request latency without confounding it with concurrency effects. The optimized engines' batching advantage compounds under concurrent load — a single-client test gives a clean latency floor for each backend.

**Why client-side latency measurement?**  
The benchmark measures end-to-end round-trip time (`time.perf_counter()`), not server-side generation time. HTTP serialization overhead is identical across backends, making the comparison fair.

**Why `do_sample=False` (greedy decoding)?**  
Greedy decoding is deterministic, eliminating sampling randomness from the latency distribution. All backends produce identical output for the same prompt — a pure infrastructure comparison.

**Why pin Docker image versions?**  
`vllm/vllm-openai:latest` as of mid-2026 requires CUDA 13.0. Pinning to `v0.6.6.post1` ensures compatibility with the CUDA 12.6 driver. Same logic applies to the SGLang and Triton images. See `ISSUES.md` for the full history.
