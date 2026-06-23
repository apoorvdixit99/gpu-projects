# High-Throughput LLM Inference Server

Benchmarks three production-style serving backends for GPT-2 (FP16) on a single GPU, comparing latency, throughput, and GPU utilization across a plain HuggingFace server, a PagedAttention-based engine, and NVIDIA's production inference runtime.

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

---

## Results

Single-client sequential benchmark · 5 warmup + 50 measured requests · prompt: `"The future of artificial intelligence is"` · `max_new_tokens=50`

| Server  | p50 (ms) | p95 (ms) | Throughput (req/s) | GPU Util |
|---------|----------|----------|--------------------|----------|
| FastAPI | 256.7    | 268.6    | 3.89               | 38.2%    |
| vLLM    | **85.0** | **93.8** | **11.56**          | **62.2%**|
| Triton  | 258.8    | 268.4    | 3.85               | 38.1%    |

### Key findings

**vLLM is 3× faster** than both FastAPI and Triton on this single-client sequential workload. The gap comes from PagedAttention's KV cache management and continuous batching, which keeps the GPU busier between requests (62% vs 38% utilization).

**FastAPI ≈ Triton** in latency and throughput. Both process one request at a time with no batching. Triton adds Python backend overhead that roughly cancels out any scheduling benefit.

**GPU utilization is the signal**: at 38%, both FastAPI and Triton are mostly idle between requests — the GPU fires for ~100ms then waits for the next HTTP roundtrip. vLLM's scheduler minimizes that idle time.

> Note: this benchmark is single-client sequential (one request at a time). vLLM's batching advantage compounds significantly under concurrent load — its throughput advantage over HuggingFace would be far larger with multiple parallel clients.

---

## Project Structure

```
project-2-inference-server/
├── servers/
│   ├── fastapi_hf/
│   │   └── server.py              # HuggingFace + FastAPI server
│   ├── vllm_server/
│   │   └── run.sh                 # docker run vllm/vllm-openai:v0.6.6.post1
│   └── triton_server/
│       ├── Dockerfile             # tritonserver:23.10-py3 + torch cu124
│       ├── run.sh                 # build image + run container
│       └── model_repo/gpt2/
│           ├── config.pbtxt       # Triton Python backend config
│           └── 1/model.py         # GPT-2 generation backend
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
- Docker Engine with NVIDIA Container Toolkit (for vLLM and Triton)

---

## Setup

```bash
# 1. Verify GPU access in WSL2
nvidia-smi

# 2. Create venv and install dependencies
bash setup_wsl.sh

# 3. Activate venv (required before running any server or benchmark)
source /mnt/c/Users/apoor/Desktop/projects/Nvidia/.venv2/bin/activate

# 4. Verify Docker has GPU access
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

---

## Running the Servers

Each server runs in its own terminal with the venv activated.

### FastAPI (HuggingFace)

```bash
cd /mnt/c/Users/apoor/Desktop/projects/Nvidia/project-2-inference-server
uvicorn servers.fastapi_hf.server:app --port 8000
```

Ready when you see: `Application startup complete.`

### vLLM

```bash
bash servers/vllm_server/run.sh
```

Pulls `vllm/vllm-openai:v0.6.6.post1` (~8 GB, first run only). Ready when you see: `Application startup complete.`

### Triton Inference Server

```bash
bash servers/triton_server/run.sh
```

Builds a custom image on first run (~5 min). Ready when you see: `Started HTTPService at 0.0.0.0:8000`

---

## Running the Benchmark

With a server running, benchmark it from a separate terminal:

```bash
source /mnt/c/Users/apoor/Desktop/projects/Nvidia/.venv2/bin/activate
cd /mnt/c/Users/apoor/Desktop/projects/Nvidia/project-2-inference-server

python benchmark/benchmark.py --server fastapi   # FastAPI on :8000
python benchmark/benchmark.py --server vllm      # vLLM on :8001
python benchmark/benchmark.py --server triton    # Triton on :8002
```

Run all three (each server must be running):

```bash
python benchmark/benchmark.py --server all
```

Results are saved to `results/benchmark_results.csv`. Plots are generated when two or more servers are benchmarked together.

---

## Key Design Decisions

**Why sequential single-client benchmarking?**  
This isolates per-request latency without confounding it with concurrency effects. vLLM's batching advantage is real but only appears under concurrent load — a single-client test gives a clean latency floor for each backend.

**Why client-side latency measurement?**  
The benchmark measures end-to-end round-trip time from the client (`time.perf_counter()`), not the server-side generation time. This includes HTTP serialization overhead, which is the same for all three backends and makes the comparison fair.

**Why `do_sample=False` (greedy decoding)?**  
Greedy decoding is deterministic, eliminating sampling randomness from the latency distribution. All three backends produce identical output for the same prompt, making the benchmark a pure infrastructure comparison.

**Why pin Docker image versions?**  
`vllm/vllm-openai:latest` tracks the newest vLLM release, which as of mid-2026 requires CUDA 13.0. Pinning to `v0.6.6.post1` ensures compatibility with the CUDA 12.6 driver. Same logic applies to the Triton base image.
