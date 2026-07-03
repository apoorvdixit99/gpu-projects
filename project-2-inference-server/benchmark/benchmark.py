"""
Single-client benchmark: FastAPI (HuggingFace) vs vLLM vs Triton
Measures p50/p95 latency, throughput, and GPU utilization.
"""
import argparse
import subprocess
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import tritonclient.http as httpclient
import matplotlib.pyplot as plt


PROMPT = "The future of artificial intelligence is"
MAX_NEW_TOKENS = 50
WARMUP_REQUESTS = 5
BENCHMARK_REQUESTS = 50

SERVERS = {
    "fastapi": "http://localhost:8000",
    "vllm":    "http://localhost:8001",
    "triton":  "http://localhost:8002",
    "sglang":  "http://localhost:8003",
    "trtllm":  "http://localhost:8004",
}


# --------------------------------------------------------------------------- #
# GPU utilization sampler (background thread)
# --------------------------------------------------------------------------- #

class GpuSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return float(np.mean(self.samples)) if self.samples else 0.0

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    text=True,
                )
                self.samples.append(float(out.strip().split("\n")[0]))
            except Exception:
                pass
            time.sleep(self.interval)


# --------------------------------------------------------------------------- #
# Per-server request helpers
# --------------------------------------------------------------------------- #

def call_fastapi(client: httpx.Client) -> float:
    t0 = time.perf_counter()
    r = client.post(
        f"{SERVERS['fastapi']}/generate",
        json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS},
        timeout=120,
    )
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def call_vllm(client: httpx.Client) -> float:
    t0 = time.perf_counter()
    r = client.post(
        f"{SERVERS['vllm']}/v1/completions",
        json={
            "model": "gpt2",
            "prompt": PROMPT,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
        },
        timeout=120,
    )
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def call_sglang(client: httpx.Client) -> float:
    t0 = time.perf_counter()
    r = client.post(
        f"{SERVERS['sglang']}/v1/completions",
        json={
            "model": "gpt2",
            "prompt": PROMPT,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
        },
        timeout=120,
    )
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def call_trtllm(client: httpx.Client) -> float:
    t0 = time.perf_counter()
    r = client.post(
        f"{SERVERS['trtllm']}/generate",
        json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS},
        timeout=120,
    )
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def call_triton(triton_client: httpclient.InferenceServerClient) -> float:
    prompt_input = httpclient.InferInput("prompt", [1], "BYTES")
    prompt_input.set_data_from_numpy(
        np.array([PROMPT.encode("utf-8")], dtype=object)
    )

    tokens_input = httpclient.InferInput("max_new_tokens", [1], "INT32")
    tokens_input.set_data_from_numpy(
        np.array([MAX_NEW_TOKENS], dtype=np.int32)
    )

    output = httpclient.InferRequestedOutput("generated_text")

    t0 = time.perf_counter()
    triton_client.infer("gpt2", [prompt_input, tokens_input], outputs=[output])
    return (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------- #
# Benchmark runner
# --------------------------------------------------------------------------- #

def run_benchmark(server: str) -> dict:
    print(f"\n{'=' * 50}")
    print(f"Benchmarking: {server.upper()}")
    print(f"{'=' * 50}")

    if server == "triton":
        client = httpclient.InferenceServerClient(
            url="localhost:8002", verbose=False
        )
        call_fn = lambda _: call_triton(client)
        http_client = None
    else:
        http_client = httpx.Client()
        call_fn = {"fastapi": call_fastapi, "vllm": call_vllm, "sglang": call_sglang, "trtllm": call_trtllm}[server]

    # Warmup
    print(f"  Warmup ({WARMUP_REQUESTS} requests)...")
    for _ in range(WARMUP_REQUESTS):
        call_fn(http_client)

    # Benchmark
    print(f"  Benchmark ({BENCHMARK_REQUESTS} requests)...")
    sampler = GpuSampler()
    sampler.start()

    latencies: list[float] = []
    bench_start = time.perf_counter()
    for i in range(BENCHMARK_REQUESTS):
        latencies.append(call_fn(http_client))
        print(f"    [{i+1:>2}/{BENCHMARK_REQUESTS}] {latencies[-1]:.1f} ms", end="\r")

    total_time = time.perf_counter() - bench_start
    gpu_util = sampler.stop()

    if http_client:
        http_client.close()

    latencies_arr = np.array(latencies)
    results = {
        "server":        server,
        "p50_ms":        float(np.percentile(latencies_arr, 50)),
        "p95_ms":        float(np.percentile(latencies_arr, 95)),
        "mean_ms":       float(np.mean(latencies_arr)),
        "throughput_rps": round(BENCHMARK_REQUESTS / total_time, 2),
        "gpu_util_pct":  round(gpu_util, 1),
    }

    print(f"\n  p50={results['p50_ms']:.1f}ms  "
          f"p95={results['p95_ms']:.1f}ms  "
          f"throughput={results['throughput_rps']:.2f} req/s  "
          f"GPU={results['gpu_util_pct']}%")

    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def save_plots(df: pd.DataFrame, out_dir: Path):
    servers = df["server"].tolist()
    x = np.arange(len(servers))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("LLM Inference Server Benchmark — GPT-2 (FP16)", fontsize=14)

    # Latency (p50 + p95)
    ax = axes[0]
    w = 0.35
    ax.bar(x - w/2, df["p50_ms"], w, label="p50", color=colors, alpha=0.85)
    ax.bar(x + w/2, df["p95_ms"], w, label="p95", color=colors, alpha=0.45,
           edgecolor=colors, linewidth=1.2)
    ax.set_title("Latency (ms) — lower is better")
    ax.set_xticks(x); ax.set_xticklabels(servers)
    ax.set_ylabel("ms")
    ax.legend()

    # Throughput
    ax = axes[1]
    bars = ax.bar(servers, df["throughput_rps"], color=colors, alpha=0.85)
    ax.set_title("Throughput (req/s) — higher is better")
    ax.set_ylabel("requests / second")
    for bar, val in zip(bars, df["throughput_rps"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # GPU utilization
    ax = axes[2]
    bars = ax.bar(servers, df["gpu_util_pct"], color=colors, alpha=0.85)
    ax.set_title("Avg GPU Utilization (%) — higher is better")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, df["gpu_util_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = out_dir / "benchmark_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="LLM serving benchmark")
    parser.add_argument(
        "--server",
        choices=["fastapi", "vllm", "triton", "sglang", "trtllm", "all"],
        default="all",
        help="Which server to benchmark",
    )
    args = parser.parse_args()

    targets = ["fastapi", "vllm", "triton", "sglang", "trtllm"] if args.server == "all" else [args.server]

    print(f"\nPrompt        : '{PROMPT}'")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Warmup        : {WARMUP_REQUESTS}")
    print(f"Requests      : {BENCHMARK_REQUESTS}")

    all_results = []
    for server in targets:
        try:
            all_results.append(run_benchmark(server))
        except Exception as e:
            print(f"\n  ERROR benchmarking {server}: {e}")
            print(f"  Is the {server} server running?")

    if not all_results:
        return

    df = pd.DataFrame(all_results)
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved : {csv_path}")

    print("\n" + "=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)

    if len(all_results) > 1:
        save_plots(df, out_dir)


if __name__ == "__main__":
    main()
