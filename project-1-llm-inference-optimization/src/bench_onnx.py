"""ONNX Runtime (CUDAExecutionProvider) inference benchmark for GPT-2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def _make_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    active = sess.get_providers()
    if "CUDAExecutionProvider" not in active:
        print(f"  Warning: CUDAExecutionProvider not active — using {active}")
    return sess


def _time_fn(fn, warmup: int, iterations: int) -> np.ndarray:
    """Time using CUDA events (works even for ORT because it shares the CUDA device)."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    return np.array(times, dtype=np.float32)


def benchmark(
    onnx_path: str,
    batch_sizes: list[int] = [1, 4, 8, 16],
    seq_lens: list[int] = [64, 128, 256],
    warmup: int = 10,
    iterations: int = 100,
) -> list[dict]:
    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print(f"Loading ONNX session from {onnx_path} …")
    sess = _make_session(onnx_path)
    input_names = [inp.name for inp in sess.get_inputs()]

    results = []

    for bs in batch_sizes:
        for seq_len in seq_lens:
            ids_np = np.random.randint(0, 50257, (bs, seq_len), dtype=np.int64)
            mask_np = np.ones((bs, seq_len), dtype=np.int64)
            feeds = {"input_ids": ids_np, "attention_mask": mask_np}

            def fn():
                sess.run(None, feeds)

            # Warmup to let ORT allocate GPU memory, then snapshot peak.
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            # ORT allocates through CUDA but outside PyTorch's allocator;
            # use the broader pynvml reading in run_benchmark.py if needed.
            # Here we record whatever PyTorch can see (activations may be partial).
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

            times = _time_fn(fn, warmup, iterations)
            mean_ms = float(times.mean())
            throughput = (bs * seq_len) / (mean_ms / 1_000)

            row = {
                "backend": "onnxruntime_fp16",
                "batch_size": bs,
                "seq_len": seq_len,
                "latency_ms_mean": round(mean_ms, 3),
                "latency_ms_std": round(float(times.std()), 3),
                "latency_ms_p50": round(float(np.percentile(times, 50)), 3),
                "latency_ms_p95": round(float(np.percentile(times, 95)), 3),
                "latency_ms_p99": round(float(np.percentile(times, 99)), 3),
                "throughput_tok_per_sec": round(throughput),
                "gpu_memory_mb": round(peak_mem_mb, 1),
            }
            results.append(row)
            print(
                f"  bs={bs:2d}  seq={seq_len:3d} | "
                f"{mean_ms:7.2f} ms ±{times.std():.2f} | "
                f"{throughput:>8,.0f} tok/s | "
                f"{peak_mem_mb:.0f} MB"
            )

    return results
