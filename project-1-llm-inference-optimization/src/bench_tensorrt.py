"""TensorRT FP16 inference benchmark for GPT-2.

Uses PyTorch CUDA tensors as device buffers so there is no dependency on
pycuda or cuda-python — TensorRT accepts raw data_ptr() integers directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

_LOGGER = trt.Logger(trt.Logger.WARNING)
_VOCAB = 50257  # GPT-2 vocab size


class _TRTSession:
    """Thin wrapper that loads a serialized TRT engine and manages GPU buffers."""

    def __init__(self, engine_path: str):
        runtime = trt.Runtime(_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()   # dedicated non-default stream
        self._buf_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    def _get_bufs(self, bs: int, seq_len: int) -> dict[str, torch.Tensor]:
        key = (bs, seq_len)
        if key not in self._buf_cache:
            self._buf_cache[key] = {
                "input_ids":      torch.empty(bs, seq_len, dtype=torch.int64, device="cuda"),
                "attention_mask": torch.empty(bs, seq_len, dtype=torch.int64, device="cuda"),
                "logits":         torch.empty(bs, seq_len, _VOCAB, dtype=torch.float32, device="cuda"),
            }
        return self._buf_cache[key]

    def infer(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
        bs, seq_len = input_ids.shape
        bufs = self._get_bufs(bs, seq_len)

        bufs["input_ids"].copy_(torch.from_numpy(input_ids))
        bufs["attention_mask"].copy_(torch.from_numpy(attention_mask))

        self.context.set_input_shape("input_ids",      (bs, seq_len))
        self.context.set_input_shape("attention_mask", (bs, seq_len))
        self.context.set_tensor_address("input_ids",      bufs["input_ids"].data_ptr())
        self.context.set_tensor_address("attention_mask", bufs["attention_mask"].data_ptr())
        self.context.set_tensor_address("logits",         bufs["logits"].data_ptr())

        self.context.execute_async_v3(self.stream.cuda_stream)


def _time_fn(fn, warmup: int, iterations: int, stream: torch.cuda.Stream | None = None) -> np.ndarray:
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start_ev.record(stream)
        fn()
        end_ev.record(stream)
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    return np.array(times, dtype=np.float32)


def benchmark(
    engine_path: str,
    batch_sizes: list[int] = [1, 4, 8, 16],
    seq_lens: list[int] = [64, 128, 256],
    warmup: int = 10,
    iterations: int = 100,
) -> list[dict]:
    if not Path(engine_path).exists():
        raise FileNotFoundError(f"TRT engine not found: {engine_path}")

    print(f"Loading TRT engine from {engine_path} …")
    sess = _TRTSession(engine_path)
    results = []

    for bs in batch_sizes:
        for seq_len in seq_lens:
            ids_np = np.random.randint(0, 50257, (bs, seq_len), dtype=np.int64)
            mask_np = np.ones((bs, seq_len), dtype=np.int64)

            def fn():
                sess.infer(ids_np, mask_np)

            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

            times = _time_fn(fn, warmup, iterations, stream=sess.stream)
            mean_ms = float(times.mean())
            throughput = (bs * seq_len) / (mean_ms / 1_000)

            row = {
                "backend": "tensorrt_fp16",
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
