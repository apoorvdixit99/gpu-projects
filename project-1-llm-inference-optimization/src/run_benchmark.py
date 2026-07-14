"""Main entry point for the GPT-2 inference benchmark.

Usage examples
--------------
# First-time: export ONNX + build TRT, then run all backends
python src/run_benchmark.py --export

# Re-run without re-building
python src/run_benchmark.py

# Only PyTorch (no ONNX / TRT needed)
python src/run_benchmark.py --backends pytorch

# Custom sweep
python src/run_benchmark.py --batch-sizes 1 8 32 --seq-lens 128 512 --iterations 50
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 Inference Benchmark")
    p.add_argument(
        "--backends", nargs="+",
        default=["pytorch_fp32", "pytorch", "pytorch_sdpa", "pytorch_compile", "onnx", "tensorrt"],
        choices=["pytorch_fp32", "pytorch", "pytorch_sdpa", "pytorch_compile", "onnx", "tensorrt"],
        help="Which backends to benchmark (default: all six)",
    )
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16])
    p.add_argument("--seq-lens",    nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--warmup",      type=int,  default=10,  help="Warmup iterations")
    p.add_argument("--iterations",  type=int,  default=100, help="Timed iterations")
    p.add_argument(
        "--fp16", action="store_true", default=True,
        help="Use FP16 for PyTorch and TRT (default: True)",
    )
    p.add_argument(
        "--export", action="store_true",
        help="(Re-)export ONNX and (re-)build TRT engine before benchmarking",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    onnx_fp32_path = MODELS_DIR / "gpt2_fp32.onnx"
    onnx_fp16_path = MODELS_DIR / "gpt2_fp16.onnx"
    trt_path       = MODELS_DIR / ("gpt2_fp16.trt" if args.fp16 else "gpt2_fp32.trt")

    # ── Model export / engine build ──────────────────────────────────────────
    if "onnx" in args.backends or "tensorrt" in args.backends:
        if args.export or not onnx_fp16_path.exists():
            print("\n=== Exporting GPT-2 → ONNX (FP16) ===")
            from export_onnx import export
            export(str(onnx_fp16_path), fp16=True)

    if "tensorrt" in args.backends:
        if args.export or not trt_path.exists():
            print("\n=== Building TensorRT engine ===")
            from build_trt import build
            build(
                str(onnx_fp16_path), str(trt_path),
                fp16=args.fp16,
                batch_sizes=args.batch_sizes,
                seq_lens=args.seq_lens,
            )

    # ── Benchmarks ───────────────────────────────────────────────────────────
    all_results: list[dict] = []

    if "pytorch_fp32" in args.backends:
        print("\n=== PyTorch FP32 Benchmark ===")
        from bench_pytorch import benchmark
        all_results.extend(
            benchmark(args.batch_sizes, args.seq_lens, fp16=False, warmup=args.warmup, iterations=args.iterations)
        )

    if "pytorch" in args.backends:
        print("\n=== PyTorch FP16 Benchmark ===")
        from bench_pytorch import benchmark
        all_results.extend(
            benchmark(args.batch_sizes, args.seq_lens, args.fp16, args.warmup, args.iterations)
        )

    if "pytorch_sdpa" in args.backends:
        print("\n=== PyTorch SDPA Benchmark ===")
        from bench_pytorch import benchmark
        all_results.extend(
            benchmark(args.batch_sizes, args.seq_lens, args.fp16, args.warmup, args.iterations, mode="sdpa")
        )

    if "pytorch_compile" in args.backends:
        print("\n=== PyTorch torch.compile Benchmark ===")
        from bench_pytorch import benchmark
        all_results.extend(
            benchmark(args.batch_sizes, args.seq_lens, args.fp16, args.warmup, args.iterations, mode="compile")
        )

    if "onnx" in args.backends:
        print("\n=== ONNX Runtime Benchmark ===")
        from bench_onnx import benchmark
        all_results.extend(
            benchmark(str(onnx_fp16_path), args.batch_sizes, args.seq_lens, args.warmup, args.iterations)
        )

    if "tensorrt" in args.backends:
        print("\n=== TensorRT Benchmark ===")
        from bench_tensorrt import benchmark
        all_results.extend(
            benchmark(str(trt_path), args.batch_sizes, args.seq_lens, args.warmup, args.iterations)
        )

    # ── Save results ─────────────────────────────────────────────────────────
    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"benchmark_{ts}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\nResults saved → {csv_path}")

    if not args.no_plot:
        print("\n=== Generating plots ===")
        from plot_results import plot
        plot(df, RESULTS_DIR / "plots")


if __name__ == "__main__":
    main()
