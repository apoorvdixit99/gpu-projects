"""Main entry point for the Deep Learning Workload Performance Analysis.

Usage examples
--------------
# Full analysis — all four models, all batch sizes
python src/run_analysis.py

# Custom model selection
python src/run_analysis.py --models gpt2 distilgpt2

# Custom batch sizes
python src/run_analysis.py --batch-sizes 1 8 32

# Specify GPU peak FP16 TFLOPS (default: 74.4 for RTX 4080 Laptop GPU)
python src/run_analysis.py --peak-fp16-tflops 74.4

# Skip plot generation
python src/run_analysis.py --no-plot

# Single module directly
python src/measure_flops.py
python src/measure_latency.py
python src/measure_memory.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

sys.path.insert(0, str(Path(__file__).parent))

from models import MODEL_MAP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep Learning Workload Performance Analysis")
    p.add_argument(
        "--models", nargs="+",
        default=list(MODEL_MAP.keys()),
        choices=list(MODEL_MAP.keys()),
        metavar="NAME",
        help=f"Models to analyse (default: all). Choices: {list(MODEL_MAP.keys())}",
    )
    p.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[1, 4, 8, 16, 32],
        metavar="N",
        help="Batch sizes to sweep (default: 1 4 8 16 32)",
    )
    p.add_argument(
        "--warmup", type=int, default=10, metavar="N",
        help="Warmup iterations per config (default: 10)",
    )
    p.add_argument(
        "--iterations", type=int, default=50, metavar="N",
        help="Timed iterations per config (default: 50)",
    )
    p.add_argument(
        "--peak-fp16-tflops", type=float, default=74.4, metavar="T",
        help="GPU peak FP16 TFLOPS for MFU (default: 74.4 for RTX 4080 Laptop GPU)",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    for d in (RESULTS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    specs = [MODEL_MAP[name] for name in args.models]

    print(f"\nDevice          : {torch.cuda.get_device_name(0)}")
    print(f"Models          : {[s.label for s in specs]}")
    print(f"Batch sizes     : {args.batch_sizes}")
    print(f"Warmup / iters  : {args.warmup} / {args.iterations}")
    print(f"Peak FP16       : {args.peak_fp16_tflops:.1f} TFLOPS")

    from measure_flops   import measure_flops
    from measure_latency import measure_latency
    from measure_memory  import measure_memory

    # ── FLOPs ────────────────────────────────────────────────────────────────
    flops_rows   = measure_flops(specs)
    flops_lookup = {r["model"]: int(r["flops_g"] * 1e9) for r in flops_rows}

    # ── Latency + MFU ────────────────────────────────────────────────────────
    latency_rows = measure_latency(
        specs,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        iterations=args.iterations,
        peak_fp16_tflops=args.peak_fp16_tflops,
        flops_per_sample=flops_lookup,
    )

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_rows = measure_memory(specs, args.batch_sizes, warmup=args.warmup)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(flops_rows).to_csv(RESULTS_DIR / f"flops_{ts}.csv",   index=False)
    pd.DataFrame(latency_rows).to_csv(RESULTS_DIR / f"latency_{ts}.csv", index=False)
    pd.DataFrame(memory_rows).to_csv(RESULTS_DIR / f"memory_{ts}.csv",  index=False)

    print(f"\n{'='*60}")
    print("\n--- Architecture summary ---")
    print(pd.DataFrame(flops_rows)[["label", "params_m", "flops_g"]].to_string(index=False))

    print(f"\nResults saved -> results/flops_{ts}.csv")
    print(f"             -> results/latency_{ts}.csv")
    print(f"             -> results/memory_{ts}.csv")

    if not args.no_plot:
        print("\n=== Generating plots ===")
        from plot_results import plot_all
        plot_all(flops_rows, latency_rows, memory_rows, PLOTS_DIR)


if __name__ == "__main__":
    main()
