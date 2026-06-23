"""Main entry point for the GPU Profiling & Bottleneck Analysis suite.

Usage examples
--------------
# Full profiling run — all three modules, all batch sizes
python src/run_profiler.py

# Custom batch sizes and sequence length
python src/run_profiler.py --batch-sizes 1 4 16 32 --seq-len 64

# Skip plot generation
python src/run_profiler.py --no-plot

# All options
python src/run_profiler.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KINETO_LOG_LEVEL", "5")  # suppress libkineto USDT trace spam

import pandas as pd
import torch

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
TRACES_DIR  = RESULTS_DIR / "traces"
REPORTS_DIR = RESULTS_DIR / "reports"
PLOTS_DIR   = RESULTS_DIR / "plots"

sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 GPU Profiling & Bottleneck Analysis")
    p.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32],
        metavar="N", help="Batch sizes to sweep (default: 1 2 4 8 16 32)",
    )
    p.add_argument(
        "--seq-len", type=int, default=128,
        metavar="N", help="Sequence length (default: 128)",
    )
    p.add_argument(
        "--warmup", type=int, default=5,
        metavar="N", help="Warmup iterations per config (default: 5)",
    )
    p.add_argument(
        "--iterations", type=int, default=20,
        metavar="N", help="Timed iterations for bottleneck module (default: 20)",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    for d in (RESULTS_DIR, TRACES_DIR, REPORTS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice      : {torch.cuda.get_device_name(0)}")
    print(f"Batch sizes : {args.batch_sizes}")
    print(f"Seq length  : {args.seq_len}")
    print(f"Warmup      : {args.warmup}   Iterations : {args.iterations}")

    from profile_torch       import profile_kernels
    from profile_bottlenecks import profile_bottlenecks
    from profile_memory      import profile_memory

    kernel_rows     = profile_kernels(args.batch_sizes, args.seq_len, TRACES_DIR, REPORTS_DIR, args.warmup)
    bottleneck_rows = profile_bottlenecks(args.batch_sizes, args.seq_len, args.warmup, args.iterations)
    memory_rows     = profile_memory(args.batch_sizes, args.seq_len, args.warmup)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(kernel_rows).to_csv(RESULTS_DIR / f"kernels_{ts}.csv",     index=False)
    pd.DataFrame(bottleneck_rows).to_csv(RESULTS_DIR / f"bottlenecks_{ts}.csv", index=False)
    pd.DataFrame(memory_rows).to_csv(RESULTS_DIR / f"memory_{ts}.csv",      index=False)

    print(f"\nResults saved → results/kernels_{ts}.csv")
    print(f"             → results/bottlenecks_{ts}.csv")
    print(f"             → results/memory_{ts}.csv")

    if not args.no_plot:
        print("\n=== Generating plots ===")
        from plot_results import plot_all
        plot_all(kernel_rows, bottleneck_rows, memory_rows, PLOTS_DIR)


if __name__ == "__main__":
    main()
