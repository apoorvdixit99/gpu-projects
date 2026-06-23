"""Main entry point for the CUDA Kernel Optimization benchmark suite.

Usage examples
--------------
# Full run — all three kernels, all default sizes
python src/run_benchmark.py

# Custom sizes
python src/run_benchmark.py --vec-sizes 1048576 16777216 268435456
python src/run_benchmark.py --mat-sizes 256 512 1024 2048

# Faster run with fewer iterations
python src/run_benchmark.py --warmup 5 --iterations 50

# Skip CPU baselines (much faster for large problem sizes)
python src/run_benchmark.py --no-cpu

# Skip plot generation
python src/run_benchmark.py --no-plot
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CUDA Kernel Optimization Benchmark")
    p.add_argument(
        "--vec-sizes", nargs="+", type=int,
        default=[2**20, 2**22, 2**24, 2**26, 2**28],
        metavar="N",
        help="Element counts for vector-add / reduction (default: 1M 4M 16M 64M 256M)",
    )
    p.add_argument(
        "--mat-sizes", nargs="+", type=int,
        default=[256, 512, 1024, 2048],
        metavar="N",
        help="Square matrix side lengths (default: 256 512 1024 2048)",
    )
    p.add_argument("--warmup",     type=int, default=10,  metavar="N",
                   help="Warmup iterations per config (default: 10)")
    p.add_argument("--iterations", type=int, default=100, metavar="N",
                   help="Timed iterations per config (default: 100)")
    p.add_argument("--no-cpu",  action="store_true",
                   help="Skip NumPy / PyTorch-CPU baselines entirely")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    for d in (RESULTS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nDevice     : {gpu_name}")
    print(f"Vec sizes  : {args.vec_sizes}")
    print(f"Mat sizes  : {args.mat_sizes}")
    print(f"Warmup     : {args.warmup}   Iterations : {args.iterations}")

    skip_cpu = 0 if args.no_cpu else 2**26

    # ── Vector Addition ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    from bench_vector_add import benchmark as bench_vec
    vec_rows = bench_vec(
        sizes=args.vec_sizes,
        warmup=args.warmup,
        iterations=args.iterations,
        skip_cpu_above=skip_cpu,
    )

    # ── Matrix Multiplication ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    from bench_matmul import benchmark as bench_mat
    mat_rows = bench_mat(
        sizes=args.mat_sizes,
        warmup=args.warmup,
        iterations=args.iterations,
        skip_cpu_above=skip_cpu // 1024,  # mat sizes are much smaller
    )

    # ── Parallel Reduction ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    from bench_reduction import benchmark as bench_red
    red_rows = bench_red(
        sizes=args.vec_sizes,
        warmup=args.warmup,
        iterations=args.iterations,
        skip_cpu_above=skip_cpu,
    )

    # ── Save results ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "vec":       RESULTS_DIR / f"vec_add_{ts}.csv",
        "matmul":    RESULTS_DIR / f"matmul_{ts}.csv",
        "reduction": RESULTS_DIR / f"reduction_{ts}.csv",
    }
    pd.DataFrame(vec_rows).to_csv(paths["vec"],       index=False)
    pd.DataFrame(mat_rows).to_csv(paths["matmul"],    index=False)
    pd.DataFrame(red_rows).to_csv(paths["reduction"], index=False)

    print(f"\n{'='*60}")
    print(f"Results saved -> results/vec_add_{ts}.csv")
    print(f"              -> results/matmul_{ts}.csv")
    print(f"              -> results/reduction_{ts}.csv")

    if not args.no_plot:
        print("\n=== Generating plots ===")
        from plot_results import plot_all
        plot_all(vec_rows, red_rows, mat_rows, PLOTS_DIR, gpu_name=gpu_name)


if __name__ == "__main__":
    main()
