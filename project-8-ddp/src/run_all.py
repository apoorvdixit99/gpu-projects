"""Main entry point: runs all four training modes back-to-back, then plots
a side-by-side comparison.

Usage examples
--------------
# Full run — base, manual DDP, torch.distributed DDP, manual DDP from scratch, then plots
python src/run_all.py

# Smaller/faster smoke test
python src/run_all.py --train-subset 200 --epochs 1

# Skip a mode or the plotting step
python src/run_all.py --skip-manual-ddp
python src/run_all.py --no-plot
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run base / manual-DDP / torch-DDP / manual-DDP-from-scratch BERT fine-tuning and compare")
    p.add_argument("--epochs", type=int, default=2, metavar="N")
    p.add_argument("--train-subset", type=int, default=1000, metavar="N", help="Total training examples (default: 1000)")
    p.add_argument("--world-size", type=int, default=2, metavar="N", help="DDP processes for all DDP modes (default: 2)")
    p.add_argument("--per-rank-batch-size", type=int, default=16, metavar="N",
                   help="Per-rank batch size for DDP modes; base uses this * world-size (default: 16)")
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--skip-manual-ddp", action="store_true")
    p.add_argument("--skip-torch-ddp", action="store_true")
    p.add_argument("--skip-manual-ddp-2", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    global_batch = args.per_rank_batch_size * args.world_size

    if not args.skip_base:
        print("\n" + "=" * 60 + "\nCase 1: Base (no DDP)\n" + "=" * 60)
        run([sys.executable, str(SRC / "train_base.py"),
             "--epochs", str(args.epochs),
             "--batch-size", str(global_batch),
             "--train-subset", str(args.train_subset)])

    if not args.skip_manual_ddp:
        print("\n" + "=" * 60 + "\nCase 2: Manual DDP\n" + "=" * 60)
        run([sys.executable, str(SRC / "train_manual_ddp.py"),
             "--world-size", str(args.world_size),
             "--epochs", str(args.epochs),
             "--batch-size", str(args.per_rank_batch_size),
             "--train-subset", str(args.train_subset)])

    if not args.skip_torch_ddp:
        print("\n" + "=" * 60 + "\nCase 3: torch.distributed DDP\n" + "=" * 60)
        run([sys.executable, str(SRC / "launch_torch_ddp.py"),
             "--world-size", str(args.world_size),
             "--epochs", str(args.epochs),
             "--batch-size", str(args.per_rank_batch_size),
             "--train-subset", str(args.train_subset)])

    if not args.skip_manual_ddp_2:
        print("\n" + "=" * 60 + "\nCase 4: Manual DDP, from scratch (no torch.distributed)\n" + "=" * 60)
        run([sys.executable, str(SRC / "train_manual_ddp_2.py"),
             "--world-size", str(args.world_size),
             "--epochs", str(args.epochs),
             "--batch-size", str(args.per_rank_batch_size),
             "--train-subset", str(args.train_subset)])

    if not args.no_plot:
        print("\n" + "=" * 60 + "\nPlotting comparison\n" + "=" * 60)
        import torch

        sys.path.insert(0, str(SRC))
        from plot_results import plot_all

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        plot_all(RESULTS_DIR, PLOTS_DIR, gpu_name=gpu)


if __name__ == "__main__":
    main()
