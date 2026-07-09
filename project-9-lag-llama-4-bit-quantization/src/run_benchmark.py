"""Main entry point for the Lag-Llama FP32 vs NF4 vs int4-ao benchmark.

Usage examples
--------------
# Full run -- latency/throughput/memory + zero-shot accuracy, all three precisions
python src/run_benchmark.py

# Just the two int4 variants
python src/run_benchmark.py --precisions nf4 int4-ao

# Latency only, skip accuracy (faster, no dataset downloads beyond the test split)
python src/run_benchmark.py --no-accuracy

# Custom context length sweep / datasets
python src/run_benchmark.py --context-lengths 32 64 --datasets airpassengers m4_hourly
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from itertools import islice
from pathlib import Path

import pandas as pd
import torch

# gluonts<=0.14.4 builds forecast indices with pd.Period/BDay internally, which
# newer pandas flags as deprecated on every single series evaluated -- thousands
# of identical warnings per run that bury the actual progress/results. Not
# actionable from this project (it's gluonts' internal indexing, not our code).
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Period.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*PeriodDtype.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*_check_is_size.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*non-tuple sequence for multidimensional indexing.*")

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lag-Llama NF4 Quantization Benchmark")
    p.add_argument("--precisions", nargs="+", default=["fp32", "nf4", "int4-ao"], choices=["fp32", "nf4", "int4-ao"])
    p.add_argument("--context-lengths", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--prediction-length", type=int, default=24, help="Used for the latency benchmark only")
    p.add_argument("--num-latency-series", type=int, default=32, help="Series drawn from airpassengers-like synthetic subset for latency timing")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--datasets", nargs="+", default=["airpassengers", "exchange_rate", "m4_hourly"])
    p.add_argument("--accuracy-context-length", type=int, default=32)
    p.add_argument("--num-samples", type=int, default=100, help="Forecast trajectories sampled per series")
    p.add_argument("--max-series-per-dataset", type=int, default=50, help="Cap series evaluated per dataset (speed)")
    p.add_argument("--no-accuracy", action="store_true", help="Skip zero-shot accuracy evaluation")
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice           : {torch.cuda.get_device_name(0)}")
    print(f"Precisions       : {args.precisions}")
    print(f"Context lengths  : {args.context_lengths}")
    print(f"Datasets         : {args.datasets}")

    # -- Latency / throughput / memory ---------------------------------------
    from bench_latency import benchmark as bench_latency
    from gluonts.dataset.repository.datasets import get_dataset

    print(f"\n{'='*60}\n=== Latency / Throughput / Memory ===")
    latency_dataset = get_dataset("airpassengers")
    test_series = list(islice(latency_dataset.test, args.num_latency_series)) or list(latency_dataset.test)

    all_perf: list[dict] = []
    for precision in args.precisions:
        rows = bench_latency(
            precision=precision,
            test_series=test_series,
            context_lengths=args.context_lengths,
            prediction_length=args.prediction_length,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        all_perf.extend(rows)

    # -- Zero-shot accuracy ---------------------------------------------------
    all_acc: list[dict] = []
    if not args.no_accuracy:
        from evaluate_accuracy import evaluate

        print(f"\n{'='*60}\n=== Zero-shot Accuracy (CRPS / MASE / sMAPE) ===")
        all_acc = evaluate(
            dataset_names=args.datasets,
            precisions=args.precisions,
            context_length=args.accuracy_context_length,
            num_samples=args.num_samples,
            max_series=args.max_series_per_dataset,
        )

    # -- Save results ----------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    perf_df = pd.DataFrame(all_perf)
    perf_csv = RESULTS_DIR / f"latency_{ts}.csv"
    perf_df.to_csv(perf_csv, index=False)
    print(f"\n{'='*60}")
    print(perf_df.to_string(index=False))
    print(f"\nLatency results saved -> {perf_csv}")

    acc_csv = None
    if all_acc:
        acc_df = pd.DataFrame(all_acc)
        acc_csv = RESULTS_DIR / f"accuracy_{ts}.csv"
        acc_df.to_csv(acc_csv, index=False)
        print(f"\n{acc_df.to_string(index=False)}")
        print(f"Accuracy results saved -> {acc_csv}")

    # -- Plots -------------------------------------------------------------
    if not args.no_plot:
        print("\n=== Generating plots ===")
        from plot_results import plot

        acc_df = pd.read_csv(acc_csv) if acc_csv else pd.DataFrame()
        plot(perf_df, acc_df, PLOTS_DIR)


if __name__ == "__main__":
    main()
