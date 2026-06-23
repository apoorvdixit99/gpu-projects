"""Main entry point for the GPT-2 quantization benchmark.

Usage examples
--------------
# Full run -- all four precisions, benchmark + perplexity
python src/run_benchmark.py

# Subset of precisions
python src/run_benchmark.py --precisions fp32 fp16

# Custom sweep
python src/run_benchmark.py --batch-sizes 1 8 32 --seq-lens 128 256 --iterations 50

# Skip perplexity measurement (faster)
python src/run_benchmark.py --no-perplexity

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
from transformers import AutoTokenizer

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

sys.path.insert(0, str(Path(__file__).parent))

_PRECISION_MODULES = {
    "fp32": "bench_fp32",
    "fp16": "bench_fp16",
    "int8": "bench_int8",
    "int4": "bench_int4",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 Quantization Benchmark")
    p.add_argument(
        "--precisions", nargs="+",
        default=["fp32", "fp16", "int8", "int4"],
        choices=list(_PRECISION_MODULES),
        help="Precision levels to benchmark (default: all four)",
    )
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16])
    p.add_argument("--seq-lens",    nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--warmup",      type=int,  default=10,  help="Warmup iterations")
    p.add_argument("--iterations",  type=int,  default=100, help="Timed iterations")
    p.add_argument("--no-perplexity", action="store_true", help="Skip perplexity measurement")
    p.add_argument("--no-plot",       action="store_true", help="Skip plot generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice      : {torch.cuda.get_device_name(0)}")
    print(f"Precisions  : {args.precisions}")
    print(f"Batch sizes : {args.batch_sizes}")
    print(f"Seq lengths : {args.seq_lens}")
    print(f"Warmup      : {args.warmup}   Iterations : {args.iterations}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2") if not args.no_perplexity else None

    all_perf: list[dict] = []
    all_ppl:  list[dict] = []

    for precision in args.precisions:
        module_name = _PRECISION_MODULES[precision]
        mod = __import__(module_name)

        # -- Latency / throughput / memory benchmark --------------------------
        print(f"\n{'='*60}")
        print(f"=== {precision.upper()} Benchmark ===")
        rows = mod.benchmark(args.batch_sizes, args.seq_lens, args.warmup, args.iterations)
        all_perf.extend(rows)

        # -- Perplexity -------------------------------------------------------
        if not args.no_perplexity:
            from measure_perplexity import measure_perplexity
            print(f"\n--- {precision.upper()} Perplexity ---")
            model = mod.load_model(torch.device("cuda"))
            ppl   = measure_perplexity(model, tokenizer)
            backend = rows[0]["backend"] if rows else precision
            print(f"  Perplexity [{backend}]: {ppl:.4f}")
            all_ppl.append({"backend": backend, "perplexity": round(ppl, 4)})
            del model
            torch.cuda.empty_cache()

    # -- Save results ---------------------------------------------------------
    if not all_perf:
        print("No results collected.")
        return

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    perf_df  = pd.DataFrame(all_perf)
    perf_csv = RESULTS_DIR / f"benchmark_{ts}.csv"
    perf_df.to_csv(perf_csv, index=False)
    print(f"\n{'='*60}")
    print(perf_df.to_string(index=False))
    print(f"\nResults saved -> {perf_csv}")

    if all_ppl:
        ppl_csv = RESULTS_DIR / f"perplexity_{ts}.csv"
        pd.DataFrame(all_ppl).to_csv(ppl_csv, index=False)
        print(f"Perplexity  -> {ppl_csv}")

    # -- Plots ----------------------------------------------------------------
    if not args.no_plot:
        print("\n=== Generating plots ===")
        from plot_results import plot
        plot(perf_df, all_ppl, PLOTS_DIR)


if __name__ == "__main__":
    main()
