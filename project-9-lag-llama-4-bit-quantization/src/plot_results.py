"""Generate comparison charts from benchmark results.

Produces PNGs in results/plots/:
  latency-ms-mean.png   -- latency vs context length, FP32 vs NF4 vs int4-ao
  throughput.png        -- series/sec vs context length
  gpu-memory-mb.png     -- peak GPU memory vs context length
  accuracy-mase.png     -- MASE per dataset, FP32 vs NF4 vs int4-ao
  accuracy-crps.png     -- CRPS (approx) per dataset, FP32 vs NF4 vs int4-ao
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COLORS = {"lagllama_fp32": "#76b900", "lagllama_nf4": "#ff5722", "lagllama_int4-ao": "#2a78d6"}
_LABELS = {
    "lagllama_fp32": "FP32",
    "lagllama_nf4": "NF4 (bitsandbytes)",
    "lagllama_int4-ao": "int4 (torchao)",
}


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out.name}")


def _plot_perf_metric(df: pd.DataFrame, metric: str, ylabel: str, note: str, plots_dir: Path, name: str) -> None:
    backends = [b for b in _COLORS if b in df["backend"].values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend in backends:
        bdf = df[df["backend"] == backend].sort_values("context_length")
        if bdf.empty:
            continue
        ax.plot(
            bdf["context_length"], bdf[metric],
            label=_LABELS.get(backend, backend), color=_COLORS[backend],
            marker="o", linewidth=2, markersize=7,
        )
    ax.set_title(f"{ylabel}  --  {note}", fontweight="bold")
    ax.set_xlabel("Context length")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df["context_length"].unique()))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, plots_dir, name)


def _plot_accuracy_metric(acc_df: pd.DataFrame, metric: str, ylabel: str, plots_dir: Path, name: str) -> None:
    datasets = sorted(acc_df["dataset"].unique())
    backends = [b for b in _COLORS if b in acc_df["backend"].values]

    x = np.arange(len(datasets))
    width = 0.8 / max(len(backends), 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, backend in enumerate(backends):
        bdf = acc_df[acc_df["backend"] == backend].set_index("dataset").reindex(datasets)
        offset = (i - (len(backends) - 1) / 2) * width
        bars = ax.bar(
            x + offset, bdf[metric], width,
            label=_LABELS.get(backend, backend), color=_COLORS[backend],
            edgecolor="white", linewidth=0.8,
        )
        for bar, val in zip(bars, bdf[metric]):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{ylabel}  --  lower is better", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, plots_dir, name)


def _print_summary(perf_df: pd.DataFrame, acc_df: pd.DataFrame) -> None:
    if not perf_df.empty:
        pivot = perf_df.pivot_table(index="context_length", columns="backend", values="latency_ms_mean")
        print("\n=== Latency (ms) FP32 vs NF4 ===")
        print(pivot.to_string())

        mem_pivot = perf_df.pivot_table(index="context_length", columns="backend", values="gpu_memory_mb")
        print("\n=== Peak GPU memory (MB) FP32 vs NF4 ===")
        print(mem_pivot.to_string())

    if not acc_df.empty:
        print("\n=== Accuracy summary ===")
        print(acc_df.to_string(index=False))


def plot(perf_df: pd.DataFrame, acc_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not perf_df.empty:
        _plot_perf_metric(perf_df, "latency_ms_mean", "Latency (ms)", "lower is better", plots_dir, "latency-ms-mean.png")
        _plot_perf_metric(perf_df, "throughput_series_per_sec", "Throughput (series / s)", "higher is better", plots_dir, "throughput.png")
        _plot_perf_metric(perf_df, "gpu_memory_mb", "Peak GPU Memory (MB)", "lower is better", plots_dir, "gpu-memory-mb.png")

    if not acc_df.empty:
        _plot_accuracy_metric(acc_df, "MASE", "MASE", plots_dir, "accuracy-mase.png")
        _plot_accuracy_metric(acc_df, "CRPS_approx", "CRPS (mean weighted quantile loss)", plots_dir, "accuracy-crps.png")

    _print_summary(perf_df, acc_df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <latency.csv> [accuracy.csv]")
        sys.exit(1)

    perf_df = pd.read_csv(sys.argv[1])
    acc_df = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else pd.DataFrame()
    plot(perf_df, acc_df, Path("results/plots"))
