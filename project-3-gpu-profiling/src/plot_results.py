"""Generate profiling charts from collected results.

Produces four PNGs in results/plots/:
  kernel_breakdown.png   — top-5 kernels per batch size (stacked bars)
  bottlenecks.png        — GPU kernel time vs CPU overhead + overlap % (two panels)
  memory_usage.png       — peak allocated vs reserved vs batch size (line chart)
  throughput_scaling.png — tokens/sec vs batch size (line chart)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

_GREEN  = "#76b900"   # NVIDIA green
_ORANGE = "#e87722"
_BLUE   = "#2196f3"


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out.name}")


def _plot_kernel_breakdown(kernel_rows: list[dict], plots_dir: Path) -> None:
    df = pd.DataFrame(kernel_rows)
    batch_sizes = sorted(df["batch_size"].unique())

    all_kernels: list[str] = []
    top_per_bs: dict[int, pd.DataFrame] = {}
    for bs in batch_sizes:
        top = (
            df[df["batch_size"] == bs]
            .nlargest(5, "cuda_time_ms")[["kernel", "cuda_time_ms"]]
            .copy()
        )
        top["kernel"] = top["kernel"].str[:40]
        top_per_bs[bs] = top
        for k in top["kernel"]:
            if k not in all_kernels:
                all_kernels.append(k)
    all_kernels = all_kernels[:8]

    fig, ax = plt.subplots(figsize=(max(10, len(batch_sizes) * 1.6), 5))
    cmap    = plt.get_cmap("tab10")
    bottoms = [0.0] * len(batch_sizes)

    for i, kernel in enumerate(all_kernels):
        values = []
        for bs in batch_sizes:
            sub   = top_per_bs[bs]
            match = sub[sub["kernel"] == kernel]["cuda_time_ms"]
            values.append(float(match.values[0]) if len(match) else 0.0)
        ax.bar(range(len(batch_sizes)), values, bottom=bottoms,
               label=kernel, color=cmap(i), width=0.6)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([f"bs={bs}" for bs in batch_sizes])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("CUDA time (ms)")
    ax.set_title("Top kernel CUDA time by batch size")
    ax.legend(loc="upper left", fontsize=7)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    _save(fig, plots_dir, "kernel_breakdown.png")


def _plot_bottlenecks(bottleneck_rows: list[dict], plots_dir: Path) -> None:
    df = pd.DataFrame(bottleneck_rows).sort_values("batch_size")
    x  = range(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("CPU / GPU Bottleneck Analysis", fontsize=13, fontweight="bold")

    ax1.bar(x, df["gpu_time_ms"],     0.5, label="GPU kernel time", color=_GREEN)
    ax1.bar(x, df["cpu_overhead_ms"], 0.5, bottom=df["gpu_time_ms"],
            label="CPU overhead", color=_ORANGE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"bs={bs}" for bs in df["batch_size"]])
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("GPU kernel time vs CPU overhead")
    ax1.legend()
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax2.plot(df["batch_size"], df["overlap_pct"], marker="o", color=_GREEN, linewidth=2)
    ax2.axhline(80, linestyle="--", color="grey", linewidth=1, label="80% threshold")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Overlap (%)")
    ax2.set_title("CPU/GPU execution overlap")
    ax2.set_ylim(0, 105)
    ax2.legend()

    fig.tight_layout()
    _save(fig, plots_dir, "bottlenecks.png")


def _plot_memory(memory_rows: list[dict], plots_dir: Path) -> None:
    df = pd.DataFrame(memory_rows).sort_values("batch_size")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["batch_size"], df["peak_allocated_mb"], marker="o",
            label="Peak allocated", color=_GREEN, linewidth=2)
    ax.plot(df["batch_size"], df["peak_reserved_mb"], marker="s",
            label="Peak reserved", color=_BLUE, linewidth=2, linestyle="--")
    ax.fill_between(df["batch_size"], df["peak_allocated_mb"], df["peak_reserved_mb"],
                    alpha=0.12, color=_BLUE, label="Allocator cache gap")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("GPU memory (MB)")
    ax.set_title("GPU memory usage vs batch size")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    fig.tight_layout()
    _save(fig, plots_dir, "memory_usage.png")


def _plot_throughput(bottleneck_rows: list[dict], plots_dir: Path) -> None:
    df = pd.DataFrame(bottleneck_rows).sort_values("batch_size")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["batch_size"], df["throughput_tok_per_sec"] / 1e3,
            marker="o", color=_GREEN, linewidth=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (k tokens / s)")
    ax.set_title("Throughput scaling with batch size")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    _save(fig, plots_dir, "throughput_scaling.png")


def plot_all(
    kernel_rows:     list[dict],
    bottleneck_rows: list[dict],
    memory_rows:     list[dict],
    plots_dir:       Path,
) -> None:
    _plot_kernel_breakdown(kernel_rows, plots_dir)
    _plot_bottlenecks(bottleneck_rows, plots_dir)
    _plot_memory(memory_rows, plots_dir)
    _plot_throughput(bottleneck_rows, plots_dir)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_results.py kernels.csv bottlenecks.csv memory.csv")
        sys.exit(1)
    plot_all(
        pd.read_csv(sys.argv[1]).to_dict("records"),
        pd.read_csv(sys.argv[2]).to_dict("records"),
        pd.read_csv(sys.argv[3]).to_dict("records"),
        Path("results/plots"),
    )
