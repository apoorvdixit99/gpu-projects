"""Generate comparison charts from benchmark results.

Produces five PNGs in results/plots/:
  latency-ms-mean.png         -- latency vs batch size, one line per precision
  throughput-tok-per-sec.png  -- throughput vs batch size
  gpu-memory-mb.png           -- peak GPU memory vs batch size
  perplexity.png              -- perplexity bar chart, one bar per precision
  speedup-vs-fp32.png         -- latency speedup relative to FP32 baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

_COLORS = {
    "pytorch_fp32":  "#a8d86e",   # light green
    "pytorch_fp16":  "#76b900",   # NVIDIA green
    "quanto_int8":   "#2196f3",   # blue
    "quanto_int4":   "#ff5722",   # orange-red
    "torchao_nvfp4": "#9c27b0",   # purple
    "torchao_mxfp8": "#00bcd4",   # cyan
}
_MARKERS = {
    "pytorch_fp32":  "s",
    "pytorch_fp16":  "o",
    "quanto_int8":   "^",
    "quanto_int4":   "D",
    "torchao_nvfp4": "P",
    "torchao_mxfp8": "X",
}
_LABELS = {
    "pytorch_fp32":  "FP32",
    "pytorch_fp16":  "FP16",
    "quanto_int8":   "INT8 (quanto)",
    "quanto_int4":   "INT4 (quanto)",
    "torchao_nvfp4": "NVFP4 (torchao, sw-emu)",
    "torchao_mxfp8": "MXFP8 (torchao, sw-emu)",
}


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out.name}")


def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    note: str,
    plots_dir: Path,
) -> None:
    seq_lens = sorted(df["seq_len"].unique())
    backends = [b for b in _COLORS if b in df["backend"].values]

    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5 * len(seq_lens), 4), sharey=False)
    if len(seq_lens) == 1:
        axes = [axes]

    fig.suptitle(f"{ylabel}  --  {note}", fontsize=13, fontweight="bold")

    for ax, seq_len in zip(axes, seq_lens):
        sub = df[df["seq_len"] == seq_len]
        for backend in backends:
            bdf = sub[sub["backend"] == backend].sort_values("batch_size")
            if bdf.empty:
                continue
            ax.plot(
                bdf["batch_size"],
                bdf[metric],
                label=_LABELS.get(backend, backend),
                color=_COLORS[backend],
                marker=_MARKERS[backend],
                linewidth=2,
                markersize=7,
            )
        ax.set_title(f"seq_len = {seq_len}", fontsize=11)
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["batch_size"].unique()))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    plt.tight_layout()
    slug = metric.replace("_", "-")
    _save(fig, plots_dir, f"{slug}.png")


def _plot_perplexity(ppl_rows: list[dict], plots_dir: Path) -> None:
    df = pd.DataFrame(ppl_rows)
    backends = [b for b in _COLORS if b in df["backend"].values]
    labels   = [_LABELS.get(b, b) for b in backends]
    values   = [df.loc[df["backend"] == b, "perplexity"].values[0] for b in backends]
    colors   = [_COLORS[b] for b in backends]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Accuracy degradation by precision  --  lower is better", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, plots_dir, "perplexity.png")


def _plot_speedup(df: pd.DataFrame, plots_dir: Path) -> None:
    baseline = "pytorch_fp32"
    if baseline not in df["backend"].values:
        return

    base = df[df["backend"] == baseline][["batch_size", "seq_len", "latency_ms_mean"]]
    base = base.rename(columns={"latency_ms_mean": "base_ms"})
    merged = df.merge(base, on=["batch_size", "seq_len"])
    merged["speedup"] = merged["base_ms"] / merged["latency_ms_mean"]

    seq_lens = sorted(merged["seq_len"].unique())
    backends = [b for b in _COLORS if b in merged["backend"].values and b != baseline]

    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5 * len(seq_lens), 4), sharey=True)
    if len(seq_lens) == 1:
        axes = [axes]

    fig.suptitle("Latency speedup vs FP32  --  higher is better", fontsize=13, fontweight="bold")

    for ax, seq_len in zip(axes, seq_lens):
        sub = merged[merged["seq_len"] == seq_len]
        for backend in backends:
            bdf = sub[sub["backend"] == backend].sort_values("batch_size")
            if bdf.empty:
                continue
            ax.plot(
                bdf["batch_size"],
                bdf["speedup"],
                label=_LABELS.get(backend, backend),
                color=_COLORS[backend],
                marker=_MARKERS[backend],
                linewidth=2,
                markersize=7,
            )
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="FP32 baseline")
        ax.set_title(f"seq_len = {seq_len}", fontsize=11)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Speedup (x)")
        ax.set_xticks(sorted(merged["batch_size"].unique()))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, plots_dir, "speedup-vs-fp32.png")


def _print_summary(df: pd.DataFrame, ppl_rows: list[dict]) -> None:
    baseline = "pytorch_fp32"
    base = df[df["backend"] == baseline][["batch_size", "seq_len", "latency_ms_mean"]]
    base = base.rename(columns={"latency_ms_mean": "base_ms"})
    merged = df.merge(base, on=["batch_size", "seq_len"])
    merged["speedup"] = (merged["base_ms"] / merged["latency_ms_mean"]).round(2)

    pivot = merged.pivot_table(
        index=["batch_size", "seq_len"],
        columns="backend",
        values="speedup",
    )
    print("\n=== Latency speedup vs pytorch_fp32 ===")
    print(pivot.to_string())

    if ppl_rows:
        print("\n=== Perplexity summary ===")
        ppl_df = pd.DataFrame(ppl_rows)
        base_ppl = ppl_df.loc[ppl_df["backend"] == baseline, "perplexity"].values
        if base_ppl.size:
            ppl_df["ppl_delta"] = ((ppl_df["perplexity"] - base_ppl[0]) / base_ppl[0] * 100).round(2)
        print(ppl_df.to_string(index=False))


def plot(
    perf_df:   pd.DataFrame,
    ppl_rows:  list[dict],
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric(perf_df, "latency_ms_mean",        "Latency (ms)",           "lower is better", plots_dir)
    _plot_metric(perf_df, "throughput_tok_per_sec",  "Throughput (tokens / s)", "higher is better", plots_dir)
    _plot_metric(perf_df, "gpu_memory_mb",           "Peak GPU Memory (MB)",   "lower is better", plots_dir)
    _plot_speedup(perf_df, plots_dir)

    if ppl_rows:
        _plot_perplexity(ppl_rows, plots_dir)

    _print_summary(perf_df, ppl_rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <benchmark.csv> [perplexity.csv]")
        sys.exit(1)

    perf_df  = pd.read_csv(sys.argv[1])
    ppl_rows = pd.read_csv(sys.argv[2]).to_dict("records") if len(sys.argv) > 2 else []
    plot(perf_df, ppl_rows, Path("results/plots"))
