"""Generate comparison charts for the Deep Learning Workload Analysis.

Produces five PNGs in results/plots/:
  architecture_overview.png  — FLOPs (GFLOPs) and parameter count bar charts
  latency_sweep.png          — latency vs batch size, all models
  throughput_sweep.png       — throughput vs batch size (NLP | Vision subplots)
  memory_sweep.png           — peak GPU memory vs batch size, all models
  mfu_sweep.png              — MFU% vs batch size, all models
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
_PURPLE = "#9c27b0"

_PALETTE = [_GREEN, _ORANGE, _BLUE, _PURPLE]


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out.name}")


def _model_colors(df: pd.DataFrame) -> dict[str, str]:
    """Assign consistent palette colors keyed by model name."""
    names = list(df["model"].unique())
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(names)}


# ── Individual charts ────────────────────────────────────────────────────────

def _plot_architecture(flops_rows: list[dict], plots_dir: Path) -> None:
    df = pd.DataFrame(flops_rows).sort_values("flops_g", ascending=False)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(df))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Architecture Comparison", fontsize=13, fontweight="bold")

    bars1 = ax1.bar(df["label"], df["flops_g"], color=colors, width=0.5)
    ax1.set_ylabel("GFLOPs (per sample)")
    ax1.set_title("Theoretical FLOPs (seq=128 / 224×224)")
    ax1.bar_label(bars1, fmt="%.1f G", padding=3, fontsize=9)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right", fontsize=9)

    bars2 = ax2.bar(df["label"], df["params_m"], color=colors, width=0.5)
    ax2.set_ylabel("Parameters (M)")
    ax2.set_title("Parameter Count")
    ax2.bar_label(bars2, fmt="%.0f M", padding=3, fontsize=9)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right", fontsize=9)

    fig.tight_layout()
    _save(fig, plots_dir, "architecture_overview.png")


def _plot_latency(latency_rows: list[dict], plots_dir: Path) -> None:
    df     = pd.DataFrame(latency_rows)
    colors = _model_colors(df)
    models = df[["model", "label"]].drop_duplicates().itertuples(index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for row in models:
        sub = df[df["model"] == row.model].sort_values("batch_size")
        ax.plot(sub["batch_size"], sub["latency_ms_mean"],
                marker="o", label=row.label, color=colors[row.model], linewidth=2)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference latency vs batch size (FP16)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    _save(fig, plots_dir, "latency_sweep.png")


def _plot_throughput(latency_rows: list[dict], plots_dir: Path) -> None:
    df      = pd.DataFrame(latency_rows)
    colors  = _model_colors(df)
    nlp_df  = df[df["modality"] == "nlp"]
    vis_df  = df[df["modality"] == "vision"]

    n_panels = (1 if len(nlp_df) > 0 else 0) + (1 if len(vis_df) > 0 else 0)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    fig.suptitle("Throughput vs batch size (FP16)", fontsize=13, fontweight="bold")
    ax_idx = 0

    if len(nlp_df) > 0:
        ax = axes[0][ax_idx]; ax_idx += 1
        for name in nlp_df["model"].unique():
            sub = nlp_df[nlp_df["model"] == name].sort_values("batch_size")
            ax.plot(sub["batch_size"], sub["throughput"] / 1e3,
                    marker="o", label=sub["label"].iloc[0],
                    color=colors[name], linewidth=2)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput (k tok/s)")
        ax.set_title("NLP models — tokens / second")
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    if len(vis_df) > 0:
        ax = axes[0][ax_idx]
        for name in vis_df["model"].unique():
            sub = vis_df[vis_df["model"] == name].sort_values("batch_size")
            ax.plot(sub["batch_size"], sub["throughput"],
                    marker="s", label=sub["label"].iloc[0],
                    color=colors[name], linewidth=2)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput (img/s)")
        ax.set_title("Vision models — images / second")
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    fig.tight_layout()
    _save(fig, plots_dir, "throughput_sweep.png")


def _plot_memory(memory_rows: list[dict], plots_dir: Path) -> None:
    df     = pd.DataFrame(memory_rows)
    colors = _model_colors(df)
    models = df[["model", "label"]].drop_duplicates().itertuples(index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for row in models:
        sub = df[df["model"] == row.model].sort_values("batch_size")
        ax.plot(sub["batch_size"], sub["peak_allocated_mb"],
                marker="o", label=row.label, color=colors[row.model], linewidth=2)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.set_title("GPU memory footprint vs batch size (FP16)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    fig.tight_layout()
    _save(fig, plots_dir, "memory_sweep.png")


def _plot_mfu(latency_rows: list[dict], plots_dir: Path) -> None:
    df = pd.DataFrame(latency_rows)
    if "mfu_pct" not in df.columns or df["mfu_pct"].max() == 0:
        return

    colors = _model_colors(df)
    models = df[["model", "label"]].drop_duplicates().itertuples(index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for row in models:
        sub = df[df["model"] == row.model].sort_values("batch_size")
        ax.plot(sub["batch_size"], sub["mfu_pct"],
                marker="o", label=row.label, color=colors[row.model], linewidth=2)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("MFU (%)")
    ax.set_title("Model FLOP Utilization vs batch size (FP16)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    _save(fig, plots_dir, "mfu_sweep.png")


# ── Public entry point ───────────────────────────────────────────────────────

def plot_all(
    flops_rows:   list[dict],
    latency_rows: list[dict],
    memory_rows:  list[dict],
    plots_dir:    Path,
) -> None:
    _plot_architecture(flops_rows, plots_dir)
    _plot_latency(latency_rows, plots_dir)
    _plot_throughput(latency_rows, plots_dir)
    _plot_memory(memory_rows, plots_dir)
    _plot_mfu(latency_rows, plots_dir)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_results.py flops.csv latency.csv memory.csv")
        sys.exit(1)
    plot_all(
        pd.read_csv(sys.argv[1]).to_dict("records"),
        pd.read_csv(sys.argv[2]).to_dict("records"),
        pd.read_csv(sys.argv[3]).to_dict("records"),
        Path("results/plots"),
    )
