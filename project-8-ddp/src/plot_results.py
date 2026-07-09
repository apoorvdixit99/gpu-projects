"""Builds comparison plots across the four training modes from their CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODE_ORDER = ["base", "manual_ddp", "torch_ddp", "manual_ddp_2"]
MODE_LABELS = {
    "base": "Base (no DDP)",
    "manual_ddp": "Manual DDP",
    "torch_ddp": "torch.distributed DDP",
    "manual_ddp_2": "Manual DDP (from scratch)",
}
MODE_COLORS = {"base": "#4C72B0", "manual_ddp": "#DD8452", "torch_ddp": "#55A868", "manual_ddp_2": "#C44E52"}


def _load(results_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for mode in MODE_ORDER:
        path = results_dir / f"{mode}.csv"
        if path.exists():
            frames[mode] = pd.read_csv(path)
    return frames


def plot_all(results_dir: Path, plots_dir: Path, gpu_name: str = "") -> None:
    frames = _load(results_dir)
    if not frames:
        print("No result CSVs found — run the training scripts first.")
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    subtitle = f" — {gpu_name}" if gpu_name else ""

    # ── Training loss curves ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, df in frames.items():
        ax.plot(df["epoch"], df["train_loss"], marker="o", label=MODE_LABELS[mode], color=MODE_COLORS[mode])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title(f"BERT fine-tuning: training loss{subtitle}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "train_loss.png", dpi=150)
    plt.close(fig)

    # ── Validation accuracy curves ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, df in frames.items():
        ax.plot(df["epoch"], df["val_accuracy"], marker="o", label=MODE_LABELS[mode], color=MODE_COLORS[mode])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(f"BERT fine-tuning: validation accuracy{subtitle}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "val_accuracy.png", dpi=150)
    plt.close(fig)

    # ── Epoch time + throughput (bar charts, mean across epochs) ───────────
    modes = [m for m in MODE_ORDER if m in frames]
    epoch_times = [frames[m]["epoch_time_sec"].mean() for m in modes]
    throughputs = [frames[m]["throughput_samples_sec"].mean() for m in modes]
    colors = [MODE_COLORS[m] for m in modes]
    labels = [MODE_LABELS[m] for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(labels, epoch_times, color=colors)
    axes[0].set_ylabel("Mean epoch time (s)")
    axes[0].set_title("Epoch wall-clock time")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, throughputs, color=colors)
    axes[1].set_ylabel("Throughput (samples/s)")
    axes[1].set_title("Training throughput")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"BERT fine-tuning: speed comparison{subtitle}")
    fig.tight_layout()
    fig.savefig(plots_dir / "speed_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved -> {plots_dir}")


if __name__ == "__main__":
    import torch

    ROOT = Path(__file__).parent.parent
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    plot_all(ROOT / "results", ROOT / "results" / "plots", gpu_name=gpu)
