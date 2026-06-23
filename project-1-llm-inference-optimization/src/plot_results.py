"""Generate comparison charts from benchmark CSV data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_BACKEND_COLORS = {
    "pytorch_fp16":  "#76b900",   # NVIDIA green
    "pytorch_fp32":  "#a8d86e",
    "onnxruntime":   "#2196f3",   # blue
    "tensorrt_fp16": "#ff5722",   # orange-red
}
_MARKERS = {"pytorch_fp16": "o", "pytorch_fp32": "s", "onnxruntime": "^", "tensorrt_fp16": "D"}


def plot(df: pd.DataFrame, out_dir: str | Path = "results/plots") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = sorted(df["seq_len"].unique())
    backends = df["backend"].unique()

    metrics = [
        ("latency_ms_mean",        "Latency (ms)",            "lower is better"),
        ("throughput_tok_per_sec", "Throughput (tokens / s)", "higher is better"),
        ("gpu_memory_mb",          "Peak GPU Memory (MB)",    "lower is better"),
    ]

    for metric, ylabel, note in metrics:
        fig, axes = plt.subplots(1, len(seq_lens), figsize=(5 * len(seq_lens), 4), sharey=False)
        if len(seq_lens) == 1:
            axes = [axes]

        fig.suptitle(f"{ylabel}  —  {note}", fontsize=13, fontweight="bold")

        for ax, seq_len in zip(axes, seq_lens):
            sub = df[df["seq_len"] == seq_len]
            for backend in backends:
                bdf = sub[sub["backend"] == backend].sort_values("batch_size")
                if bdf.empty:
                    continue
                color = _BACKEND_COLORS.get(backend, "gray")
                marker = _MARKERS.get(backend, "x")
                ax.plot(
                    bdf["batch_size"],
                    bdf[metric],
                    label=backend,
                    color=color,
                    marker=marker,
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
        save_path = out_dir / f"{slug}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {save_path}")

    _speedup_table(df)


def _speedup_table(df: pd.DataFrame) -> None:
    """Print a quick speedup table comparing backends against pytorch_fp16."""
    baseline = "pytorch_fp16"
    if baseline not in df["backend"].values:
        return

    print("\n=== Speedup vs pytorch_fp16 (latency) ===")
    base_df = df[df["backend"] == baseline][["batch_size", "seq_len", "latency_ms_mean"]]
    base_df = base_df.rename(columns={"latency_ms_mean": "base_ms"})

    merged = df.merge(base_df, on=["batch_size", "seq_len"])
    merged["speedup"] = (merged["base_ms"] / merged["latency_ms_mean"]).round(2)

    pivot = merged.pivot_table(
        index=["batch_size", "seq_len"],
        columns="backend",
        values="speedup",
    )
    print(pivot.to_string())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <path/to/benchmark.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    out = Path(sys.argv[1]).parent / "plots"
    plot(df, out)
