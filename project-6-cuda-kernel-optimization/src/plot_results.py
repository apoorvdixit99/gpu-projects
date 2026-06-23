"""Generate benchmark charts from collected results.

Produces four PNGs in results/plots/:
  bandwidth_utilization.png  — achieved GB/s for vec add and reduction (naive/opt vs size)
  matmul_performance.png     — GFLOPS for naive / tiled / cuBLAS vs matrix size
  speedup_comparison.png     — speedup of each CUDA variant over NumPy at each size
  latency_scaling.png        — latency (ms) vs problem size for all reduction variants
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
_GREY   = "#9e9e9e"
_RED    = "#e53935"

# Theoretical peak memory bandwidth by GPU name (GB/s).
# Used as a reference line in the bandwidth chart.
_PEAK_BW_GBS: dict[str, float] = {
    "NVIDIA GeForce RTX 4080 Laptop GPU": 432.0,
    "NVIDIA GeForce RTX 4090":            1008.0,
    "NVIDIA GeForce RTX 3090":            936.0,
    "NVIDIA GeForce RTX 3080":            760.0,
}


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out.name}")


# ---------------------------------------------------------------------------
def _plot_bandwidth(vec_rows: list[dict], red_rows: list[dict],
                    plots_dir: Path, gpu_name: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Achieved Memory Bandwidth", fontsize=13, fontweight="bold")
    peak_bw = _PEAK_BW_GBS.get(gpu_name)

    def _draw(ax, rows, title, cuda_variants):
        df = pd.DataFrame(rows)
        sizes = sorted(df["N"].unique())
        x = range(len(sizes))

        color_map = {
            "numpy":         _GREY,
            "torch_cpu":     _BLUE,
            cuda_variants[0]: _ORANGE,
            cuda_variants[1]: _GREEN,
            "torch_add_gpu": _RED,
        }
        label_map = {
            "numpy":         "NumPy (CPU)",
            "torch_cpu":     "PyTorch CPU",
            cuda_variants[0]: "CUDA naive",
            cuda_variants[1]: "CUDA opt",
            "torch_add_gpu": "torch.add (GPU)",
            "reduce_naive":      "CUDA naive",
            "reduce_sequential": "CUDA sequential",
            "reduce_shuffle":    "CUDA shuffle",
        }
        for variant, color in color_map.items():
            sub = df[df["variant"] == variant]
            if sub.empty:
                continue
            bw = [sub[sub["N"] == s]["bandwidth_gb_s"].values[0]
                  if len(sub[sub["N"] == s]) else 0 for s in sizes]
            ax.plot(range(len(sizes)), bw, marker="o", label=label_map.get(variant, variant),
                    color=color, linewidth=2)

        if peak_bw is not None:
            ax.axhline(peak_bw, linestyle="--", color="black", linewidth=1,
                       label=f"Theoretical peak ({peak_bw:.0f} GB/s)")

        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{s//1_048_576}M" for s in sizes], fontsize=8)
        ax.set_xlabel("Array size (elements)")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    _draw(ax1, vec_rows, "Vector Addition", ["cuda_naive", "cuda_opt"])
    # For reduction, only CUDA variants have bandwidth data
    red_df = pd.DataFrame(red_rows)
    for variant, color, label in [
        ("reduce_naive",      _ORANGE, "CUDA naive"),
        ("reduce_sequential", _GREEN,  "CUDA sequential"),
        ("reduce_shuffle",    _BLUE,   "CUDA shuffle"),
    ]:
        sub = red_df[red_df["variant"] == variant]
        if sub.empty:
            continue
        x2 = list(range(len(sub)))
        ax2.plot(x2, sub["bandwidth_gb_s"].values, marker="o",
                 label=label, color=color, linewidth=2)
    if peak_bw is not None:
        ax2.axhline(peak_bw, linestyle="--", color="black", linewidth=1,
                    label=f"Theoretical peak ({peak_bw:.0f} GB/s)")

    sizes_r = sorted(red_df["N"].unique())
    ax2.set_xticks(range(len(sizes_r)))
    ax2.set_xticklabels([f"{s//1_048_576}M" for s in sizes_r], fontsize=8)
    ax2.set_xlabel("Array size (elements)")
    ax2.set_ylabel("Bandwidth (GB/s)")
    ax2.set_title("Parallel Reduction")
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    fig.tight_layout()
    _save(fig, plots_dir, "bandwidth_utilization.png")


# ---------------------------------------------------------------------------
def _plot_matmul(mat_rows: list[dict], plots_dir: Path) -> None:
    df = pd.DataFrame(mat_rows)
    cuda_variants = {
        "cuda_naive":  (_ORANGE, "CUDA naive"),
        "cuda_tiled":  (_GREEN,  "CUDA tiled (shared mem)"),
        "cublas":      (_BLUE,   "cuBLAS (torch.mm)"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Matrix Multiplication Performance", fontsize=13, fontweight="bold")

    for variant, (color, label) in cuda_variants.items():
        sub = df[df["variant"] == variant].sort_values("N")
        if sub.empty:
            continue
        ax1.plot(sub["N"], sub["gflops"], marker="o", label=label, color=color, linewidth=2)

    ax1.set_xlabel("Matrix size N (N×N)")
    ax1.set_ylabel("GFLOPS")
    ax1.set_title("GFLOPS vs matrix size (CUDA variants)")
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    # Speedup over cuBLAS
    cublas_sub = df[df["variant"] == "cublas"].set_index("N")
    for variant, (color, label) in {
        "cuda_naive": (_ORANGE, "naive / cuBLAS"),
        "cuda_tiled": (_GREEN,  "tiled / cuBLAS"),
    }.items():
        sub = df[df["variant"] == variant].sort_values("N")
        if sub.empty or cublas_sub.empty:
            continue
        ratio = sub.set_index("N")["latency_ms_mean"] / cublas_sub["latency_ms_mean"]
        ratio = ratio.dropna()
        ax2.plot(ratio.index, ratio.values, marker="o", label=label, color=color, linewidth=2)

    ax2.axhline(1.0, linestyle="--", color=_BLUE, linewidth=1, label="cuBLAS baseline")
    ax2.set_xlabel("Matrix size N")
    ax2.set_ylabel("Latency ratio (lower = closer to cuBLAS)")
    ax2.set_title("Latency vs cuBLAS (1.0 = cuBLAS speed)")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, plots_dir, "matmul_performance.png")


# ---------------------------------------------------------------------------
def _plot_speedup(vec_rows: list[dict], red_rows: list[dict],
                  mat_rows: list[dict], plots_dir: Path) -> None:
    """Speedup over NumPy at the largest shared size where CPU data exists."""

    def _speedup_at_max(rows, cuda_variants):
        df   = pd.DataFrame(rows)
        # Only consider sizes where numpy baseline was measured
        has_numpy = df[df["variant"] == "numpy"]["N"].unique()
        if len(has_numpy) == 0:
            return {}, []
        max_n = max(has_numpy)
        sub   = df[df["N"] == max_n]
        result = {}
        for v in cuda_variants:
            row = sub[sub["variant"] == v]
            if not row.empty:
                result[v] = float(row["speedup_vs_numpy"].values[0])
        return result, max_n

    cuda_vec = ["cuda_naive", "cuda_opt", "torch_add_gpu"]
    cuda_red = ["reduce_naive", "reduce_sequential", "reduce_shuffle"]
    cuda_mat = ["cuda_naive", "cuda_tiled", "cublas"]

    sp_vec, n_vec = _speedup_at_max(vec_rows, cuda_vec)
    sp_red, n_red = _speedup_at_max(red_rows, cuda_red)
    sp_mat, n_mat = _speedup_at_max(mat_rows, cuda_mat)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Speedup vs NumPy (CPU baseline)", fontsize=13, fontweight="bold")

    configs = [
        (axes[0], sp_vec, cuda_vec,
         ["Naive", "Float4\ngrid-stride", "torch.add\n(GPU)"],
         f"Vector Add (N={n_vec//1_048_576}M)"),
        (axes[1], sp_red, cuda_red,
         ["Naive\n(divergent)", "Sequential\n(no div.)", "Warp\nShuffle"],
         f"Reduction (N={n_red//1_048_576}M)"),
        (axes[2], sp_mat, cuda_mat,
         ["Naive\n(global mem)", "Tiled\n(shared mem)", "cuBLAS\n(torch.mm)"],
         f"MatMul (N={n_mat})"),
    ]

    colors = [_ORANGE, _GREEN, _BLUE]

    for ax, sp_dict, variants, labels, title in configs:
        values = [sp_dict.get(v, 0.0) for v in variants]
        bars   = ax.bar(range(len(variants)), values, color=colors[:len(variants)], width=0.6)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Speedup (×)")
        ax.set_title(title)
        ax.axhline(1.0, linestyle="--", color=_GREY, linewidth=1, label="NumPy baseline")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, plots_dir, "speedup_comparison.png")


# ---------------------------------------------------------------------------
def _plot_latency(red_rows: list[dict], plots_dir: Path) -> None:
    """Latency vs array size for all reduction variants."""
    df    = pd.DataFrame(red_rows)
    sizes = sorted(df["N"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))

    for variant, color, label in [
        ("reduce_naive",      _ORANGE, "CUDA naive (interleaved)"),
        ("reduce_sequential", _GREEN,  "CUDA sequential (no divergence)"),
        ("reduce_shuffle",    _BLUE,   "CUDA warp shuffle"),
    ]:
        sub = df[df["variant"] == variant].sort_values("N")
        if sub.empty:
            continue
        ax.plot(sub["N"] / 1_048_576, sub["latency_ms_mean"],
                marker="o", label=label, color=color, linewidth=2)
        ax.fill_between(
            sub["N"] / 1_048_576,
            sub["latency_ms_mean"] - sub["latency_ms_std"],
            sub["latency_ms_mean"] + sub["latency_ms_std"],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Array size (millions of elements)")
    ax.set_ylabel("Kernel latency (ms)")
    ax.set_title("Reduction kernel latency vs input size")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    _save(fig, plots_dir, "latency_scaling.png")


# ---------------------------------------------------------------------------
def plot_all(
    vec_rows: list[dict],
    red_rows: list[dict],
    mat_rows: list[dict],
    plots_dir: Path,
    gpu_name:  str = "",
) -> None:
    _plot_bandwidth(vec_rows, red_rows, plots_dir, gpu_name)
    _plot_matmul(mat_rows, plots_dir)
    _plot_speedup(vec_rows, red_rows, mat_rows, plots_dir)
    _plot_latency(red_rows, plots_dir)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_results.py vec.csv reduction.csv matmul.csv")
        sys.exit(1)
    plot_all(
        pd.read_csv(sys.argv[1]).to_dict("records"),
        pd.read_csv(sys.argv[2]).to_dict("records"),
        pd.read_csv(sys.argv[3]).to_dict("records"),
        Path("results/plots"),
    )
