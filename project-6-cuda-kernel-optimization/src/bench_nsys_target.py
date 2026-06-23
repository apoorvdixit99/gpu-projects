"""Thin target script for Nsight Systems / Nsight Compute profiling.

Runs a single kernel with NVTX range annotations so Nsight Systems produces
a short, navigable trace with clearly labelled iterations.  Not intended to
be called directly — use nsight/run_nsys.ps1 or nsight/run_ncu.ps1.

NVTX ranges in the trace
------------------------
  cuda_kernel_opt   outer range: covers all profiled iterations
  iter_N            per-iteration inner range; GPU idle gaps between bars
                    indicate kernel launch overhead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _ext import get_ext


_KERNELS = [
    "vec_add_naive", "vec_add_opt",
    "matmul_naive",  "matmul_tiled",
    "reduce_naive",  "reduce_sequential", "reduce_shuffle",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nsight profiling target for CUDA kernels")
    p.add_argument("--kernel",     choices=_KERNELS, default="matmul_tiled",
                   help="Which kernel to profile (default: matmul_tiled)")
    p.add_argument("--size",       type=int, default=1024,
                   help="N for vector/reduction (millions of elements) or matrix side length (default: 1024)")
    p.add_argument("--warmup",     type=int, default=5,
                   help="Warmup iterations before NVTX range (default: 5)")
    p.add_argument("--iterations", type=int, default=20,
                   help="Profiled iterations inside NVTX range (default: 20)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    ext = get_ext()
    kernel_fn = getattr(ext, args.kernel)

    # ── Build inputs ─────────────────────────────────────────────────────────
    if "vec_add" in args.kernel or "reduce" in args.kernel:
        N = args.size * 1_048_576      # size arg = millions of elements
        N = (N // 4) * 4              # ensure float4 alignment
        A = torch.randn(N, device="cuda")
        B = torch.randn(N, device="cuda") if "vec_add" in args.kernel else None
        def run():
            if B is not None:
                kernel_fn(A, B)
            else:
                kernel_fn(A)
    else:
        M = args.size
        A = torch.randn(M, M, device="cuda")
        B = torch.randn(M, M, device="cuda")
        def run():
            kernel_fn(A, B)

    print(f"Device  : {torch.cuda.get_device_name(0)}")
    print(f"Kernel  : {args.kernel}")
    print(f"Size    : {args.size}")
    print(f"Warmup  : {args.warmup}   Iterations : {args.iterations}")

    # ── Warmup (outside NVTX range) ──────────────────────────────────────────
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    # ── Profiled iterations with NVTX annotations ─────────────────────────────
    torch.cuda.nvtx.range_push(f"cuda_kernel_opt_{args.kernel}")
    for i in range(args.iterations):
        torch.cuda.nvtx.range_push(f"iter_{i}")
        run()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
