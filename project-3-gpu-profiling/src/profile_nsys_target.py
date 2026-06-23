"""Nsight Systems / Nsight Compute profiling target for GPT-2.

This script is designed to be launched under nsys or ncu — do not run it
directly for profiling purposes.  NVTX ranges mark the inference region so
the Nsight timeline is easy to navigate.

Usage (via nsight/run_nsys.ps1)
--------------------------------
nsys profile --trace=cuda,nvtx,osrt ^
     --output=results/nsight/gpt2 ^
     python src/profile_nsys_target.py --batch-size 1 --seq-len 128

Usage (via nsight/run_ncu.ps1)
-------------------------------
ncu --set full --output=results/nsight/gpt2_ncu ^
    python src/profile_nsys_target.py --batch-size 1 --seq-len 64 --iterations 3
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import GPT2LMHeadModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size",  type=int, default=1)
    p.add_argument("--seq-len",     type=int, default=128)
    p.add_argument("--warmup",      type=int, default=10)
    p.add_argument("--iterations",  type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading GPT-2 (bs={args.batch_size}, seq={args.seq_len}) …")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().half().cuda()
    ids  = torch.randint(0, 50257, (args.batch_size, args.seq_len), dtype=torch.long, device="cuda")
    mask = torch.ones(args.batch_size, args.seq_len, dtype=torch.long, device="cuda")

    # Warmup outside the NVTX range so startup overhead is not captured.
    with torch.no_grad():
        for _ in range(args.warmup):
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    torch.cuda.synchronize()

    print(f"Running {args.iterations} profiled iterations …")

    torch.cuda.nvtx.range_push("gpt2_inference")
    with torch.no_grad():
        for i in range(args.iterations):
            torch.cuda.nvtx.range_push(f"forward_{i}")
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            torch.cuda.nvtx.range_pop()
    # Synchronize before range_pop so all kernels fall inside the NVTX range.
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("Done.")


if __name__ == "__main__":
    main()
