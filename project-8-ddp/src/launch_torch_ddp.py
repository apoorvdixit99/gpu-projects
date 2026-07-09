"""Launches train_torch_ddp.py as N subprocesses with the RANK / WORLD_SIZE /
LOCAL_RANK / MASTER_ADDR / MASTER_PORT environment variables torchrun would
normally set for each worker.

Why not just use torchrun: torchrun's own rendezvous bootstrap creates its
coordination TCPStore without passing use_libuv=False, and this PyTorch
build has no libuv support on Windows, so torchrun itself fails before a
single worker even launches ("PyTorch was built without libuv support").
That's a bug in torchrun's launcher, not in train_torch_ddp.py — its own
dist.init_process_group() call uses the env:// init path, which already
defaults use_libuv to False on win32 (torch/distributed/rendezvous.py) and
works correctly. This script reproduces just enough of torchrun's env-var
contract to route around the broken piece.

Usage
-----
python src/launch_torch_ddp.py --world-size 2 --epochs 2 --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Manual multi-process launcher for train_torch_ddp.py")
    p.add_argument("--world-size", type=int, default=2, metavar="N")
    p.add_argument("--port", type=int, default=29500, metavar="N")
    args, extra = p.parse_known_args()
    return args, extra


def main() -> None:
    args, extra = parse_args()

    env_base = os.environ.copy()
    env_base["MASTER_ADDR"] = "localhost"
    env_base["MASTER_PORT"] = str(args.port)
    env_base["WORLD_SIZE"] = str(args.world_size)

    procs = []
    for rank in range(args.world_size):
        env = env_base.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        cmd = [sys.executable, str(SRC / "train_torch_ddp.py"), *extra]
        procs.append(subprocess.Popen(cmd, env=env))

    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
