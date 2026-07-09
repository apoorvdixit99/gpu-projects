"""Case 3: Production DDP — torch.distributed + nn.parallel.DistributedDataParallel
+ DistributedSampler, the standard PyTorch multi-GPU training pattern.

Compared to train_manual_ddp.py, the DDP wrapper takes over gradient
synchronization entirely: it registers autograd hooks that all_reduce each
gradient bucket as soon as it's ready, overlapping communication with the
rest of backward() instead of waiting until backward is fully done. The
training loop itself is otherwise identical to the base case.

NCCL is not available on Windows, so this uses the gloo backend. gloo's
CUDA collectives crash on Windows in this PyTorch build, so DDP's default
gradient all_reduce is replaced with a CPU-bounced comm hook
(common.cpu_gloo_allreduce_hook) — forward/backward still run on the GPU.

Meant to be launched the way torchrun would launch it (one process per
rank, RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT set per process);
see launch_torch_ddp.py for why this repo uses a small custom launcher
instead of torchrun itself on this machine. All ranks share the single
physical GPU here, so this demonstrates DDP mechanics/correctness rather
than a wall-clock speedup.

Usage
-----
python src/launch_torch_ddp.py --world-size 2 --epochs 2 --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(Path(__file__).parent))

from common import cpu_gloo_allreduce_hook, evaluate, save_metrics_csv, set_seed, train_one_epoch
from data import build_dataloaders, load_sst2
from model import build_model, build_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Case 3: torch.distributed DDP fine-tuning of BERT on SST-2")
    p.add_argument("--epochs", type=int, default=2, metavar="N")
    p.add_argument("--batch-size", type=int, default=16, metavar="N", help="Per-rank batch size (default: 16)")
    p.add_argument("--lr", type=float, default=2e-5, metavar="LR")
    p.add_argument("--max-length", type=int, default=128, metavar="N")
    p.add_argument("--train-subset", type=int, default=1000, metavar="N",
                   help="Total training examples, split across ranks (default: 1000)")
    p.add_argument("--seed", type=int, default=42, metavar="N")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    # torchrun sets these environment variables for every launched process.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="gloo")
    device = torch.device("cuda:0")
    set_seed(args.seed)

    tokenizer = build_tokenizer()
    train_ds, val_ds = load_sst2(tokenizer, max_length=args.max_length, train_subset=args.train_subset, seed=args.seed)
    train_loader, val_loader, sampler = build_dataloaders(
        train_ds, val_ds, args.batch_size, distributed=True, rank=rank, world_size=world_size, seed=args.seed
    )

    model = build_model().to(device)
    model = DDP(model, device_ids=[0])
    model.register_comm_hook(state=dist.group.WORLD, hook=cpu_gloo_allreduce_hook)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        print(f"\nDevice        : {torch.cuda.get_device_name(0)}  (shared across {world_size} ranks)")
        print(f"Train examples: {len(train_ds)}   Val examples: {len(val_ds)}")
        print(f"Batch size    : {args.batch_size}/rank ({args.batch_size * world_size} global)   Epochs: {args.epochs}\n")

    records = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        stats = train_one_epoch(model, train_loader, optimizer, device)  # DDP all-reduces during .backward()
        if rank == 0:
            acc = evaluate(model.module, val_loader, device)
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
            record = {"epoch": epoch + 1, "mode": "torch_ddp", "world_size": world_size,
                       "val_accuracy": acc, "peak_mem_mb": peak_mem, **stats}
            records.append(record)
            print(f"[torch_ddp] epoch {epoch + 1}/{args.epochs}  loss={stats['train_loss']:.4f}  "
                  f"acc={acc:.4f}  time={stats['epoch_time_sec']:.1f}s  "
                  f"thpt={stats['throughput_samples_sec']:.1f} samp/s")
        dist.barrier()

    if rank == 0:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_metrics_csv(records, RESULTS_DIR / "torch_ddp.csv")
        print(f"\nResults saved -> results/torch_ddp.csv")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
