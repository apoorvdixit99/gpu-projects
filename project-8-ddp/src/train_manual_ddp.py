"""Case 2: Manual DDP — torch.distributed primitives with a hand-rolled
gradient sync, no nn.parallel.DistributedDataParallel wrapper.

Each rank holds its own full model replica and its own data shard
(DistributedSampler). After loss.backward() populates local gradients,
every rank's gradients are summed via dist.all_reduce and divided by
world_size, so every replica takes an identical optimizer step — this is
exactly what DistributedDataParallel does internally (minus its
gradient-bucketing / backward-overlap performance optimizations). Initial
weights are broadcast from rank 0 so every replica starts identical, the
same way DDP's constructor does.

NCCL is not available on Windows, so this uses the gloo backend. All
--world-size processes share the single physical GPU on this machine
(gloo doesn't require one GPU per rank), so this demonstrates DDP
mechanics/correctness rather than a wall-clock speedup.

Usage
-----
python src/train_manual_ddp.py --world-size 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(Path(__file__).parent))

from common import broadcast_from_rank0, evaluate, manual_grad_allreduce, save_metrics_csv, set_seed, train_one_epoch
from data import build_dataloaders, load_sst2
from model import build_model, build_tokenizer


def worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    device = torch.device("cuda:0")
    set_seed(args.seed)

    tokenizer = build_tokenizer()
    train_ds, val_ds = load_sst2(tokenizer, max_length=args.max_length, train_subset=args.train_subset, seed=args.seed)
    train_loader, val_loader, sampler = build_dataloaders(
        train_ds, val_ds, args.batch_size, distributed=True, rank=rank, world_size=world_size, seed=args.seed
    )

    model = build_model().to(device)
    broadcast_from_rank0(model)  # every replica starts from rank 0's weights

    optimizer = AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        print(f"\nDevice        : {torch.cuda.get_device_name(0)}  (shared across {world_size} ranks)")
        print(f"Train examples: {len(train_ds)}   Val examples: {len(val_ds)}")
        print(f"Batch size    : {args.batch_size}/rank ({args.batch_size * world_size} global)   Epochs: {args.epochs}\n")

    records = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        stats = train_one_epoch(
            model, train_loader, optimizer, device,
            after_backward=lambda m: manual_grad_allreduce(m, world_size),
        )
        if rank == 0:
            acc = evaluate(model, val_loader, device)
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
            record = {"epoch": epoch + 1, "mode": "manual_ddp", "world_size": world_size,
                       "val_accuracy": acc, "peak_mem_mb": peak_mem, **stats}
            records.append(record)
            print(f"[manual_ddp] epoch {epoch + 1}/{args.epochs}  loss={stats['train_loss']:.4f}  "
                  f"acc={acc:.4f}  time={stats['epoch_time_sec']:.1f}s  "
                  f"thpt={stats['throughput_samples_sec']:.1f} samp/s")
        dist.barrier()

    if rank == 0:
        save_metrics_csv(records, RESULTS_DIR / "manual_ddp.csv")
        print(f"\nResults saved -> results/manual_ddp.csv")

    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Case 2: Manual DDP fine-tuning of BERT on SST-2")
    p.add_argument("--world-size", type=int, default=2, metavar="N",
                   help="Number of DDP processes (default: 2, sharing the single GPU)")
    p.add_argument("--epochs", type=int, default=2, metavar="N")
    p.add_argument("--batch-size", type=int, default=16, metavar="N", help="Per-rank batch size (default: 16)")
    p.add_argument("--lr", type=float, default=2e-5, metavar="LR")
    p.add_argument("--max-length", type=int, default=128, metavar="N")
    p.add_argument("--train-subset", type=int, default=1000, metavar="N",
                   help="Total training examples, split across ranks (default: 1000)")
    p.add_argument("--seed", type=int, default=42, metavar="N")
    p.add_argument("--port", type=int, default=29500, metavar="N")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Warm the HF datasets/tokenizer cache once up front so the spawned
    # workers don't race each other on the first download.
    load_sst2(build_tokenizer(), max_length=args.max_length, train_subset=args.train_subset, seed=args.seed)

    mp.spawn(worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
