"""Case 1: Base — single-process, single-GPU BERT fine-tuning (no DDP).

The reference point every DDP variant is compared against: same data
(SST-2), same model (bert-base-uncased), same global batch size, same
number of epochs.

Usage
-----
python src/train_base.py
python src/train_base.py --epochs 3 --batch-size 32 --train-subset 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(Path(__file__).parent))

from common import evaluate, save_metrics_csv, set_seed, train_one_epoch
from data import build_dataloaders, load_sst2
from model import build_model, build_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Case 1: Base single-GPU BERT fine-tuning on SST-2 (no DDP)")
    p.add_argument("--epochs", type=int, default=2, metavar="N")
    p.add_argument("--batch-size", type=int, default=32, metavar="N", help="Global batch size (default: 32)")
    p.add_argument("--lr", type=float, default=2e-5, metavar="LR")
    p.add_argument("--max-length", type=int, default=128, metavar="N")
    p.add_argument("--train-subset", type=int, default=1000, metavar="N",
                   help="Number of training examples to use (default: 1000)")
    p.add_argument("--seed", type=int, default=42, metavar="N")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0")
    set_seed(args.seed)

    tokenizer = build_tokenizer()
    train_ds, val_ds = load_sst2(tokenizer, max_length=args.max_length, train_subset=args.train_subset, seed=args.seed)
    train_loader, val_loader, _ = build_dataloaders(train_ds, val_ds, args.batch_size)

    model = build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(f"\nDevice        : {torch.cuda.get_device_name(0)}")
    print(f"Train examples: {len(train_ds)}   Val examples: {len(val_ds)}")
    print(f"Batch size    : {args.batch_size}   Epochs: {args.epochs}\n")

    records = []
    for epoch in range(args.epochs):
        stats = train_one_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, val_loader, device)
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
        record = {"epoch": epoch + 1, "mode": "base", "world_size": 1, "val_accuracy": acc, "peak_mem_mb": peak_mem, **stats}
        records.append(record)
        print(f"[base] epoch {epoch + 1}/{args.epochs}  loss={stats['train_loss']:.4f}  "
              f"acc={acc:.4f}  time={stats['epoch_time_sec']:.1f}s  "
              f"thpt={stats['throughput_samples_sec']:.1f} samp/s")

    save_metrics_csv(records, RESULTS_DIR / "base.csv")
    print(f"\nResults saved -> results/base.csv")


if __name__ == "__main__":
    main()
