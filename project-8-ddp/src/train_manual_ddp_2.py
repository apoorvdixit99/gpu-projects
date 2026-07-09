"""Case 2b: Manual DDP with a from-scratch process launcher AND process group.

Same DDP mechanics as train_manual_ddp.py -- each rank holds a full model
replica, gradients are synced by hand instead of via nn.parallel.
DistributedDataParallel -- but with two more layers pulled out of torch
and reimplemented directly:

  1. torch.multiprocessing.spawn -> launch(), built on stdlib
     multiprocessing.Process: start world_size processes each running
     worker(rank, world_size, args), wait for all of them, fail loudly
     (killing any survivors) if one dies. Nothing torch-specific is
     needed here since the workers never share CUDA tensors across the
     process boundary -- each rank builds its own model straight from
     the HF checkpoint.

  2. torch.distributed (init_process_group / barrier / all_reduce /
     broadcast / destroy_process_group) -> ProcessGroup, a hand-rolled
     hub-and-spoke IPC layer built on stdlib multiprocessing.Pipe. Rank
     0 is the hub, every other rank only talks to rank 0 over its own
     dedicated Pipe, and every "collective" is rank 0 looping over its
     connections to gather from everyone, then looping again to scatter
     the result back:

       - barrier()             : hub recv()s a "ready" ping from every
                                  leaf, then send()s "go" back to every
                                  leaf
       - broadcast_from_rank0() : hub send()s a tensor to every leaf
       - all_reduce_mean()      : hub recv()s a tensor from every leaf,
                                  sums them (+ its own), averages,
                                  send()s the result back to every leaf

This is a genuine simplification: real backends (gloo/NCCL) connect
every rank to every other rank and run ring/tree algorithms so no single
rank is a bottleneck. Hub-and-spoke doesn't scale past a handful of
ranks, but it makes every step of a "process group" visible as plain
Python instead of a library call.

One more thing this removes almost as a side effect: no MASTER_ADDR/
MASTER_PORT/rendezvous handshake at all. torch.distributed needs those
because it's built for processes on different machines that have to find
each other over a network. Everything here runs on one machine, so the
parent process just creates the Pipe objects directly and hands each end
to the right child at spawn time -- there's nothing to "rendezvous."

Tensors are still bounced through CPU before going into a Pipe -- not
because of the Windows gloo+CUDA bug this project worked around
elsewhere (there's no gloo here at all), but because plain
multiprocessing doesn't know how to pickle a live CUDA tensor across a
process boundary the way torch.multiprocessing's custom reducers do.

Usage
-----
python src/train_manual_ddp_2.py --world-size 2
"""

from __future__ import annotations

import argparse
import multiprocessing as std_mp
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


class ProcessGroup:
    """From-scratch stand-in for a torch.distributed process group -- a
    hub-and-spoke topology instead of gloo/NCCL's smarter ones. See the
    module docstring for the tradeoff."""

    def __init__(self, rank: int, world_size: int, conn=None, conns=None):
        self.rank = rank
        self.world_size = world_size
        self.conn = conn    # non-zero ranks: their Pipe end to rank 0
        self.conns = conns  # rank 0 only: one Pipe end per other rank

    def barrier(self) -> None:
        if self.rank == 0:
            for conn in self.conns:
                conn.recv()  # wait for every rank's "ready" ping
            for conn in self.conns:
                conn.send(True)  # release everyone
        else:
            self.conn.send(True)
            self.conn.recv()

    def broadcast_from_rank0(self, tensor: torch.Tensor) -> None:
        if self.rank == 0:
            cpu_t = tensor.detach().to("cpu")
            for conn in self.conns:
                conn.send(cpu_t)
        else:
            received = self.conn.recv()
            tensor.copy_(received.to(tensor.device, non_blocking=True))

    def all_reduce_mean(self, tensor: torch.Tensor) -> None:
        cpu_t = tensor.detach().to("cpu")
        if self.rank == 0:
            total = cpu_t.clone()
            for conn in self.conns:
                total += conn.recv()
            total /= self.world_size
            for conn in self.conns:
                conn.send(total)
            tensor.copy_(total.to(tensor.device, non_blocking=True))
        else:
            self.conn.send(cpu_t)
            total = self.conn.recv()
            tensor.copy_(total.to(tensor.device, non_blocking=True))

    def destroy(self) -> None:
        if self.rank == 0:
            for conn in self.conns:
                conn.close()
        else:
            self.conn.close()


def build_pipes(world_size: int):
    """One Pipe per non-zero rank, connecting it to rank 0. Returns
    (hub_conns, leaf_conns): hub_conns[i] is rank 0's end of the pipe to
    rank i+1, leaf_conns[i] is rank i+1's end of that same pipe."""
    hub_conns, leaf_conns = [], []
    for _ in range(1, world_size):
        hub_end, leaf_end = std_mp.Pipe()
        hub_conns.append(hub_end)
        leaf_conns.append(leaf_end)
    return hub_conns, leaf_conns


def launch(worker_fn, world_size: int, args: argparse.Namespace) -> None:
    """From-scratch stand-in for torch.multiprocessing.spawn(..., join=True),
    extended to also wire up each rank's ProcessGroup connection before
    spawning."""
    ctx = std_mp.get_context("spawn")  # required on Windows; explicit everywhere else too
    hub_conns, leaf_conns = build_pipes(world_size)

    processes = []
    for rank in range(world_size):
        pg_init = ("hub", hub_conns) if rank == 0 else ("leaf", leaf_conns[rank - 1])
        p = ctx.Process(target=worker_fn, args=(rank, world_size, args, pg_init))
        processes.append(p)

    for p in processes:
        p.start()
    for conn in hub_conns + leaf_conns:
        conn.close()  # parent's copies; children keep theirs open
    for p in processes:
        p.join()

    failed = [(rank, p.exitcode) for rank, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise RuntimeError(f"Worker process(es) failed: {failed}")


def worker(rank: int, world_size: int, args: argparse.Namespace, pg_init) -> None:
    kind, conn_data = pg_init
    pg = ProcessGroup(rank, world_size, conns=conn_data) if kind == "hub" else ProcessGroup(rank, world_size, conn=conn_data)

    device = torch.device("cuda:0")
    set_seed(args.seed)

    tokenizer = build_tokenizer()
    train_ds, val_ds = load_sst2(tokenizer, max_length=args.max_length, train_subset=args.train_subset, seed=args.seed)
    train_loader, val_loader, sampler = build_dataloaders(
        train_ds, val_ds, args.batch_size, distributed=True, rank=rank, world_size=world_size, seed=args.seed
    )

    model = build_model().to(device)
    for p in model.parameters():
        pg.broadcast_from_rank0(p.data)  # every replica starts from rank 0's weights

    optimizer = AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        print(f"\nDevice        : {torch.cuda.get_device_name(0)}  (shared across {world_size} ranks)")
        print(f"Train examples: {len(train_ds)}   Val examples: {len(val_ds)}")
        print(f"Batch size    : {args.batch_size}/rank ({args.batch_size * world_size} global)   Epochs: {args.epochs}\n")

    def after_backward(m):
        for p in m.parameters():
            if p.grad is not None:
                pg.all_reduce_mean(p.grad)

    records = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        stats = train_one_epoch(model, train_loader, optimizer, device, after_backward=after_backward)
        if rank == 0:
            acc = evaluate(model, val_loader, device)
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
            record = {"epoch": epoch + 1, "mode": "manual_ddp_2", "world_size": world_size,
                       "val_accuracy": acc, "peak_mem_mb": peak_mem, **stats}
            records.append(record)
            print(f"[manual_ddp_2] epoch {epoch + 1}/{args.epochs}  loss={stats['train_loss']:.4f}  "
                  f"acc={acc:.4f}  time={stats['epoch_time_sec']:.1f}s  "
                  f"thpt={stats['throughput_samples_sec']:.1f} samp/s")
        pg.barrier()

    if rank == 0:
        save_metrics_csv(records, RESULTS_DIR / "manual_ddp_2.csv")
        print(f"\nResults saved -> results/manual_ddp_2.csv")

    pg.destroy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Case 2b: Manual DDP with a from-scratch process group (no torch.distributed)")
    p.add_argument("--world-size", type=int, default=2, metavar="N",
                   help="Number of DDP processes (default: 2, sharing the single GPU)")
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Warm the HF datasets/tokenizer cache once up front so the spawned
    # workers don't race each other on the first download.
    load_sst2(build_tokenizer(), max_length=args.max_length, train_subset=args.train_subset, seed=args.seed)

    launch(worker, args.world_size, args)


if __name__ == "__main__":
    main()
