"""Shared training loop, evaluation, and metric logging for all three modes."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, after_backward=None) -> dict:
    """One epoch of standard supervised fine-tuning.

    `after_backward`, if given, is called with `model` right after `loss.backward()`
    and before `optimizer.step()` — this is where manual DDP hooks in its
    gradient all_reduce (case 3's DDP wrapper does the equivalent internally,
    via hooks fired during backward itself, so no callback is needed there).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        if after_backward is not None:
            after_backward(model)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        n_samples += batch["labels"].size(0)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    return {
        "train_loss": total_loss / n_batches,
        "epoch_time_sec": elapsed,
        "throughput_samples_sec": n_samples / elapsed,
    }


@torch.no_grad()
def evaluate(model, val_loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in val_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    model.train()
    return correct / total


def manual_grad_allreduce(model, world_size: int) -> None:
    """The core of "manual DDP": sum each parameter's gradient across ranks
    and divide by world_size, i.e. exactly what nn.parallel.DistributedDataParallel
    does internally via backward hooks (minus its bucketing/overlap optimizations).

    gloo's CUDA collectives crash (access violation) on Windows in this
    PyTorch build, so each gradient is bounced through CPU for the
    all_reduce and copied back rather than reduced in place on the GPU
    tensor. See cpu_gloo_allreduce_hook for the DDP-wrapper equivalent."""
    for p in model.parameters():
        if p.grad is not None:
            cpu_grad = p.grad.to("cpu")
            dist.all_reduce(cpu_grad, op=dist.ReduceOp.SUM)
            p.grad.copy_(cpu_grad.to(p.grad.device, non_blocking=True) / world_size)


def broadcast_from_rank0(model) -> None:
    """Broadcasts rank 0's parameters to every other rank, bounced through
    CPU for the same reason as manual_grad_allreduce."""
    for p in model.parameters():
        cpu_p = p.data.to("cpu")
        dist.broadcast(cpu_p, src=0)
        p.data.copy_(cpu_p.to(p.data.device, non_blocking=True))


def cpu_gloo_allreduce_hook(process_group, bucket):
    """DDP comm hook: replaces DDP's default gradient-bucket all_reduce
    (which crashes on gloo+CUDA on Windows) with a CPU-bounced version,
    while forward/backward compute stays on the GPU."""
    buf = bucket.buffer()
    cpu_buf = buf.to("cpu")
    dist.all_reduce(cpu_buf, op=dist.ReduceOp.SUM, group=process_group)
    cpu_buf /= dist.get_world_size(process_group)
    buf.copy_(cpu_buf.to(buf.device, non_blocking=True))
    fut: torch.futures.Future = torch.futures.Future()
    fut.set_result(buf)
    return fut


def save_metrics_csv(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
