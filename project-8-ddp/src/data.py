"""SST-2 (GLUE) loading, tokenization, and dataloader construction.

Shared by all three training modes so that the base run and both DDP runs
fine-tune on exactly the same tokenized examples (only the sharding differs).
"""

from __future__ import annotations

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def load_sst2(tokenizer, max_length: int = 128, train_subset: int | None = None, seed: int = 42):
    ds = load_dataset("nyu-mll/glue", "sst2")

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length, padding="max_length")

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_ds = ds["train"]
    if train_subset is not None:
        train_ds = train_ds.shuffle(seed=seed).select(range(train_subset))
    val_ds = ds["validation"]
    return train_ds, val_ds


def build_dataloaders(
    train_ds,
    val_ds,
    batch_size: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
):
    """Returns (train_loader, val_loader, train_sampler). train_sampler is None
    unless distributed=True — call sampler.set_epoch(epoch) each epoch when set,
    so every rank reshuffles identically-seeded but disjoint shards."""
    sampler = None
    if distributed:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, sampler
