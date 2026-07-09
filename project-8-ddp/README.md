# Distributed Data Parallel (DDP) Training

Fine-tunes BERT-base on SST-2 sentiment classification four ways — no DDP, DDP built by hand from `torch.distributed` primitives, production `nn.parallel.DistributedDataParallel`, and manual DDP with `torch.distributed` itself removed and rebuilt from scratch — to demonstrate exactly what DDP does under the hood, at progressively deeper layers, and how the API abstracts it away.

**Hardware:** NVIDIA RTX 4080 Laptop GPU (Ada Lovelace) · CUDA 12.6 · Windows 11
**Model:** `bert-base-uncased` (110M params) · **Dataset:** GLUE SST-2 (binary sentiment)

---

## The four cases

### 1. Base — no DDP (`train_base.py`)
Single process, single GPU, standard PyTorch fine-tuning loop. The reference point every DDP variant is checked against for correctness and speed.

### 2. Manual DDP (`train_manual_ddp.py`)
Multiple processes (`torch.multiprocessing.spawn`), each holding a full model replica and training on its own shard (`DistributedSampler`) — but with **no `DistributedDataParallel` wrapper**. Gradient synchronization is done by hand:

1. Rank 0's weights are `dist.broadcast` to every other rank at startup, so all replicas start identical.
2. Each rank runs its own forward/backward pass on its local shard, producing local gradients.
3. After `loss.backward()`, every parameter's gradient is summed across ranks with `dist.all_reduce(..., op=SUM)` and divided by `world_size`.
4. Every replica now holds the identical averaged gradient and takes the identical optimizer step.

This is functionally what `DistributedDataParallel` does internally — it's just missing DDP's performance optimizations (bucketing gradients and overlapping communication with backward instead of waiting for it to finish).

### 3. `torch.distributed` + DDP (`train_torch_ddp.py`)
The standard production pattern: `dist.init_process_group`, `DistributedSampler`, and the model wrapped in `nn.parallel.DistributedDataParallel`. DDP registers autograd hooks that all-reduce each gradient bucket as soon as it's ready during `backward()`, so the training loop is otherwise identical to the base case — no manual sync code at all.

### 4. Manual DDP, from scratch (`train_manual_ddp_2.py`)
Same manual gradient-sync mechanics as case 2, but with `torch.multiprocessing.spawn` *and* `torch.distributed` itself removed and reimplemented from scratch on nothing but stdlib `multiprocessing`:

- **Process launch** (`launch()`): starts `world_size` `multiprocessing.Process`es running the worker function, joins them, fails loudly if one dies — exactly what `mp.spawn` does, minus the torch-specific CUDA-tensor-sharing machinery this project never needed (each rank builds its own model from the HF checkpoint rather than receiving one from another process).
- **Process group** (`ProcessGroup`): a hand-rolled hub-and-spoke replacement for `gloo`, built on `multiprocessing.Pipe`. Rank 0 is the hub; every other rank only talks to rank 0 over its own dedicated pipe; every "collective" is rank 0 looping over its connections to gather from everyone, then looping again to scatter the result back — `barrier()`, `broadcast_from_rank0()`, and `all_reduce_mean()` are ~10 lines each.

It's a genuine simplification (real backends connect every rank to every other rank and run ring/tree algorithms so no single rank bottlenecks — this doesn't scale past a handful of ranks), and it sidesteps both Windows findings below entirely: no `gloo`, so no CUDA-collective crash; no network rendezvous, so no `torchrun`/libuv bug either — the parent process just hands each `Pipe` end to the right child at spawn time. Verified to produce identical loss/accuracy to cases 2 and 3.

---

## Windows-specific findings

This machine has a **single physical GPU**, and `NCCL` isn't available on Windows at all — only the `gloo` backend is supported. Two real bugs in this PyTorch build (`2.12.1+cu126`) surfaced while getting `gloo` working here, both worth knowing about if you hit them elsewhere:

1. **`gloo`'s CUDA collectives crash on Windows.** `dist.all_reduce`/`dist.broadcast` on a CUDA tensor over `gloo` segfaults (access violation) on this build — confirmed with a minimal repro outside this project. CPU tensors work fine. The fix used throughout: bounce every collective through CPU (`.to("cpu")` → collective → `.to("cuda")` back), so forward/backward still run on GPU and only the sync step pays a CPU round-trip. For case 2 this is explicit (`common.manual_grad_allreduce`, `common.broadcast_from_rank0`); for case 3, DDP's default comm is replaced with a custom hook that does the same thing (`common.cpu_gloo_allreduce_hook`, registered via `model.register_comm_hook`).
2. **`torchrun`'s rendezvous bootstrap is broken on Windows in this build.** It creates its coordination `TCPStore` without `use_libuv=False`, and this build has no libuv support on Windows, so `torchrun` itself fails before launching any worker (`PyTorch was built without libuv support`) — even with `USE_LIBUV=0` set, since that env var isn't actually read on this code path. This is a torchrun bug, not a `train_torch_ddp.py` problem: `dist.init_process_group()`'s own `env://` path already defaults `use_libuv` to `False` on win32 and works correctly. Case 3 is launched with a small custom launcher (`launch_torch_ddp.py`) that sets the same `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT` env vars torchrun would, without going through its broken rendezvous store.

Case 4 sidesteps both of these entirely: no `gloo`, so no CUDA-collective crash; no network rendezvous, so no `torchrun`/libuv bug either — the parent process just hands each `multiprocessing.Pipe` end to the right child at spawn time.

Because of the single GPU, every DDP process (cases 2-4) also binds to the same `cuda:0` device. So these runs demonstrate DDP **mechanics and gradient-synchronization correctness**, not a wall-clock speedup — on real multi-GPU hardware with NCCL, cases 2-4 would show near-linear scaling instead of running roughly **10-20x slower** than case 1, which is what's actually measured here. That slowdown is mostly a CPU-bounce tax: case 2 and case 4 both pay one CPU round-trip *per parameter tensor* (~199 of them for BERT-base) every step, since neither is bucketed like real DDP; case 3 fares a bit better since DDP still buckets gradients into a handful of buffers before the comm hook bounces each one through CPU. In practice cases 2-4 land within noise of each other on wall-clock time — the per-parameter socket/pipe overhead dominates regardless of which transport (`gloo` vs. raw `Pipe`) carries it. As a correctness check: with the same seed, cases 2-4 all produce **identical** per-epoch loss and accuracy, confirming every layer of hand-rolled sync is doing exactly what `DistributedDataParallel` does.

Given that overhead, the default `--train-subset` (1000) is intentionally small so a full `run_all.py` finishes in a reasonable time rather than the well over an hour it'd take at, say, 4000+ examples — bump it up if you want a more meaningful accuracy number and don't mind the wait.

All four modes train on the same global batch size (default 32) and the same number of examples, so loss/accuracy curves are directly comparable.

---

## Usage

```powershell
cd project-8-ddp

# Run all four modes back-to-back and plot the comparison
python src/run_all.py

# Faster smoke test
python src/run_all.py --train-subset 200 --epochs 1

# Run one mode at a time
python src/train_base.py --epochs 2 --batch-size 32
python src/train_manual_ddp.py --world-size 2 --epochs 2 --batch-size 16
python src/launch_torch_ddp.py --world-size 2 --epochs 2 --batch-size 16
python src/train_manual_ddp_2.py --world-size 2 --epochs 2 --batch-size 16

# Skip a mode
python src/run_all.py --skip-manual-ddp-2

# Re-plot from existing CSVs
python src/plot_results.py
```

Results land in `results/` as `base.csv` / `manual_ddp.csv` / `torch_ddp.csv` / `manual_ddp_2.csv` plus comparison plots (training loss, validation accuracy, epoch time, throughput) under `results/plots/`.

---

## Files

| File | Purpose |
|---|---|
| `src/data.py` | SST-2 loading/tokenization, `DataLoader`/`DistributedSampler` construction |
| `src/model.py` | BERT tokenizer/model construction |
| `src/common.py` | Shared train loop, eval, manual gradient all-reduce, CPU-bounce comm hook, CSV logging |
| `src/train_base.py` | Case 1 |
| `src/train_manual_ddp.py` | Case 2 |
| `src/train_torch_ddp.py` | Case 3 |
| `src/launch_torch_ddp.py` | Multi-process launcher for case 3 (works around the torchrun/libuv bug above) |
| `src/train_manual_ddp_2.py` | Case 4 — case 2 with `torch.multiprocessing` and `torch.distributed` also reimplemented from scratch |
| `src/plot_results.py` | Comparison plots from all four CSVs |
| `src/run_all.py` | Runs all four modes, then plots |
