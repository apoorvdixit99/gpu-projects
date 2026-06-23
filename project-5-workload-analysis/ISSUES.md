# Known Issues & Fixes

Issues encountered during development and first run, in chronological order.

---

## Issue 1 — ResNet-50 crashes with input/weight dtype mismatch

**Error**
```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor)
should be the same
```
at `measure_latency.py`, line `model(**inputs)`.

**Cause**
`torch.randn(...)` produces a `float32` tensor by default.  When the model is loaded in
FP16 (`model.half()`), all weight tensors are `torch.float16`, but the vision input
tensor remains `torch.float32`.  PyTorch's `Conv2d` requires the input and weight
dtypes to match exactly — it does not silently upcast.

NLP models (GPT-2, BERT) are not affected because their inputs are integer tensors
(`torch.long` token IDs), which are not subject to dtype matching with the float weights.

**Fix**
After creating inputs in `measure_latency.py` and `measure_memory.py`, cast any
floating-point tensors to the model's parameter dtype before the forward call:
```python
raw_inputs  = spec.make_inputs(bs)
model_dtype = next(model.parameters()).dtype
inputs = {
    k: v.to(model_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
    for k, v in raw_inputs.items()
}
```
The `isinstance(v, torch.Tensor)` guard preserves scalar Python values such as
`use_cache=False` in GPT-2 inputs without attempting `.to()` on them.

---

## Issue 2 — Transformer FLOPs wildly wrong from torchinfo (BERT 100× low; GPT-2 10× high)

**Symptom** (first run)
```
GPT-2 (124M)       params 163.0M | MACs 163.34G | FLOPs 326.69G   [torchinfo]
DistilGPT-2 (82M)  params 120.5M | MACs  81.71G | FLOPs 163.42G   [torchinfo]
BERT-base (110M)   params 109.5M | MACs   0.11G | FLOPs   0.22G   [torchinfo]
```
Expected (analytical per-layer breakdown, seq=128, h=768, f=3072):

| Op                     | MACs / layer | × 12 layers |
|------------------------|--------------|-------------|
| QKV projections        | 226 M        | 2.72 G      |
| QK attention scores    | 12.6 M       | 0.15 G      |
| Attention × value      | 12.6 M       | 0.15 G      |
| Output projection      | 75.5 M       | 0.91 G      |
| FFN (two linears)      | 604 M        | 7.24 G      |
| **Transformer total**  | **931 M**    | **11.2 G**  |
| LM head (GPT-2 only)   | —            | + 4.94 G    |
| Pooler (BERT only)     | —            | + ~0 G      |

Expected totals: GPT-2 ≈ 16.1 G MACs; DistilGPT-2 ≈ 10.5 G MACs; BERT ≈ 11.2 G MACs.
torchinfo reports BERT at 100× too low and GPT-2 at ~10× too high.

**Root cause — torchinfo 1.8.0 bug with 3D `nn.Linear` inputs**
Diagnosed by inspecting `LayerInfo.macs` per module:

```
Linear [1, 128, 768] -> [1, 128, 768]   MACs = 0.59 M   (reported)
                                         MACs = 75.5 M   (expected: 128 × 768 × 768)
```

torchinfo 1.8.0 computes MACs for `nn.Linear` as `out_features × in_features`
(~0.59 M), ignoring the batch and **sequence** dimensions.  For a 3D input
`(batch=1, seq=128, features=768)` it should be `1 × 128 × 768 × 768 = 75.5 M MACs`,
but torchinfo treats the sequence dimension as part of a "batch" it doesn't count.
This produces results that are `seq_len`× too low for all BERT linear layers.

GPT-2 escapes the bug because HuggingFace's `Conv1D` reshapes its input to 2D with
`x.view(-1, x.size(-1))` before calling `torch.addmm`.  torchinfo sees a 2D matmul
`(128, 768) × (768, 2304)` and counts `128 × 768 × 2304 = 226 M MACs` correctly.
This explains why GPT-2 is overcounted relative to analytical: `Conv1D` happens to
give torchinfo the right 2D view, while element-wise ops (LayerNorm, GELU, residuals)
may also be inflating the total.

**Failed first attempt**
Initial attempt set `m._attn_implementation = 'eager'` on module instances
post-construction.  This has no effect: `attn_implementation` governs which class
(`BertSdpaSelfAttention` vs `BertSelfAttention`) is chosen at `__init__` time.
The bug is not in which attention kernel runs — it is in how torchinfo counts
`nn.Linear` MACs for 3D inputs regardless of attention type.

**Fix — analytical MACs for transformer models**
`ModelSpec` was extended with an optional `analytical_macs: int | None` field.
`models.py` computes the exact MAC count from architecture hyperparameters:

```python
def _transformer_macs(seq_len, hidden, ffn_dim, num_layers,
                       vocab_size=None, has_pooler=False) -> int:
    per_layer = (
        4 * seq_len * hidden * hidden        # QKV + output proj
        + 2 * seq_len * seq_len * hidden     # attn scores + weighted sum
        + 2 * seq_len * hidden * ffn_dim     # FFN
    )
    total = num_layers * per_layer
    if vocab_size: total += seq_len * hidden * vocab_size  # LM head
    if has_pooler: total += hidden * hidden                 # pooler
    return total
```

`measure_flops.py` uses `spec.analytical_macs` when set and falls back to torchinfo
for ResNet-50 (where torchinfo's `nn.Conv2d` counting is accurate).  The source is
labelled in the output (`[analytical]` vs `[torchinfo]`).

BERT is also loaded with `attn_implementation="eager"` so the inference timing uses
the same explicit-matmul attention path that the analytical formula accounts for.

---

## Issue 3 — UnicodeEncodeError on `→` in Windows terminal

**Error**
```
UnicodeEncodeError: 'charmap' codec can't encode character '→' in position N:
character maps to <undefined>
```
at any `print()` call in `run_analysis.py` containing the `→` arrow character.

**Cause**
Windows console defaults to code page 1252 (Western European Latin-1).  The Unicode
right-arrow `→` (U+2192) has no mapping in cp1252, so Python's default `sys.stdout`
encoder raises `UnicodeEncodeError` before anything is printed.

**Fix**
Replace every `→` in `run_analysis.py` print statements with the ASCII two-character
sequence `->`:
```python
# Before
print(f"Results saved → results/flops_{ts}.csv")
# After
print(f"Results saved -> results/flops_{ts}.csv")
```
No change to logic; purely cosmetic.

---

## Issue 4 — BERT LOAD REPORT: UNEXPECTED keys at model load

**Symptom**
Every BERT load prints a `[transformers] BertModel LOAD REPORT` table listing seven
keys as UNEXPECTED:
```
cls.predictions.transform.LayerNorm.weight  | UNEXPECTED
cls.seq_relationship.bias                   | UNEXPECTED
...
```

**Cause**
The `bert-base-uncased` checkpoint on HuggingFace was saved from a `BertForPreTraining`
model, which includes the MLM prediction head (`cls.predictions.*`) and the NSP head
(`cls.seq_relationship.*`).  Loading into `BertModel` (the bare encoder without heads)
silently discards those weights.  Transformers 5.x logs each discarded key explicitly.

**Resolution**
This is expected and harmless — the encoder weights are loaded correctly.  The
UNEXPECTED keys correspond to task-specific head layers that `BertModel` does not have.
No fix is required.  To suppress the log output, call
`transformers.logging.set_verbosity_error()` before the `from_pretrained` call.
