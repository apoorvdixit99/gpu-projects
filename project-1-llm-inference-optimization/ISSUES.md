# Known Issues & Fixes

Issues encountered during setup and first run, in chronological order.

---

## Issue 1 — Missing `onnxscript` module

**Error**
```
ModuleNotFoundError: No module named 'onnxscript'
```

**Cause**
PyTorch 2.x dynamo ONNX exporter depends on `onnxscript` but does not auto-install it as a dependency.

**Fix**
Added `onnxscript>=0.1.0` to `requirements.txt` and installed it:
```powershell
pip install onnxscript
```

---

## Issue 2 — PowerShell parse error (em dash in `setup.ps1`)

**Error**
```
Unexpected token ')' in expression or statement
```
on the `Write-Host` line containing `—`.

**Cause**
PowerShell 5.1 reads scripts as Windows-1252 by default. The UTF-8 em dash `—` (U+2014) is misread as a multi-byte sequence that the parser rejects.

**Fix**
Replaced all `—` characters with plain hyphens `-` in `setup.ps1`. Avoid any non-ASCII characters in `.ps1` files when targeting PowerShell 5.1.

---

## Issue 3 — ONNX export crash (`num_outputs` on `Split`, opset mismatch)

**Error**
```
onnx.checker.ValidationError: Unrecognized attribute: num_outputs for operator Split
==> Context: Bad node spec for node. Name: node_Split_18 OpType: Split
```

**Cause**
PyTorch 2.x uses the dynamo-based ONNX exporter by default, which generates opset-18 ops (`Split` with `num_outputs`, added in opset 18). Requesting `opset_version=17` triggered a downconversion that the ONNX converter cannot complete for this operator.

**Fix**
Bumped the default `opset` argument from `17` → `18` in `src/export_onnx.py`.

---

## Issue 4 — `model.eval()` warning during ONNX export

**Warning**
```
Exporting a model while it is in training mode. Please ensure that this is intended ...
Calling model.eval() before export is recommended.
```

**Cause**
`.eval()` was called on the inner `GPT2LMHeadModel` (`base.eval()`) but not on the `_GPT2Wrapper` instance that was actually passed to `torch.onnx.export`. PyTorch checks the top-level module.

**Fix**
Chained `.eval()` on the wrapper:
```python
model = _GPT2Wrapper(base).cuda().eval()
```

---

## Issue 5 — `dynamic_axes` deprecation warning

**Warning**
```
'dynamic_axes' is not recommended when dynamo=True, and may lead to
torch._dynamo.exc.UserError: Constraints violated.
Supply the 'dynamic_shapes' argument instead.
```

**Cause**
`dynamic_axes` is the legacy API for the TorchScript exporter. The new dynamo exporter (default in PyTorch 2.x) requires `dynamic_shapes` with `torch.export.Dim` objects.

**Fix**
Replaced `dynamic_axes` with `dynamic_shapes` in `src/export_onnx.py`:
```python
from torch.export import Dim

batch_dim = Dim("batch_size", min=1, max=64)
seq_dim   = Dim("seq_len",    min=1, max=1024)

dynamic_shapes={
    "input_ids":      {0: batch_dim, 1: seq_dim},
    "attention_mask": {0: batch_dim, 1: seq_dim},
}
```

---

## Issue 6 — `pynvml` deprecation warning

**Warning**
```
The pynvml package is deprecated. Please install nvidia-ml-py instead.
```

**Cause**
`pynvml` on PyPI is the old, unmaintained package. `nvidia-ml-py` is the official NVIDIA-maintained successor that provides the same `pynvml` Python import without the deprecation warning. PyTorch itself emits this warning when it finds the old package.

**Fix**
Replaced `pynvml>=11.5.0` with `nvidia-ml-py>=12.0.0` in `requirements.txt`, then swapped the packages:
```powershell
pip uninstall pynvml -y
pip install nvidia-ml-py
```

---

## Issue 7 — TensorRT builder crashes with CUDA initialization error 35

**Error**
```
[TRT] [E] createInferBuilder: Error Code 6: API Usage Error (CUDA initialization failure with error: 35)
TypeError: pybind11::init(): factory function returned nullptr
```

**Cause**
`build_trt.py` had no `import torch` and performed no CUDA operations before calling `trt.Builder()`. TensorRT's C++ layer calls `ensureCudaInitialized` internally; if no prior CUDA operation has run in the current process it finds no active CUDA context and returns `nullptr`, which surfaces as a pybind11 `TypeError`.

**Fix**
Added `import torch` to `src/build_trt.py` and called `torch.cuda.init()` immediately before `trt.Builder()` to guarantee the CUDA context is live:
```python
torch.cuda.init()
builder = trt.Builder(_LOGGER)
```

---

## Issue 8 — ONNX exporter warns axis name "will not be used"

**Warning**
```
UserWarning: # The axis name: batch_size will not be used, since it shares
the same shape constraints with another axis: batch_size.
```
(same for `seq_len`)

**Cause**
The ONNX exporter deduplicates axis constraints by name, not by object identity. Whether the same `Dim` instance is reused or two separate instances share the same string name, the exporter merges them into one constraint and warns that the second name is redundant.

**Status: benign — no complete fix available**
The model exports and verifies correctly (`ONNX model verified`). Axis names are only metadata labels in ONNX; they do not affect dynamic shape inference at runtime. TensorRT ignores them entirely and operates on tensor names (`input_ids`, `attention_mask`). Attempted using separate `Dim` instances with the same name — warning persists because the string name is what triggers the deduplication.

---

## Issue 9 — TensorRT CUDA runtime version mismatch (error 35)

**Error** *(persisted after Issue 7 fix)*
```
[TRT] [E] createInferBuilder: Error Code 6: API Usage Error
(CUDA initialization failure with error: 35)
TypeError: pybind11::init(): factory function returned nullptr
```

**Root cause**
`pip install tensorrt` from standard PyPI installed **TRT 11.1.0.106**, which bundles a CUDA runtime newer than 12.6. CUDA error 35 = `cudaErrorInsufficientDriver` — the GPU driver supports CUDA 12.6 but TRT's bundled runtime requires a newer driver version. PyTorch works because it uses the explicitly installed CUDA 12.6 wheel; TRT fails because it uses its own bundled runtime.

Verified with:
```powershell
python -c "import tensorrt; print(tensorrt.__version__)"  # 11.1.0.106
python -c "import torch; print(torch.version.cuda)"       # 12.6
```

**Fix**
Uninstall the mismatched package and install the CUDA-12-specific build from NVIDIA's own PyPI index:
```powershell
pip uninstall tensorrt -y
pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com
```

`tensorrt-cu12` is the NVIDIA-maintained package built against the CUDA 12.x runtime, avoiding the bundled-runtime mismatch. Updated `setup.ps1` to use this install path for all future environment setups.

---

## Issue 10 — `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` removed in TRT 10.x

**Error**
```
AttributeError: type object 'tensorrt_bindings.tensorrt.NetworkDefinitionCreati'
has no attribute 'EXPLICIT_BATCH'
```

**Cause**
In TensorRT 8.x and 9.x, networks had to be created with the `EXPLICIT_BATCH` flag to enable dynamic batch sizes. In TRT 10.x this flag was removed — explicit batch is now the default for all networks and the enum value no longer exists.

**Fix**
Removed the flag argument from `builder.create_network()` in `src/build_trt.py`:
```python
# Before (TRT 8/9):
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# After (TRT 10+):
network = builder.create_network()
```

---

## Issue 16 — TRT warning: default stream causes extra syncs in `execute_async_v3`

**Warning**
```
[TRT] [W] Using default stream in enqueueV3() may lead to performance issues
due to additional calls to cudaStreamSynchronize() by TensorRT to ensure
correct synchronization. Please use non-default stream instead.
```

**Cause**
`_TRTSession.infer()` was passing `torch.cuda.current_stream().cuda_stream` to `execute_async_v3`. If no custom stream has been created, this returns stream 0 (the default). When TRT receives the default stream, it inserts extra `cudaStreamSynchronize` calls around the kernel launch to guarantee correctness, adding unnecessary overhead and inflating latency measurements.

**Fix**
Added a dedicated `torch.cuda.Stream()` to `_TRTSession.__init__` and used it everywhere — in `execute_async_v3` and in the CUDA event recording in `_time_fn`. Events must be recorded on the same stream as the work they bracket to produce accurate elapsed times:
```python
# _TRTSession.__init__
self.stream = torch.cuda.Stream()

# infer()
self.context.execute_async_v3(self.stream.cuda_stream)

# _time_fn — events recorded on the dedicated stream
start_ev.record(stream)
fn()
end_ev.record(stream)
```

---

## Issue 15 — ONNX Runtime CUDA provider fails, silently falls back to CPU

**Error**
```
[ONNXRuntimeError] : 1 : FAIL : Error loading "onnxruntime_providers_cuda.dll"
which depends on "cublasLt64_13.dll" which is missing.
Warning: CUDAExecutionProvider not active — using ['CPUExecutionProvider']
```

**Symptom**
Benchmark appeared to run but latencies were 10-100x slower than expected (CPU times, e.g. ~200 ms for bs=8 seq=128 instead of ~10 ms).

**Cause**
`pip install onnxruntime-gpu>=1.20.0` resolved to a version that requires CUDA 13.x (`cublasLt64_13.dll`). The system has CUDA 12.6. ORT logged the error but did not raise — it silently fell back to CPU, making the benchmark appear to run normally while producing meaningless GPU comparison data. Same pattern as Issue 9 (TRT CUDA mismatch).

**Fix**
Pinned to the last CUDA-12.x-compatible release in `requirements.txt`:
```
onnxruntime-gpu<1.21.0
```

Reinstall:
```powershell
pip uninstall onnxruntime-gpu -y
pip install "onnxruntime-gpu<1.21.0"
```

---

## Issue 14 — `build_serialized_network` returns `IHostMemory`, not `bytes`

**Error**
```
TypeError: object of type 'tensorrt_bindings.tensorrt.IHostMemory' has no len()
```

**Cause**
In TRT 10.x, `builder.build_serialized_network()` returns a `tensorrt.IHostMemory` object rather than a plain `bytes` object. `IHostMemory` does not support `len()`, and writing it directly to a file may also fail. Use `.nbytes` for the size and convert explicitly with `bytes()` before writing.

**Fix**
Updated `src/build_trt.py` to convert before use:
```python
# Before:
f.write(serialized)
size_mb = len(serialized) / 1024 ** 2

# After:
engine_bytes = bytes(serialized)
f.write(engine_bytes)
size_mb = len(engine_bytes) / 1024 ** 2
```

---

## Issue 13 — `BuilderFlag.FP16` removed in TRT 10.x

**Error**
```
AttributeError: type object 'tensorrt_bindings.tensorrt.BuilderFlag' has no attribute 'FP16'
```

**Cause**
In TRT 10.x, the per-precision `BuilderFlag` entries (`FP16`, `INT8`, `BF16`, etc.) were removed. Precision selection is now automatic — TRT profiles the network and selects the fastest supported precision (FP16 on Ada/Ampere, FP8 on Ada) without requiring an explicit flag.

**Fix**
Removed the `config.set_flag(trt.BuilderFlag.FP16)` call from `src/build_trt.py`. The `fp16` parameter is kept in the function signature for future use (e.g. if NVIDIA reintroduces explicit control), but no flag is set:
```python
# Before:
config.set_flag(trt.BuilderFlag.FP16)

# After (TRT 10.x):
# Precision is chosen automatically; no flag needed.
```

---

## Issue 12 — `Builder.platform_has_fast_half` removed in TRT 10.x

**Error**
```
AttributeError: 'tensorrt_bindings.tensorrt.Builder' object has no attribute 'platform_has_fast_half'
```

**Cause**
`platform_has_fast_half` was a TRT 8/9 API for querying whether the GPU has a fast FP16 path. It was removed in TRT 10.x; TRT now manages precision capability checks internally.

**Fix**
Removed the guard in `src/build_trt.py` and set the FP16 flag unconditionally when `fp16=True`. TRT will respect the flag on capable hardware (RTX 40xx has full FP16 support) and ignore it otherwise:
```python
# Before:
if not builder.platform_has_fast_half:
    print("Warning: ...")
else:
    config.set_flag(trt.BuilderFlag.FP16)

# After:
config.set_flag(trt.BuilderFlag.FP16)
```

---

## Issue 11 — TRT parser can't find ONNX external weight file

**Error**
```
[TRT] [E] WeightsContext.cpp:197: Failed to open file: gpt2.onnx.data
[TRT] [E] Failed to import initializer: model.transformer.wpe.weight
RuntimeError: ONNX parse failed: ... UNSUPPORTED_NODE: Assertion failed:
ctx->getWeightsContext().convertOnnxWeights(initializer, &weights)
```

**Cause**
PyTorch 2.x's dynamo ONNX exporter splits the export into two files:
- `gpt2.onnx` — the graph structure (~1.3 MB)
- `gpt2.onnx.data` — the model weights (~548 MB)

The original `build_trt.py` called `parser.parse(raw_bytes)`, which reads bytes with no file-path context. TRT looked for `gpt2.onnx.data` relative to the current working directory (wherever the script was launched from) rather than the ONNX file's directory, and failed to find it.

**Fix**
Replaced `parser.parse(raw_bytes)` with `parser.parse_from_file(onnx_path)` in `src/build_trt.py`. This API accepts a file path and resolves external data files relative to the ONNX file's own directory:
```python
# Before:
with open(onnx_path, "rb") as f:
    raw = f.read()
parser.parse(raw)

# After:
parser.parse_from_file(onnx_path)
```

---

## Issue 17 — `torch.compile` crashes: Triton not available on Windows

**Error**
```
torch._inductor.exc.TritonMissing: Cannot find a working triton installation.
Either the package is not installed or it is too old.
```
also preceded by:
```
W ... [0/0] Not enough SMs to use max_autotune_gemm mode
```

**Cause**
`torch.compile` defaults to the `inductor` backend, which uses Triton to JIT-compile CUDA kernels. Triton has no Windows build; it is Linux-only. Using `mode="reduce-overhead"` also invokes `inductor` under the hood, so the crash is the same regardless of the mode string.

**Fix**
Switched to `backend="cudagraphs"` in `src/bench_pytorch_compile.py`:
```python
# Before:
model = torch.compile(model, mode="reduce-overhead")

# After:
model = torch.compile(model, backend="cudagraphs")
```
`cudagraphs` records the forward pass as a CUDA graph and replays it on every call, eliminating per-kernel Python launch overhead — the same mechanism `reduce-overhead` was targeting — without requiring Triton.

---

## Issue 18 — `cudagraphs` backend crashes when input shape changes between iterations

**Error**
```
RuntimeError: The size of tensor a (128) must match the size of tensor b (256)
at non-singleton dimension 1
```
raised inside `torch._inductor.cudagraph_trees._copy_inputs_and_remove_from_src`.

**Cause**
CUDA graphs capture a static computation graph tied to a specific input shape. A single `torch.compile(model, backend="cudagraphs")` instance captures the graph on the first warmup call (e.g. `seq_len=128`). When the next `(batch_size, seq_len)` combination is processed, the runtime tries to replay the same graph with different-sized tensors, which fails because the captured kernel bindings have fixed dimensions.

**Fix**
Reset dynamo state and create a fresh compiled wrapper for each `(batch_size, seq_len)` combination. The base model is loaded once and reused; only the compiled wrapper changes:
```python
base = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
# ...
for bs in batch_sizes:
    for seq_len in seq_lens:
        torch._dynamo.reset()                           # clear previous graphs
        model = torch.compile(base, backend="cudagraphs")  # fresh wrapper per shape
        # warmup + timing as normal
```
`torch._dynamo.reset()` clears all compiled graph state globally, ensuring the new wrapper captures a fresh graph for the current shape. The warmup loop absorbs the compilation cost.

---

## Warnings — benign, no fix required

These warnings come from PyTorch internals and do not affect correctness or performance.

### `torchvision is not installed`
```
torch.onnx._internal.exporter._registration: torchvision is not installed.
Skipping torchvision::nms / roi_align / roi_pool
```
The ONNX op registry scans for optional torchvision ops at import time. Safe to ignore; GPT-2 uses none of these ops.

### `triton not found; flop counting will not work`
```
torch.utils.flop_counter: triton not found; flop counting will not work for triton kernels
```
PyTorch's FLOP counter tries to import Triton for kernel-level profiling. Does not affect inference benchmarking.

### `isinstance(treespec, LeafSpec) is deprecated` (FutureWarning)
```
copyreg.py:105: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated
```
An internal PyTorch tree-structure utility emits this during ONNX graph tracing. Expected to be cleaned up in a future PyTorch release.
