# Known Issues & Fixes

Issues encountered during setup and first run, in chronological order.

---

## Issue 1 — PyTorch requires newer CUDA driver than installed

**Error**
```
RuntimeError: The NVIDIA driver on your system is too old (found version 12060).
```

**Cause**  
The `.venv2` had `torch 2.11.0+cu130` installed, which requires a CUDA 13.0-capable driver. The Windows driver (560.76) supports only up to CUDA 12.6. PyTorch reports the driver capability as `12060` (CUDA 12.6.0).

**Fix**  
Reinstall PyTorch targeting the CUDA 12.4 wheel index, which is backward compatible with the CUDA 12.6 driver:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Issue 2 — pip does not downgrade torch when re-running install

**Symptom**  
After running `pip install torch --index-url https://download.pytorch.org/whl/cu124`, `pip show torch` still showed `2.11.0`. The cu124 install was silently skipped.

**Cause**  
pip will not downgrade a package unless explicitly told to. `torch 2.11.0 > 2.6.0+cu124`, so pip considered the existing version satisfactory.

**Fix**  
Use `--force-reinstall` to override, or specify an exact version:
```bash
pip install --force-reinstall "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

---

## Issue 3 — vLLM pip install overwrites torch with CUDA 13.0 build

**Error**
```
ImportError: /.../.venv2/.../torch/lib/libtorch_cuda.so: undefined symbol: ncclCommWindowDeregister
```

**Cause**  
`pip install vllm` pulled in `vllm 0.23.0`, which depends on `torch 2.11.0` (CUDA 13.0). This overwrote the cu124 torch. The `ncclCommWindowDeregister` symbol was added in NCCL 2.21+, which ships with the CUDA 13.0 build of torch but not with cu124.

**Fix**  
Install an older vLLM version whose C extensions were compiled against torch 2.5.x (CUDA 12.4):
```bash
pip uninstall vllm torch torchvision torchaudio -y
pip install "vllm==0.6.6.post1"
```

---

## Issue 4 — torchaudio still on CUDA 13.0 after torch reinstall

**Error**
```
OSError: libcudart.so.13: cannot open shared object file: No such file or directory
```

**Cause**  
When reinstalling torch and torchvision with `--force-reinstall`, torchaudio was not included in the command and remained on its CUDA 13.0 build. It was loaded transitively by `transformers` (via `loss_rnnt.py`) and failed immediately.

**Fix**  
Always reinstall all three torch packages together:
```bash
pip install --force-reinstall \
  "torch==2.5.1+cu124" \
  "torchvision==0.20.1+cu124" \
  "torchaudio==2.5.1+cu124" \
  --index-url https://download.pytorch.org/whl/cu124
```

---

## Issue 5 — torchvision version mismatch breaks transformers import

**Error**
```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'GPT2LMHeadModel'.
```

**Cause**  
`torchvision._meta_registrations` tried to register a fake implementation for `torchvision::nms`, but the operator was never registered because torchvision's C extension failed to load (CUDA version mismatch). This caused a cascade failure in transformers' lazy loader.

**Fix**  
Uninstalling torchvision entirely resolved the cascade. For a text-only model like GPT-2, torchvision is not needed:
```bash
pip uninstall torchvision -y
```

---

## Issue 6 — vLLM 0.6.6 pip: `all_special_tokens_extended` AttributeError

**Error**
```
AttributeError: GPT2Tokenizer has no attribute all_special_tokens_extended.
Did you mean: 'num_special_tokens_to_add'?
```

**Cause**  
vLLM 0.6.6's tokenizer utility calls `tokenizer.all_special_tokens_extended`, a property that was removed in a newer version of `transformers`. The `--force-reinstall transformers` command pulled in a version newer than vLLM 0.6.6 was designed for.

**Fix**  
Pin transformers to a version compatible with vLLM 0.6.6:
```bash
pip install "transformers==4.47.0"
```

---

## Issue 7 — `vllm/vllm-openai:latest` Docker image requires CUDA 13.0

**Error**
```
RuntimeError: The NVIDIA driver on your system is too old (found version 12060).
```

**Cause**  
`vllm/vllm-openai:latest` resolved to vLLM 0.23.0, which bundles a torch compiled for CUDA 13.0. The `:latest` tag always tracks the newest release.

**Fix**  
Pin to a specific image version built with CUDA 12.x. Update `servers/vllm_server/run.sh`:
```bash
docker run ... vllm/vllm-openai:v0.6.6.post1 ...
```

---

## Issue 8 — Triton config.pbtxt: `TYPE_BYTES` not recognized as a valid enum value

**Error**
```
[libprotobuf ERROR] Error parsing text-format inference.ModelConfig: 9:5:
Unknown enumeration value of "TYPE_BYTES" for field "data_type".
```

**Cause**  
Both `tritonserver:23.10-py3` and `tritonserver:24.08-py3` rejected `TYPE_BYTES` as an enum name in the protobuf text-format parser. The file had no CRLF issues and was correctly COPYed into the image. The root cause appears to be a change in how the bundled protobuf library resolves enum names in these builds.

**Fix**  
Use the numeric enum value `13` (which is `TYPE_BYTES = 13` in Triton's `model_config.proto`) instead of the string name in `config.pbtxt`:
```
# Before:
data_type: TYPE_BYTES

# After:
data_type: 13
```

---

## Issue 9 — Triton Dockerfile: `pip install torch` pulls CUDA 13.0 build

**Error**
```
RuntimeError: The NVIDIA driver on your system is too old (found version 12060).
```

**Cause**  
The Dockerfile ran `pip install "torch>=2.3.0"` without specifying a CUDA version or index URL. pip resolved this to the latest torch from PyPI, which was compiled for CUDA 13.0.

**Fix**  
Pin to a specific cu124 wheel in the Dockerfile:
```dockerfile
RUN pip install --no-cache-dir \
    "torch==2.5.1+cu124" \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

---

## Issue 10 — Triton model_repo mounted from Windows filesystem caused config parse failure

**Symptom**  
Even after fixing Issue 8 (using integer `13`), the `TYPE_BYTES` error persisted when the model_repo was mounted as a Docker volume from the Windows filesystem (`/mnt/c/...`).

**Cause**  
The 9P protocol used to expose Windows filesystem paths in WSL2 appears to affect how Docker reads files mounted from `/mnt/c`. Protobuf text parsing is sensitive to encoding and metadata.

**Fix**  
Copy the model_repo into the Docker image at build time instead of mounting it as a volume. Update the Dockerfile:
```dockerfile
COPY model_repo /models
```
And remove the `-v "$MODEL_REPO:/models"` volume mount from `run.sh`. This moves the files onto the Linux container filesystem, eliminating any Windows filesystem mount issues.
