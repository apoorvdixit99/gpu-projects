"""Build a TensorRT FP16 engine from the exported ONNX model."""

from pathlib import Path

import tensorrt as trt
import torch

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

_LOGGER = trt.Logger(trt.Logger.WARNING)


def build(
    onnx_path: str | None = None,
    engine_path: str | None = None,
    fp16: bool = True,
    workspace_gb: int = 4,
    batch_sizes: list[int] | None = None,
    seq_lens: list[int] | None = None,
) -> str:
    """Build and serialize a TRT engine. Returns path to the .trt file.

    Optimization profile covers:
      min  = (min_batch, min_seq)
      opt  = (mid_batch, mid_seq)   ← TRT uses this for kernel tuning
      max  = (max_batch, max_seq)
    """
    batch_sizes = batch_sizes or [1, 4, 8, 16]
    seq_lens = seq_lens or [64, 128, 256]

    if onnx_path is None:
        # Use the ONNX file that matches the requested precision so TRT reads
        # native-precision weights rather than converting at build time.
        onnx_tag = "fp16" if fp16 else "fp32"
        onnx_path = str(MODELS_DIR / f"gpt2_{onnx_tag}.onnx")
    if engine_path is None:
        tag = "fp16" if fp16 else "fp32"
        engine_path = str(MODELS_DIR / f"gpt2_{tag}.trt")

    if not Path(onnx_path).exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. Run export_onnx.py first."
        )

    min_b, max_b = min(batch_sizes), max(batch_sizes)
    min_s, max_s = min(seq_lens), max(seq_lens)
    opt_b = batch_sizes[len(batch_sizes) // 2]
    opt_s = seq_lens[len(seq_lens) // 2]

    # TRT's builder calls ensureCudaInitialized internally; if no prior CUDA op
    # has run in this process it fails with error 35. Touching a tensor first
    # guarantees the CUDA context is live before the builder is created.
    torch.cuda.init()

    builder = trt.Builder(_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, _LOGGER)
    config = builder.create_builder_config()

    onnx_size_mb = Path(onnx_path).stat().st_size / 1024 ** 2
    print(f"Parsing ONNX ({onnx_size_mb:.1f} MB) …")
    # parse_from_file resolves external weight files (.onnx.data) relative to
    # the ONNX path; parse(raw_bytes) has no path context and fails to find them.
    if not parser.parse_from_file(onnx_path):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    profile = builder.create_optimization_profile()
    for tensor_name in ("input_ids", "attention_mask"):
        profile.set_shape(tensor_name, (min_b, min_s), (opt_b, opt_s), (max_b, max_s))
    config.add_optimization_profile(profile)

    print("Building TRT engine — this may take several minutes …")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "TRT engine build failed. Check GPU memory and ONNX model validity."
        )

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    engine_bytes = bytes(serialized)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    size_mb = len(engine_bytes) / 1024 ** 2
    print(f"TRT engine saved — {size_mb:.1f} MB at {engine_path}")
    return engine_path


if __name__ == "__main__":
    build()
