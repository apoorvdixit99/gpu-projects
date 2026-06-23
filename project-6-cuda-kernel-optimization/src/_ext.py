"""Singleton loader for the CUDA extension.

The first call to get_ext() triggers JIT compilation via
torch.utils.cpp_extension.load().  Subsequent calls in the same process
return the cached module.  The compiled binary is cached in
torch's default build dir (~/.cache/torch_extensions/) so recompilation
only happens when a source file changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.utils.cpp_extension as cpp_ext

_EXT = None
_ROOT = Path(__file__).parent.parent


def get_ext():
    global _EXT
    if _EXT is None:
        sources = [
            str(_ROOT / "kernels" / "kernels.cpp"),
            str(_ROOT / "kernels" / "vector_add.cu"),
            str(_ROOT / "kernels" / "matmul.cu"),
            str(_ROOT / "kernels" / "reduction.cu"),
        ]
        print("Building CUDA extension (first run only) …", file=sys.stderr)
        _EXT = cpp_ext.load(
            name="cuda_kernels",
            sources=sources,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _EXT
