from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load


def _validate_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> None:
    if a.dtype is not torch.float8_e4m3fn or b.dtype is not torch.float8_e4m3fn:
        raise TypeError("a and b must be torch.float8_e4m3fn")
    if scale_a.dtype is not torch.float32 or scale_b.dtype is not torch.float32:
        raise TypeError("scale_a and scale_b must be torch.float32")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2 tensors")
    if scale_a.numel() != 1 or scale_b.numel() != 1:
        raise ValueError("only scalar scale tensors are supported")
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have the same K dimension")
    if a.shape[0] % 16 != 0 or b.shape[0] % 32 != 0 or a.shape[1] % 32 != 0:
        raise ValueError("M must be divisible by 16, N by 32, and K by 32")
    if not a.is_cuda or not b.is_cuda or not scale_a.is_cuda or not scale_b.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")


def _prepare_output(a: torch.Tensor, m: int, n: int, out: Optional[torch.Tensor]) -> torch.Tensor:
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if out.shape != (m, n) or out.dtype is not torch.bfloat16 or not out.is_cuda:
        raise ValueError("out must be a CUDA bfloat16 tensor with shape (M, N)")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("fp8_dense_sm120_cute_ext.cu")
    verbose = os.getenv("B12X_FP8_DENSE_SM120_CUTE_VERBOSE_BUILD", "0") == "1"
    build_directory = Path(os.getenv("B12X_FP8_DENSE_SM120_CUTE_BUILD_DIR", "/tmp/b12x_fp8_dense_sm120_cute_ext"))
    cutlass_include = os.getenv(
        "B12X_CUTLASS_INCLUDE_DIR",
        "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/deep_gemm/include",
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="b12x_fp8_dense_sm120_cute_ext",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_include_paths=[cutlass_include],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "-DCUTLASS_ARCH_MMA_SM120_ENABLED",
            "-DCUTLASS_ARCH_MMA_SM121_ENABLED",
            "-gencode=arch=compute_121a,code=sm_121a",
        ],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def fp8_dense_gemm_sm120_cute(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_inputs(a, b, scale_a, scale_b)
    m = int(a.shape[0])
    n = int(b.shape[0])
    out = _prepare_output(a, m, n, out)
    _load_extension().fp8_dense_gemm(a, b, scale_a.reshape(1).contiguous(), scale_b.reshape(1).contiguous(), out)
    return out


__all__ = ["fp8_dense_gemm_sm120_cute"]
