from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_dense_gemm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    out_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, k, BLOCK_K):
        k_idxs = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * k + k_idxs[None, :],
            mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[None, :] * k + k_idxs[:, None],
            mask=(offs_n[None, :] < n) & (k_idxs[:, None] < k),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    scale = tl.load(scale_a_ptr) * tl.load(scale_b_ptr)
    out = acc * scale
    tl.store(out_ptr + offs_m[:, None] * n + offs_n[None, :], out, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def _validate_inputs(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> None:
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


def fp8_dense_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    _validate_inputs(a, b, scale_a, scale_b)
    m = int(a.shape[0])
    k = int(a.shape[1])
    n = int(b.shape[0])
    out = _prepare_output(a, m, n, out)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fp8_dense_gemm_kernel[grid](
        a,
        b,
        scale_a.reshape(1).contiguous(),
        scale_b.reshape(1).contiguous(),
        out,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["fp8_dense_gemm_triton"]
