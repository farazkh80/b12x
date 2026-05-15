"""W4A16 dense GEMM micro kernel for SM120 / SM121.

Decode-only (M ≤ 32).  Consumes bf16 activations and FP4-packed weights
directly, skipping the bf16->FP4 activation quantize step that the
TRT-LLM ``cuda_core_nvfp4_gemm`` baseline pays per call.

**v1 implementation:** Triton kernel (``_triton_kernel.py``).  A
higher-perf CuTe-DSL variant is tracked for v2 — the existing dense
W4A4 kernel in ``b12x.gemm.dense`` would need bf16-stage A-operand
plumbing lifted from the W4A16 MoE micro kernel, which is a
multi-day cute_dsl surgery.

Set ``B12X_GEMM_W4A16_FORCE_REFERENCE=1`` to fall back to the Python
reference (useful for accuracy debugging — runs on CPU).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from b12x.moe.fused.w4a16.reference import unswizzle_block_scale

from ._cute_kernel import DenseGemmW4A16CuteKernel, _cute_backend_enabled
from ._triton_kernel import w4a16_dense_decode_triton
from .reference import dense_reference_w4a16

# Same M ladder as the W4A16 MoE micro kernel.
_SUPPORTED_M = (1, 2, 4, 8, 10, 12, 16, 24, 32)
# K must be a multiple of 128 (covers the Nano3.5 dense linears: 2688,
# 3712, 4096).  Note: the MoE micro insists on K % 512 == 0 because
# 2688-K MoE matmuls route to the static backend; for dense we want
# 2688 supported directly, so we only require the FP4-block / MMA tile
# divisibility.
_K_BLOCK = 128


class DenseGemmW4A16MicroKernel:
    """W4A16 dense GEMM kernel (decode-only, M ≤ 32)."""

    @classmethod
    def is_supported(cls, m: int, k: int, n: int) -> bool:
        if m not in _SUPPORTED_M:
            return False
        if k <= 0 or k % _K_BLOCK != 0:
            return False
        if n <= 0 or n % 16 != 0:
            return False
        return True

    def __init__(self) -> None:
        # Lazily-constructed CuTe-DSL backend.  Used only when
        # ``B12X_GEMM_W4A16_USE_CUTE=1`` and the kernel-body lift in
        # ``_cute_kernel.py`` has landed; otherwise we always use the
        # Triton path.
        self._cute: DenseGemmW4A16CuteKernel | None = None

    def __call__(
        self,
        x: torch.Tensor,
        w_fp4: torch.Tensor,
        w_blockscale: torch.Tensor,
        w_alpha: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m, k = x.shape
        n = w_fp4.shape[0]
        if out is None:
            out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)

        # Optional CuTe-DSL backend (env-gated, currently a scaffold —
        # see ``_cute_kernel.py``).  Falls through to Triton on
        # NotImplementedError so a missing body is non-fatal.
        if x.is_cuda and _cute_backend_enabled():
            if self._cute is None:
                self._cute = DenseGemmW4A16CuteKernel()
            try:
                return self._cute(x, w_fp4, w_blockscale, w_alpha, out=out)
            except NotImplementedError:
                pass

        # CPU shuttle is unsupported by the Triton kernel.  Fall back
        # to the reference path if asked to run on CPU.
        if not x.is_cuda:
            out_cpu = dense_reference_w4a16(
                x, w_fp4=w_fp4, w_blockscale=w_blockscale, w_alpha=w_alpha,
            )
            out.copy_(out_cpu)
            return out

        # Unswizzle the FP8 block scales -> [N, K // 16] fp32 once per
        # call.  This is cheap relative to the matmul and lets the
        # Triton kernel avoid the swizzle indexing.  v2 will read
        # swizzled scales directly inside a CuTe-DSL kernel.
        sf_fp32 = unswizzle_block_scale(
            w_blockscale, rows=n, cols_blocks=k // 16,
        ).contiguous()

        return w4a16_dense_decode_triton(
            x.contiguous(),
            w_fp4.contiguous(),
            sf_fp32,
            w_alpha.contiguous(),
            out=out,
        )


_KERNEL_CACHE: Optional[DenseGemmW4A16MicroKernel] = None


def _get_cached_kernel() -> DenseGemmW4A16MicroKernel:
    global _KERNEL_CACHE
    if _KERNEL_CACHE is None:
        _KERNEL_CACHE = DenseGemmW4A16MicroKernel()
    return _KERNEL_CACHE


def dense_gemm_w4a16(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale: torch.Tensor,
    w_alpha: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Public entry point — decode-only W4A16 dense GEMM."""
    if os.environ.get("B12X_GEMM_W4A16_FORCE_REFERENCE") == "1":
        result = dense_reference_w4a16(
            x.detach().cpu(),
            w_fp4=w_fp4.detach().cpu(),
            w_blockscale=w_blockscale.detach().cpu(),
            w_alpha=w_alpha.detach().cpu(),
        ).to(x.device)
        if out is None:
            return result
        out.copy_(result)
        return out

    m, k = x.shape
    n = w_fp4.shape[0]
    if not DenseGemmW4A16MicroKernel.is_supported(m, k, n):
        raise NotImplementedError(
            f"dense_gemm_w4a16 v1 supports M in {_SUPPORTED_M}, "
            f"K % {_K_BLOCK} == 0, N % 16 == 0; "
            f"got M={m}, K={k}, N={n}."
        )
    kernel = _get_cached_kernel()
    return kernel(x, w_fp4, w_blockscale, w_alpha, out=out)


__all__ = ["DenseGemmW4A16MicroKernel", "dense_gemm_w4a16"]
