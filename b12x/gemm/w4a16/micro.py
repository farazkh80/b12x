"""W4A16 dense GEMM micro kernel for SM120 / SM121.

Two-backend dispatch:

* **Decode** (M ≤ ``_DECODE_M_MAX``): v4 forked CuTe-DSL kernel
  (``_cute_dense_kernel.py``), warp-level bf16 MMA, tile (32, 64, 64),
  4 MMA warps.  Sweet spot for the Nano3.5 decode linears.
* **Prefill** (M > ``_DECODE_M_MAX``): v5 scaled-up CuTe-DSL kernel
  (``_cute_prefill_kernel.py``), same MMA primitive but tile
  (128, 64, 64), 8 MMA warps — absorbs 4× more M-rows per A K-tile
  load, dramatically reducing per-call overhead at M ≥ 64.

Both backends share weight layout and accuracy gates.  Crossover M is
exposed via ``B12X_GEMM_W4A16_PREFILL_M`` so we can sweep it in the
benchmark (default 33: anything above v4's per-CTA M cap).

Set ``B12X_GEMM_W4A16_FORCE_REFERENCE=1`` to fall back to the Python
reference (useful for accuracy debugging — runs on CPU).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from ._cute_dense_kernel import DenseGemmW4A16CuteDenseKernel
from ._cute_prefill_kernel import DenseGemmW4A16CutePrefillKernel
from .reference import dense_reference_w4a16


_DECODE_M_MAX = 32   # v4 was tuned for M ∈ [1, 32] but can run at any M
# Crossover threshold empirically: v4 wins at M ≤ 128 (warp-level 32×64
# tile keeps work granular), v5 takes over at M ≥ 256 (8 MMA warps +
# 128×64 tile amortizes per-CTA overhead).  At M=128 they're roughly
# equal — default to v5 from 256 onward to be safe; override via env.
_DEFAULT_PREFILL_M = int(os.environ.get("B12X_GEMM_W4A16_PREFILL_M", "256"))


def _use_prefill(m: int) -> bool:
    return m >= _DEFAULT_PREFILL_M


class DenseGemmW4A16MicroKernel:
    """W4A16 dense GEMM kernel with decode + prefill dispatch."""

    @classmethod
    def is_supported(cls, m: int, k: int, n: int) -> bool:
        # Either backend must accept the shape.
        if _use_prefill(m):
            return DenseGemmW4A16CutePrefillKernel.is_supported(m, k, n)
        return DenseGemmW4A16CuteDenseKernel.is_supported(m, k, n)

    def __init__(self) -> None:
        self._decode: DenseGemmW4A16CuteDenseKernel | None = None
        self._prefill: DenseGemmW4A16CutePrefillKernel | None = None

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

        if not x.is_cuda:
            out_cpu = dense_reference_w4a16(
                x, w_fp4=w_fp4, w_blockscale=w_blockscale, w_alpha=w_alpha,
            )
            out.copy_(out_cpu)
            return out

        if _use_prefill(m):
            if self._prefill is None:
                self._prefill = DenseGemmW4A16CutePrefillKernel()
            return self._prefill(
                x.contiguous(), w_fp4.contiguous(), w_blockscale.contiguous(),
                w_alpha.contiguous(), out=out,
            )
        if self._decode is None:
            self._decode = DenseGemmW4A16CuteDenseKernel()
        return self._decode(
            x.contiguous(), w_fp4.contiguous(), w_blockscale.contiguous(),
            w_alpha.contiguous(), out=out,
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
    """Public entry point — W4A16 dense GEMM, decode + prefill backends.

    Dispatch picks the prefill kernel when M ≥
    ``B12X_GEMM_W4A16_PREFILL_M`` (default 33), otherwise the decode
    kernel.  ``w_blockscale`` must be the swizzled FP8 e4m3 tensor (as
    produced by ``quantize_dense_weight_to_fp4``).
    """
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
            f"dense_gemm_w4a16 supports N % 64 == 0 and K % 64 == 0; "
            f"got M={m}, K={k}, N={n}."
        )
    kernel = _get_cached_kernel()
    return kernel(x, w_fp4, w_blockscale, w_alpha, out=out)


__all__ = ["DenseGemmW4A16MicroKernel", "dense_gemm_w4a16"]
