"""W4A16 dense GEMM micro kernel for SM120 / SM121.

Three-tier dispatch:

* **Decode** (M < ``B12X_GEMM_W4A16_PREFILL_M``, default 256):
  v4 forked CuTe-DSL kernel (``_cute_dense_kernel.py``), warp-level
  bf16 MMA, tile (32, 64, 64), 4 MMA warps.  Sweet spot for the
  Nano3.5 decode linears (M ≤ 32 originally; v4 still runs correctly
  at any M).
* **Prefill / n_per_cta=1**
  (``B12X_GEMM_W4A16_PREFILL_M`` ≤ M < ``B12X_GEMM_W4A16_N_PER_CTA2_M``,
  default 256 ≤ M < 1024): v5 scaled-up CuTe-DSL kernel
  (``_cute_prefill_kernel.py``), tile (128, 64, 64), 8 MMA warps.
  General fallback — works at any (N, K) within the envelope.
* **Prefill / n_per_cta=2** (M ≥ ``B12X_GEMM_W4A16_N_PER_CTA2_M``,
  default 1024, and ``num_n_tiles % 2 == 0``): v5 with A-reuse across
  2 consecutive N-tiles per CTA — cuts A TMA traffic in half, wins
  ~25-30% at large M.  Falls back to n_per_cta=1 when N isn't a
  multiple of 128 (e.g. ``mamba_in_proj`` with N=10304).

Both backends share weight layout and accuracy gates.

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

# v5's n_per_cta=2 path reuses rA across 2 consecutive N-tiles in one
# work-pair — cuts A traffic in half, wins ~25-30% at M ≥ 1024 on
# wide-N shapes (o_proj, shared.dn, mamba_out at N=2688).  Loses when
# N is small (k_proj N=256: ~2× slower) because pairing two N tiles
# halves parallelism but the kernel is already SM-saturation bound
# there, not A-traffic bound.  Conditions to engage:
#   1. M ≥ _DEFAULT_N_PER_CTA2_M     (enough M to amortize)
#   2. num_n_tiles ≥ _N_PER_CTA2_NT  (preserve grid parallelism)
#   3. num_n_tiles % 2 == 0          (kernel constraint)
#
# mamba_in_proj (K=2688, N=10304) has num_n_tiles=161 (odd) so it
# can't engage condition 3.  Investigated splitting into two launches
# (n_per_cta=2 over an even prefix + n_per_cta=1 tail): kernel-only
# win is just ~3.7% at this shape because the bottleneck is B traffic
# (10.4 MB FP4 weight) and cooperative FP4-decode staging, not A
# reuse.  The two-launch overhead (alloc + copy-back) wipes that
# delta out and goes net negative.  Conclusion: leave mamba_in_proj
# on n_per_cta=1; the 1.20× v5/marlin ratio at M=2048 on this shape
# reflects the kernel's structural memory-traffic profile, not a
# dispatch oversight.
_DEFAULT_N_PER_CTA2_M = int(os.environ.get("B12X_GEMM_W4A16_N_PER_CTA2_M", "1024"))
_N_PER_CTA2_NT_MIN = int(os.environ.get("B12X_GEMM_W4A16_N_PER_CTA2_NT_MIN", "8"))


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
        # Two prefill instances: n_per_cta=1 (general) and n_per_cta=2
        # (A-reuse fast path).  Pick by M + N divisibility at call time.
        self._prefill_n1: DenseGemmW4A16CutePrefillKernel | None = None
        self._prefill_n2: DenseGemmW4A16CutePrefillKernel | None = None

    def _pick_prefill(self, m: int, n: int) -> DenseGemmW4A16CutePrefillKernel:
        tile_n = DenseGemmW4A16CutePrefillKernel._TILE_N
        num_n_tiles = n // tile_n
        use_n2 = (
            m >= _DEFAULT_N_PER_CTA2_M
            and num_n_tiles % 2 == 0
            and num_n_tiles >= _N_PER_CTA2_NT_MIN
        )
        if use_n2:
            if self._prefill_n2 is None:
                self._prefill_n2 = DenseGemmW4A16CutePrefillKernel(n_per_cta=2)
            return self._prefill_n2
        if self._prefill_n1 is None:
            self._prefill_n1 = DenseGemmW4A16CutePrefillKernel(n_per_cta=1)
        return self._prefill_n1

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
            return self._pick_prefill(m, n)(
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
