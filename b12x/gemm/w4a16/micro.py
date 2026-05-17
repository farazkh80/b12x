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

# v5's (tile_K, n_per_cta) selection is shape-driven.  Autotune sweep
# at M ∈ {1024, 2048, 4096} on the Nano3.5 dense linears (see
# ``scripts/tune_prefill_v5.py``) tuned to **Spark (NVIDIA GB10, SM121,
# 48 SMs)** — the production target.  Rules:
#
#   K ≤ 3712 (q_proj, k_proj, shared.up, shared.dn, mamba_in_proj):
#       tile_K=32, n_per_cta=1
#       — 8-15% win over baseline (tile_K=64, n=1) across all M.
#         Smaller K means fewer total K-tiles; tile_K=32 doubles
#         K-tile count for finer-grain pipeline overlap and better
#         SM saturation on Spark's small (48) SM count.  n=2 doesn't
#         help at tile_K=32 here — the kernel is already saturated.
#
#   K ≥ 4096 (o_proj, mamba_output_proj):
#       tile_K=64, n_per_cta=2 (if eligible, else 1)
#       — 8% win.  Larger K already has enough K-tiles at the coarse
#         setting; smaller tile_K adds overhead without payoff.
#         A-reuse via n_per_cta=2 dominates the speedup.
#
# n_per_cta=2 eligibility (only applies when tile_K=64):
#   * M ≥ _DEFAULT_N_PER_CTA2_M (enough M to amortize)
#   * num_n_tiles % 2 == 0      (kernel constraint — no half-pair support)
#   * num_n_tiles ≥ _N_PER_CTA2_NT_MIN (preserve grid parallelism)
#
# Local-vs-Spark divergence: on the dev SM120 box (RTX PRO 6000
# Blackwell, ~144 SMs), shared.dn picks (32, 2) and q_proj M=2048
# picks (64, 2).  We optimize for Spark since (a) it's the production
# target and (b) the Marlin baselines we benchmark against were
# captured there.  The Spark rule is ≤9% suboptimal vs the local-best
# on local hardware (q_proj M=2048 only).
#
# ab_stage stays at 2 on both: ab_stage=3 was never a win (the kernel
# isn't pipeline-stage-starved at smem-fitting depths).
_DEFAULT_N_PER_CTA2_M = int(os.environ.get("B12X_GEMM_W4A16_N_PER_CTA2_M", "1024"))
_N_PER_CTA2_NT_MIN = int(os.environ.get("B12X_GEMM_W4A16_N_PER_CTA2_NT_MIN", "8"))
_TILE_K_SMALL = 32   # for K ≤ _TILE_K_SMALL_K_MAX
_TILE_K_LARGE = 64   # default
_TILE_K_SMALL_K_MAX = int(os.environ.get("B12X_GEMM_W4A16_TILE_K_SMALL_K_MAX", "3712"))


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
        # Cache prefill instances keyed by (tile_k, n_per_cta).
        # Picked per-call based on (K, N, M); see ``_pick_prefill_cfg``.
        self._prefill_cache: dict = {}

    def _pick_prefill_cfg(self, m: int, k: int, n: int):
        """Return (tile_k, n_per_cta) — autotune-derived dispatch.

        See ``B12X_GEMM_W4A16_TILE_K_SMALL_K_MAX`` etc. for env knobs.
        """
        tile_n = DenseGemmW4A16CutePrefillKernel._TILE_N
        num_n_tiles = n // tile_n

        # K-driven tile_K + n_per_cta selection (Spark-tuned; see the
        # block comment above).
        if k <= _TILE_K_SMALL_K_MAX:
            # Small-to-mid K: fine K-tile, no A-reuse.  n_per_cta=1
            # wins for every shape with K ≤ 3712 on Spark.
            return _TILE_K_SMALL, 1

        # Large K (≥ 4096): coarse K-tile + A-reuse.
        eligible_n2 = (
            m >= _DEFAULT_N_PER_CTA2_M
            and num_n_tiles % 2 == 0
            and num_n_tiles >= _N_PER_CTA2_NT_MIN
        )
        return _TILE_K_LARGE, (2 if eligible_n2 else 1)

    def _pick_prefill(self, m: int, k: int, n: int) -> DenseGemmW4A16CutePrefillKernel:
        cfg = self._pick_prefill_cfg(m, k, n)
        if cfg not in self._prefill_cache:
            tile_k, n_per_cta = cfg
            self._prefill_cache[cfg] = DenseGemmW4A16CutePrefillKernel(
                tile_k=tile_k, n_per_cta=n_per_cta,
            )
        return self._prefill_cache[cfg]

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
            return self._pick_prefill(m, k, n)(
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
