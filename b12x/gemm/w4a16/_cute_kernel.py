"""CuTe-DSL W4A16 dense GEMM kernel for SM120 / SM121 (v2, work-in-progress).

This is a minimal CuTe-DSL kernel that performs

    out[M, N] = (x[M, K] @ dequant(w_fp4[N, K/2], w_sf[N, K/16]).T) * alpha

with bf16 activations + bf16 output and FP4-packed weights with FP8
block scales (sf_vec_size = 16).

**Design choices for v1 simplicity:**

* No TMA, no async pipelining — plain ``cute.copy`` for gmem → smem.
* One CTA per ``(BLOCK_M, BLOCK_N)`` output tile.
* FP4 weight is loaded into smem as packed ``uint8``; each thread
  dequants its fragment to ``bf16`` in registers (multiply by the
  per-block FP32 scale and the scalar alpha).
* MMA: ``warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))`` —
  identical atom to the attention kernel.  Accumulator is ``fp32``,
  cast to ``bf16`` at the epilogue.

This will not match TRT-LLM's hand-tuned ``cuda_core_nvfp4_gemm`` perf;
that's a v3 optimization round (add TMA, pipeline stages, MMA-tile
autotune).  v2 goal is just to land a working CuTe-DSL backend so the
Triton kernel can be retired as the production path.

**Enable with** ``B12X_GEMM_W4A16_USE_CUTE=1``.  On compile / runtime
errors the public entry falls back to the Triton kernel — see
``micro.py``.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch

# CuTe DSL is heavyweight; import lazily so the package stays importable
# in CPU-only environments where the DSL isn't installed.
try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32
    from cutlass.cute.nvgpu import warp as _warp

    from b12x.cute.utils import (
        current_cuda_stream,
        make_ptr,
    )

    _CUTE_AVAILABLE = True
except ImportError:
    _CUTE_AVAILABLE = False


_BLOCK_M = 16
_BLOCK_N = 64
_BLOCK_K = 32
_NUM_WARPS = 4
_THREADS_PER_WARP = 32
_BLOCK_DIM = _NUM_WARPS * _THREADS_PER_WARP  # 128
_SF_VEC_SIZE = 16


def _cute_backend_enabled() -> bool:
    return os.environ.get("B12X_GEMM_W4A16_USE_CUTE") == "1" and _CUTE_AVAILABLE


# ---------------------------------------------------------------------------
# CuTe kernel
# ---------------------------------------------------------------------------


if _CUTE_AVAILABLE:

    class _DenseGemmW4A16CuteJit:
        """JIT-compiled CuTe-DSL kernel for one ``(BLOCK_M, BLOCK_N, BLOCK_K)`` config."""

        def __init__(self, m: int, n: int, k: int):
            self.m = m
            self.n = n
            self.k = k

            self.tiled_mma = cute.make_tiled_mma(
                _warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
                (_NUM_WARPS, 1, 1),
                permutation_mnk=(_NUM_WARPS * 16, 16, 16),
            )

        @cute.jit
        def kernel(
            self,
            x_gmem: cute.Tensor,      # [M, K] bf16
            w_gmem: cute.Tensor,      # [N, K // 2] uint8 (FP4 packed)
            sf_gmem: cute.Tensor,     # [N, K // 16] f32 (UNswizzled)
            alpha_gmem: cute.Tensor,  # [1] f32
            out_gmem: cute.Tensor,    # [M, N] bf16
        ):
            # ---------------------------------------------------------
            # WIP — kernel body skeleton.
            #
            # Implementation plan (each block is roughly one focused
            # session's worth of iterating on a live CuTe compiler):
            #
            # (1) Smem allocation.  Declare two shared-memory regions:
            #       sA : (BLOCK_M, BLOCK_K) bf16
            #       sB : (BLOCK_N, BLOCK_K) bf16 (the *dequantized* W)
            #     Use ``cute.make_smem_layout`` + ``cute.struct.Align``
            #     just like ``b12x/attention/contiguous/forward.py``.
            #
            # (2) Partition gmem -> smem copies.
            #     ``thr_copy_a = make_tiled_copy(...).get_slice(tid)``
            #     ``cute.copy(thr_copy_a, gA, sA)``  for A.
            #     For W (uint8) we load packed bytes; assign each
            #     thread a slab of ``(BLOCK_N // num_threads, BLOCK_K // 2)``
            #     ``uint8`` values.
            #
            # (3) Dequant.
            #     For each FP4 nibble in a thread's W slab: decode the
            #     magnitude (8-entry LUT chained via const_expr if/else)
            #     and sign, multiply by the corresponding ``sf_gmem``
            #     entry (broadcast across the 16-element FP4 group),
            #     multiply by the scalar alpha, store as bf16 into sB.
            #     Mirror ``_triton_kernel.py``'s ``_decode`` chain.
            #
            # (4) MMA accumulation.
            #     ``mma = self.tiled_mma.get_slice(tid)``
            #     ``tCrA = mma.partition_A(sA)``
            #     ``tCrB = mma.partition_B(sB)``
            #     ``tCrC = mma.make_fragment_C((BLOCK_M, BLOCK_N))``
            #     ``cute.gemm(mma, tCrA, tCrB, tCrC)``
            #
            # (5) Epilogue.
            #     Cast ``tCrC`` (fp32) -> bf16 register fragment.
            #     ``cute.copy(epi_copy, tCrC_bf16, gC_slice)`` to gmem.
            #
            # Until (1)-(5) are wired up, raise compile-time so the
            # outer driver can fall back to the Triton kernel.
            cutlass.const_expr(
                False,
                "CuTe-DSL W4A16 dense kernel body not yet implemented; "
                "v2 scaffold only.  Triton backend remains the default."
            )

        @cute.jit
        def __call__(
            self,
            x_ptr: cute.Pointer,
            w_ptr: cute.Pointer,
            sf_ptr: cute.Pointer,
            alpha_ptr: cute.Pointer,
            out_ptr: cute.Pointer,
            stream,
        ):
            x = cute.make_tensor(
                x_ptr,
                layout=cute.make_ordered_layout(
                    (self.m, self.k), order=(0, 1),
                ),
            )
            w = cute.make_tensor(
                w_ptr,
                layout=cute.make_ordered_layout(
                    (self.n, self.k // 2), order=(0, 1),
                ),
            )
            sf = cute.make_tensor(
                sf_ptr,
                layout=cute.make_ordered_layout(
                    (self.n, self.k // _SF_VEC_SIZE), order=(0, 1),
                ),
            )
            alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((1,)))
            out = cute.make_tensor(
                out_ptr,
                layout=cute.make_ordered_layout(
                    (self.m, self.n), order=(0, 1),
                ),
            )

            grid = (
                (self.m + _BLOCK_M - 1) // _BLOCK_M,
                (self.n + _BLOCK_N - 1) // _BLOCK_N,
                1,
            )
            self.kernel(x, w, sf, alpha, out).launch(
                grid=grid,
                block=(_BLOCK_DIM, 1, 1),
                smem=0,
                stream=stream,
            )

else:

    class _DenseGemmW4A16CuteJit:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("cutlass.cute (DSL) is not available in this environment")


# ---------------------------------------------------------------------------
# Public backend wrapper (caches compiled kernels)
# ---------------------------------------------------------------------------


class DenseGemmW4A16CuteKernel:
    """CuTe-DSL W4A16 dense GEMM backend (work-in-progress).

    The kernel body in ``_DenseGemmW4A16CuteJit`` is currently a
    scaffold — it raises a compile-time const_expr error to make the
    "not implemented" state loud and pushes the dispatch back to the
    Triton kernel via ``NotImplementedError``.  Once the body lands,
    flipping ``B12X_GEMM_W4A16_USE_CUTE=1`` will route through here
    automatically.
    """

    @classmethod
    def is_supported(cls, m: int, k: int, n: int) -> bool:
        from .micro import DenseGemmW4A16MicroKernel
        if not DenseGemmW4A16MicroKernel.is_supported(m, k, n):
            return False
        # Stricter constraints once the kernel-body is implemented.
        if m > _BLOCK_M:
            return False
        if n % _BLOCK_N != 0:
            return False
        if k % _BLOCK_K != 0:
            return False
        return True

    def __init__(self) -> None:
        self._compile_cache: dict[Tuple[int, int, int], _DenseGemmW4A16CuteJit] = {}

    def _get_compiled(self, m: int, n: int, k: int) -> _DenseGemmW4A16CuteJit:
        key = (m, n, k)
        if key not in self._compile_cache:
            self._compile_cache[key] = _DenseGemmW4A16CuteJit(m=m, n=n, k=k)
        return self._compile_cache[key]

    def __call__(
        self,
        x: torch.Tensor,
        w_fp4: torch.Tensor,
        w_blockscale_unswizzled_fp32: torch.Tensor,
        w_alpha: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not _CUTE_AVAILABLE:
            raise NotImplementedError("cutlass.cute (DSL) not available")
        m, k = x.shape
        n = w_fp4.shape[0]
        if out is None:
            out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)

        compiled = self._get_compiled(m, n, k)
        stream = current_cuda_stream()
        compiled(
            make_ptr(BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint8, w_fp4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Float32, w_blockscale_unswizzled_fp32.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Float32, w_alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            stream,
        )
        return out


__all__ = ["DenseGemmW4A16CuteKernel", "_cute_backend_enabled"]
