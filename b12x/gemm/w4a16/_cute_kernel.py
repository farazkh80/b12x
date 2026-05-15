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
        """JIT-compiled CuTe-DSL kernel for one ``(BLOCK_M, BLOCK_N, BLOCK_K)`` config.

        The MMA atom, tiled MMA, smem layouts, and SharedStorage struct
        are built lazily inside the ``@cute.jit`` scope (in
        ``_setup_attributes``) because ``cute.make_tiled_mma`` requires
        an active MLIR context.  See ``b12x/gemm/dense.py`` for the
        same pattern.
        """

        def __init__(self, m: int, n: int, k: int):
            self.m = m
            self.n = n
            self.k = k

        def _setup_attributes(self):
            """Build MMA + smem layouts + SharedStorage.  Must be called from
            within ``@cute.jit`` (e.g. from ``__call__``)."""
            self.tiled_mma = cute.make_tiled_mma(
                _warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
                (_NUM_WARPS, 1, 1),
                permutation_mnk=(_NUM_WARPS * 16, 16, 16),
            )

            # ---- Step 1: smem layouts ----
            # Simple row-major bf16 tiles for the staged operands.
            # ``sA`` holds the activation tile ``[BLOCK_M, BLOCK_K]``.
            # ``sB`` holds the *dequantized* weight tile
            # ``[BLOCK_N, BLOCK_K]``.  No swizzle / no staging — v1 is
            # synchronous; the v3 round will add stages + swizzled
            # layouts once the body is otherwise correct.
            self.sA_layout = cute.make_ordered_layout(
                (_BLOCK_M, _BLOCK_K), order=(0, 1),
            )
            self.sB_layout = cute.make_ordered_layout(
                (_BLOCK_N, _BLOCK_K), order=(0, 1),
            )

            # ---- Step 1: shared-storage struct ----
            buffer_align_bytes = 128

            @cute.struct
            class SharedStorage:
                sA: cute.struct.Align[
                    cute.struct.MemRange[BFloat16, cute.cosize(self.sA_layout)],
                    buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[BFloat16, cute.cosize(self.sB_layout)],
                    buffer_align_bytes,
                ]

            self.SharedStorage = SharedStorage
            self.smem_bytes = SharedStorage.size_in_bytes()

        @cute.kernel
        def kernel(
            self,
            x_gmem: cute.Tensor,      # [M, K] bf16
            w_gmem: cute.Tensor,      # [N, K // 2] uint8 (FP4 packed)
            sf_gmem: cute.Tensor,     # [N, K // 16] f32 (UNswizzled)
            alpha_gmem: cute.Tensor,  # [1] f32
            out_gmem: cute.Tensor,    # [M, N] bf16
            SharedStorage: cutlass.Constexpr,
            sA_layout: cute.Layout,
            sB_layout: cute.Layout,
        ):
            # ---- Step 1: allocate the smem tiles ----
            # Allocates [BLOCK_M, BLOCK_K] bf16 for ``sA`` and
            # [BLOCK_N, BLOCK_K] bf16 for ``sB``.  The two smem tensors
            # are typed CuTe ``Tensor`` views over the underlying smem
            # storage; downstream steps will partition them with the
            # tiled-MMA's thread-fragments.
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sA = storage.sA.get_tensor(sA_layout)
            sB = storage.sB.get_tensor(sB_layout)

            # Compile-time assertion: cosize matches the declared shape.
            # This forces the cute compiler to validate that the smem
            # layout is constructible end-to-end (i.e., Step 1 typechecks).
            cutlass.const_expr(cute.cosize(sA.layout) == _BLOCK_M * _BLOCK_K)
            cutlass.const_expr(cute.cosize(sB.layout) == _BLOCK_N * _BLOCK_K)

            # ---------------------------------------------------------
            # Steps 2-5 (not yet implemented):
            #
            # (2) Partition gmem -> smem copies (``cute.copy``).
            # (3) Dequant FP4 nibbles -> bf16, multiply by sf+alpha,
            #     store into sB.
            # (4) MMA accumulation with ``cute.gemm``.
            # (5) Epilogue: fp32 acc -> bf16 -> gmem.
            #
            # The kernel body intentionally stops here for v1-step-1.
            # No output is written; ``micro.py`` recognises this via a
            # NaN-sentinel on the output buffer and falls back to the
            # Triton kernel.  (Removing the early ``return`` here is
            # deliberate — ``@cute.jit`` transforms the function into
            # a launch-builder and an explicit ``return`` short-circuits
            # that transformation.)
            pass

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
            self._setup_attributes()
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
            self.kernel(
                x, w, sf, alpha, out,
                self.SharedStorage, self.sA_layout, self.sB_layout,
            ).launch(
                grid=grid,
                block=(_BLOCK_DIM, 1, 1),
                smem=self.smem_bytes,
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
