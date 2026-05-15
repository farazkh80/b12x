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
    from cutlass import BFloat16, Float32, Int32, Uint8
    from cutlass.cute.nvgpu import cpasync, warp as _warp

    from b12x.cute.utils import (
        current_cuda_stream,
        make_ptr,
    )

    _CUTE_AVAILABLE = True
except ImportError:
    _CUTE_AVAILABLE = False


_BLOCK_M = 16
_BLOCK_N = 32   # matches the tiled-MMA permutation (16, 32, 16) below
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
            # 1 warp along M (BLOCK_M=16), 4 warps along N (each warp
            # handles 8 N rows via the (16, 8, 16) atom -> total
            # 4*8 = 32 = BLOCK_N).  Total tile per MMA issue is
            # (16, 32, 16); the K-loop inside this CTA iterates
            # BLOCK_K / 16 = 2 times to consume the staged sA/sB tile.
            self.tiled_mma = cute.make_tiled_mma(
                _warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
                (1, _NUM_WARPS, 1),
                permutation_mnk=(_BLOCK_M, _BLOCK_N, 16),
            )

            # ---- Step 1: smem layouts ----
            # Simple row-major bf16 tiles for the staged operands.
            # ``sA`` holds the activation tile ``[BLOCK_M, BLOCK_K]``.
            # ``sB`` holds the *dequantized* weight tile
            # ``[BLOCK_N, BLOCK_K]``.  No swizzle / no staging — v1 is
            # synchronous; the v3 round will add stages + swizzled
            # layouts once the body is otherwise correct.
            # Row-major (K stride 1) smem layouts; mirrors the gmem
            # layout of x and w_fp4 and gives natural K-contiguous
            # access for the dequant inner loop.
            self.sA_layout = cute.make_ordered_layout(
                (_BLOCK_M, _BLOCK_K), order=(1, 0),
            )
            self.sB_layout = cute.make_ordered_layout(
                (_BLOCK_N, _BLOCK_K), order=(1, 0),
            )
            self.sWpacked_layout = cute.make_ordered_layout(
                (_BLOCK_N, _BLOCK_K // 2), order=(1, 0),
            )
            # Epilogue smem tile (Step 5): (BLOCK_M, BLOCK_N) bf16,
            # row-major. Sized to the output tile this CTA produces.
            self.sC_layout = cute.make_ordered_layout(
                (_BLOCK_M, _BLOCK_N), order=(1, 0),
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
                sWpacked: cute.struct.Align[
                    cute.struct.MemRange[Uint8, cute.cosize(self.sWpacked_layout)],
                    buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[BFloat16, cute.cosize(self.sC_layout)],
                    buffer_align_bytes,
                ]

            self.SharedStorage = SharedStorage
            self.smem_bytes = SharedStorage.size_in_bytes()

            # ---- Step 2 (v3.1): gmem -> smem tiled copies via uint32 view ----
            # The kernel body recasts bf16 / uint8 source tensors to
            # Uint32 so cute sees wider elements (4 bytes each), then
            # 128-bit copies = 4 uint32 = 16 bytes per copy.  All Nano35
            # K dims are 16-byte aligned and our host tensors are
            # ``.contiguous()``-shuttled to row-major, so the recast is
            # always safe.
            atom_copy_u32_128 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Uint32,
                num_bits_per_copy=128,
            )

            # A (bf16 view): (BLOCK_M, BLOCK_K)   -> 512 bf16
            # A (u32 view) : (BLOCK_M, BLOCK_K/2) -> 256 uint32
            # 64 threads x 4 uint32 = 256 elems = 16 bytes/thread = 128 bits ✓
            self.tA_layout = cute.make_ordered_layout(
                (_BLOCK_M, (_BLOCK_K // 2) // 4), order=(1, 0),
            )  # (16, 4) = 64 threads
            self.vA_layout = cute.make_layout((1, 4))
            self.gmem_tiled_copy_A = cute.make_tiled_copy_tv(
                atom_copy_u32_128, self.tA_layout, self.vA_layout,
            )

            # W (u8 view):  (BLOCK_N, BLOCK_K/2)   -> 512 uint8
            # W (u32 view): (BLOCK_N, BLOCK_K/8)   -> 128 uint32
            # 32 threads x 4 uint32 = 128 elems = 16 bytes/thread = 128 bits ✓
            self.tW_layout = cute.make_ordered_layout(
                (_BLOCK_N, (_BLOCK_K // 8) // 4), order=(1, 0),
            )  # (32, 1) = 32 threads
            self.vW_layout = cute.make_layout((1, 4))
            self.gmem_tiled_copy_W = cute.make_tiled_copy_tv(
                atom_copy_u32_128, self.tW_layout, self.vW_layout,
            )

            # Epilogue smem -> gmem tiled copy (Step 5, v3.1).
            # sC (bf16 view): (BLOCK_M, BLOCK_N)   -> 512 bf16
            # sC (u32 view) : (BLOCK_M, BLOCK_N/2) -> 256 uint32
            # 64 threads x 4 uint32 = 16 bytes/thread = 128 bits ✓
            self.tC_layout = cute.make_ordered_layout(
                (_BLOCK_M, (_BLOCK_N // 2) // 4), order=(1, 0),
            )  # (16, 4) = 64 threads
            self.vC_layout = cute.make_layout((1, 4))
            self.gmem_tiled_copy_C = cute.make_tiled_copy_tv(
                atom_copy_u32_128, self.tC_layout, self.vC_layout,
            )

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
            sWpacked_layout: cute.Layout,
            sC_layout: cute.Layout,
            gmem_tiled_copy_A: cute.TiledCopy,
            gmem_tiled_copy_W: cute.TiledCopy,
            gmem_tiled_copy_C: cute.TiledCopy,
            tiled_mma: cute.TiledMma,
        ):
            # ---- Step 1: allocate the smem tiles ----
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sA = storage.sA.get_tensor(sA_layout)
            sB_bf16 = storage.sB.get_tensor(sB_layout)
            sWpacked = storage.sWpacked.get_tensor(sWpacked_layout)
            sC = storage.sC.get_tensor(sC_layout)
            cutlass.const_expr(cute.cosize(sA.layout) == _BLOCK_M * _BLOCK_K)
            cutlass.const_expr(cute.cosize(sB_bf16.layout) == _BLOCK_N * _BLOCK_K)
            cutlass.const_expr(cute.cosize(sWpacked.layout) == _BLOCK_N * (_BLOCK_K // 2))
            cutlass.const_expr(cute.cosize(sC.layout) == _BLOCK_M * _BLOCK_N)

            # Identify this CTA's output tile.
            block_m, block_n, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()

            # ---- Setup: slice gmem to this CTA's tile of A, W, and SF ----
            # gmem copies operate on a Uint32 view of the operands so
            # cute can issue 128-bit (4-uint32) loads.  sA / sWpacked
            # are likewise recast to Uint32 for the dest side; their
            # Step 3 readers (bf16 / uint8) see the same underlying
            # bytes via a separate recast.
            x_u32 = cute.recast_tensor(x_gmem, cutlass.Uint32)
            w_u32 = cute.recast_tensor(w_gmem, cutlass.Uint32)
            sA_u32 = cute.recast_tensor(sA, cutlass.Uint32)
            sWpacked_u32 = cute.recast_tensor(sWpacked, cutlass.Uint32)

            gA = cute.local_tile(
                x_u32, (_BLOCK_M, _BLOCK_K // 2), (block_m, None)
            )  # uint32 view, (BLOCK_M, BLOCK_K/2, num_k_tiles)
            gW = cute.local_tile(
                w_u32, (_BLOCK_N, _BLOCK_K // 8), (block_n, None)
            )  # uint32 view, (BLOCK_N, BLOCK_K/8, num_k_tiles)
            # SF has one scale per FP4 group of 16 elements; kept as
            # fp32 because dequant reads it as a scalar per thread.
            gSF = cute.local_tile(
                sf_gmem, (_BLOCK_N, _BLOCK_K // _SF_VEC_SIZE), (block_n, None)
            )  # (BLOCK_N, BLOCK_K / 16, num_k_tiles)

            # Partition copies via uint32 view.
            thr_copy_A = gmem_tiled_copy_A.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA_u32)
            thr_copy_W = gmem_tiled_copy_W.get_slice(tidx)
            tWgW = thr_copy_W.partition_S(gW)
            tWsW = thr_copy_W.partition_D(sWpacked_u32)

            # ---- Step 4 setup: MMA partitions (one-time per CTA) ----
            # Output gmem partition.
            gC = cute.local_tile(
                out_gmem, (_BLOCK_M, _BLOCK_N), (block_m, block_n)
            )
            thr_mma = tiled_mma.get_slice(tidx)
            tCgC = thr_mma.partition_C(gC)

            # Register fragments for A/B/C.
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB_bf16)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None])

            acc_shape = tCgC.shape
            accumulators = cute.make_rmem_tensor(acc_shape, Float32)
            accumulators.fill(0.0)

            # Smem -> register copy.  Plain element-wise universal copy
            # (v3 will replace with LdMatrix8x8x16bOp for perf).
            smem_copy_atom_AB = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_AB, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_AB, tiled_mma)
            thr_smem_copy_A = smem_tiled_copy_A.get_slice(tidx)
            thr_smem_copy_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy = thr_smem_copy_A.partition_S(sA)
            tCsB_copy = thr_smem_copy_B.partition_S(sB_bf16)
            tCrA_copy_view = thr_smem_copy_A.retile(tCrA)
            tCrB_copy_view = thr_smem_copy_B.retile(tCrB)
            num_k_blocks = cute.size(tCrA, mode=[2])

            # Alpha is global (one scalar for the whole call); load once.
            alpha_val = Float32(alpha_gmem[0])

            # ---- Step 4b: K-loop over outer K-tiles ----
            num_k_tiles = cute.size(gA, mode=[2])
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                # ---- Step 2: gmem -> smem for this K-tile ----
                cute.copy(
                    gmem_tiled_copy_A,
                    tAgA[None, None, None, k_tile],
                    tAsA,
                )
                cute.copy(
                    gmem_tiled_copy_W,
                    tWgW[None, None, None, k_tile],
                    tWsW,
                )
                cute.arch.sync_threads()

                # ---- Step 3: FP4 nibble dequant -> bf16 in sB ----
                # Thread layout tW = (BLOCK_N=32, 4) order=(1,0): each
                # thread owns 4 packed bytes = 8 FP4 elems = half an
                # sf-block.  Reads one sf value per thread.
                n_idx = Int32(tidx // 4)
                k_quarter = Int32(tidx % 4)
                sf_block_idx = k_quarter // Int32(2)
                global_n = Int32(block_n) * Int32(_BLOCK_N) + n_idx
                # sf index along K for this outer K-tile: each k_tile
                # covers BLOCK_K/16 = 2 sf-groups.
                global_sf_k = Int32(k_tile) * Int32(_BLOCK_K // _SF_VEC_SIZE) + sf_block_idx

                sf_val = Float32(sf_gmem[global_n, global_sf_k])
                combined_scale = sf_val * alpha_val

                for byte_idx in cutlass.range_constexpr(4):
                    k_byte = k_quarter * Int32(4) + Int32(byte_idx)
                    k_elem = k_quarter * Int32(8) + Int32(byte_idx) * Int32(2)
                    byte = Int32(sWpacked[n_idx, k_byte])
                    lo_code = byte & Int32(0xF)
                    hi_code = (byte >> Int32(4)) & Int32(0xF)

                    def _mag(mag_code):
                        return (
                            Float32(0.0) if mag_code == Int32(0) else
                            Float32(0.5) if mag_code == Int32(1) else
                            Float32(1.0) if mag_code == Int32(2) else
                            Float32(1.5) if mag_code == Int32(3) else
                            Float32(2.0) if mag_code == Int32(4) else
                            Float32(3.0) if mag_code == Int32(5) else
                            Float32(4.0) if mag_code == Int32(6) else
                            Float32(6.0)
                        )

                    lo_mag = _mag(lo_code & Int32(7))
                    lo_sign = Float32(-1.0) if (lo_code & Int32(8)) != Int32(0) else Float32(1.0)
                    hi_mag = _mag(hi_code & Int32(7))
                    hi_sign = Float32(-1.0) if (hi_code & Int32(8)) != Int32(0) else Float32(1.0)
                    sB_bf16[n_idx, k_elem + Int32(0)] = BFloat16(lo_sign * lo_mag * combined_scale)
                    sB_bf16[n_idx, k_elem + Int32(1)] = BFloat16(hi_sign * hi_mag * combined_scale)

                cute.arch.sync_threads()

                # ---- Step 4: MMA over this K-tile's BLOCK_K/16 K-blocks ----
                for k_block in cutlass.range_constexpr(num_k_blocks):
                    cute.copy(
                        smem_tiled_copy_A,
                        tCsA_copy[None, None, k_block],
                        tCrA_copy_view[None, None, k_block],
                    )
                    cute.copy(
                        smem_tiled_copy_B,
                        tCsB_copy[None, None, k_block],
                        tCrB_copy_view[None, None, k_block],
                    )
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA[None, None, k_block],
                        tCrB[None, None, k_block],
                        accumulators,
                    )
                cute.arch.sync_threads()

            # ---- Step 5: smem-staged epilogue ----
            # Stage register accumulators -> sC -> gmem so each warp's
            # fragmented MMA output is reordered into a layout where
            # 128 threads cooperatively issue coalesced bf16 writes to
            # ``out_gmem``.
            #
            # Pattern lifted from b12x/gemm/dense.py:1031-1056 but
            # using CopyUniversalOp instead of StMatrix8x8x16bOp
            # (v3 swap-in for perf).
            copy_atom_r2s = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16,
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16,
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            # Cast fp32 acc -> bf16 in registers, then store to sC.
            acc_bf16 = cute.make_fragment_like(tRS_rAcc, BFloat16)
            acc_bf16.store(tRS_rAcc.load().to(BFloat16))
            cute.copy(copy_atom_r2s, acc_bf16, tRS_sC)
            cute.arch.sync_threads()

            # sC -> gmem via the cooperative tiled copy built in
            # _setup_attributes.  Recast sC and gC to Uint32 so the
            # 128-bit cooperative store fires (16 bytes / thread, 64
            # threads cover the 512-bf16 = 256-uint32 output tile).
            sC_u32 = cute.recast_tensor(sC, cutlass.Uint32)
            gC_u32 = cute.recast_tensor(gC, cutlass.Uint32)
            thr_gmem_copy_C = gmem_tiled_copy_C.get_slice(tidx)
            tCsC_for_gmem = thr_gmem_copy_C.partition_S(sC_u32)
            tCgC_for_gmem = thr_gmem_copy_C.partition_D(gC_u32)
            cute.copy(gmem_tiled_copy_C, tCsC_for_gmem, tCgC_for_gmem)

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
            # All host tensors are row-major (last dim stride 1) after
            # the ``.contiguous()`` shuttle in micro.py.  Express that
            # with ``order=(1, 0)`` (axis 1 is contiguous / stride 1).
            x = cute.make_tensor(
                x_ptr,
                layout=cute.make_ordered_layout(
                    (self.m, self.k), order=(1, 0),
                ),
            )
            w = cute.make_tensor(
                w_ptr,
                layout=cute.make_ordered_layout(
                    (self.n, self.k // 2), order=(1, 0),
                ),
            )
            sf = cute.make_tensor(
                sf_ptr,
                layout=cute.make_ordered_layout(
                    (self.n, self.k // _SF_VEC_SIZE), order=(1, 0),
                ),
            )
            alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((1,)))
            out = cute.make_tensor(
                out_ptr,
                layout=cute.make_ordered_layout(
                    (self.m, self.n), order=(1, 0),
                ),
            )

            grid = (
                (self.m + _BLOCK_M - 1) // _BLOCK_M,
                (self.n + _BLOCK_N - 1) // _BLOCK_N,
                1,
            )
            self.kernel(
                x, w, sf, alpha, out,
                self.SharedStorage,
                self.sA_layout, self.sB_layout, self.sWpacked_layout,
                self.sC_layout,
                self.gmem_tiled_copy_A, self.gmem_tiled_copy_W,
                self.gmem_tiled_copy_C,
                self.tiled_mma,
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
        # v1 cute kernel writes a full (BLOCK_M, BLOCK_N) tile per CTA
        # with no residual handling.  Restrict to exact-multiple shapes
        # so we don't write out of bounds on smaller M.  The Triton
        # backend covers M < BLOCK_M.
        if m != _BLOCK_M:
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
