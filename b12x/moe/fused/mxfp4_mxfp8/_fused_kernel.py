"""Single-kernel fused FC1 + SwiGLU + MXFP8 quant + FC2 — generalized shapes.

Single warp, single CTA. Parametric in:
  M  = 16  (fixed; one m16n8 row group per CTA)
  K_in_blocks    (FC1 K dim / 32)
  FC1_N_tiles    (FC1 N dim / 8). For SwiGLU FC1_N = 2 * I, so FC1_N_tiles = 2 * I_n_tiles.
  I_k_blocks     (intermediate K dim / 32)  → I_n_tiles = I_k_blocks * 4
  K_out_n_tiles  (FC2 N dim / 8)

Activation: SwiGLU. Per-row, per-block-32 UE8M0 scales for the on-device
intermediate quantize. Bytes packed via SMEM round-trip for the D-fragment to
A-fragment layout transpose.
"""

from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float32, Int32, Uint8, Uint32
from cutlass.cute.runtime import from_dlpack

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass import Int64

from b12x.cute.fp4 import (
    cvt_e2m1x8_to_e4m3x8,
    cvt_f32_to_e4m3,
    get_ptr_as_int64,
    ld_shared_i32,
    ld_shared_u8,
    mxfp8_mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    st_shared_u8,
    warp_reduce,
)
from b12x.moe.fused.mxfp4_mxfp8._quant_kernel import (
    floor_f32_to_s32,
    imax_s32,
    imin_s32,
)


@dsl_user_op
def _cp_async_16B(smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None):
    """16-byte async global→shared copy. Uses .ca (cache-always) to match b12x mla kernel."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global.L2::64B [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


M = 16
N_TILES_PER_K_BLOCK = 4   # 4 m16n8 N-tiles cover 32 cols = 1 K-block of 32


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


def _make_kernel_global(
    K_in_blocks: int,
    I_k_blocks: int,
    FC1_N_tiles: int,
    K_out_n_tiles: int,
):
    """Build a fused kernel that loads A/W13/W2 directly from row-major global.

    Inputs are row-major byte tensors:
      mA   : (M_total, K_in)            uint8 — E4M3 activation
      mA_sf: (M_total, K_in/32)         uint8 — UE8M0
      mW13 : (FC1_N, K_in/2)            uint8 — packed E2M1
      mW13_sf: (FC1_N, K_in/32)         uint8 — UE8M0
      mW2  : (K_out, I/2)               uint8 — packed E2M1
      mW2_sf: (K_out, I/32)             uint8 — UE8M0
      mY   : (M_tiles, 16, K_out)       bf16 (per-CTA slot for output)

    Each CTA processes one m16n8 row group (16 rows of A, all of W13, all of
    W2). Per-thread fragments are assembled inline from global byte loads.
    """
    I_n_tiles = I_k_blocks * N_TILES_PER_K_BLOCK
    if FC1_N_tiles != 2 * I_n_tiles:
        raise ValueError(
            f"SwiGLU requires FC1_N_tiles == 2 * I_n_tiles "
            f"(= 2 * {I_n_tiles} = {2 * I_n_tiles}), got {FC1_N_tiles}"
        )

    @cute.kernel
    def kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY):
        cta = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        g = Int32(tidx) // Int32(4)
        lane = Int32(tidx) % Int32(4)

        # CTA's row offsets in the global A.
        row_top = cta * Int32(16) + g
        row_bot = cta * Int32(16) + g + Int32(8)

        # Recast row-major byte tensors as u32 for vectorized 4-byte loads.
        # Each m_A[row, c4] is the u32 holding 4 bytes at columns [c4*4, c4*4 + 4).
        mA_u32 = cute.recast_tensor(mA, cutlass.Uint32)
        mW13_u32 = cute.recast_tensor(mW13, cutlass.Uint32)
        mW2_u32 = cute.recast_tensor(mW2, cutlass.Uint32)

        # ------------------------------------------------------------------
        # SMEM allocation: intermediate (M=16, I_cols) bytes, same as before.
        # ------------------------------------------------------------------
        smem = utils.SmemAllocator()
        I_cols = I_k_blocks * 32

        @cute.struct
        class Storage:
            sInt: cute.struct.MemRange[Uint8, M * I_cols]

        storage = smem.allocate(Storage)
        sInt = storage.sInt.get_tensor(cute.make_layout((M, I_cols)))

        # ------------------------------------------------------------------
        # FC1 K-loop with on-the-fly fragment assembly from global (u32 loads).
        # ------------------------------------------------------------------
        fc1 = [Float32(0.0) for _ in range(FC1_N_tiles * 4)]

        for k in cutlass.range_constexpr(K_in_blocks):
            # A fragment: 4 u32 per thread, each = 4 contiguous bytes at the
            # right (row, col-block) position.
            #   a0: (row_top, col_u32 = k*8 + lane*2 + 0)
            #   a1: (row_bot, col_u32 = k*8 + lane*2 + 0)
            #   a2: (row_top, col_u32 = k*8 + lane*2 + 1)
            #   a3: (row_bot, col_u32 = k*8 + lane*2 + 1)
            col_a_u32 = Int32(k * 8) + lane * Int32(2)
            a0 = Uint32(mA_u32[row_top, col_a_u32 + Int32(0)])
            a1 = Uint32(mA_u32[row_bot, col_a_u32 + Int32(0)])
            a2 = Uint32(mA_u32[row_top, col_a_u32 + Int32(1)])
            a3 = Uint32(mA_u32[row_bot, col_a_u32 + Int32(1)])

            # SFA: per-row UE8M0; lane 0 → row_top, lane 1 → row_bot.
            sfa = Uint32(0x7F)
            if lane == Int32(0):
                sfa = Uint32(mA_sf[row_top, k])
            elif lane == Int32(1):
                sfa = Uint32(mA_sf[row_bot, k])

            for n in cutlass.range_constexpr(FC1_N_tiles):
                # W13 B fragment: 1 u32 = 4 packed bytes at
                # (W13_row = n*8 + g, K_byte_u32 = k*4 + lane).
                wrow = Int32(n * 8) + g
                col_b_u32 = Int32(k * 4) + lane
                bp = Uint32(mW13_u32[wrow, col_b_u32])
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

                # SFB: per-col UE8M0; lane 0 of group g → col g of this N-tile.
                sfb = Uint32(0x7F)
                if lane == Int32(0):
                    sfb = Uint32(mW13_sf[wrow, k])

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    fc1[n * 4 + 0], fc1[n * 4 + 1], fc1[n * 4 + 2], fc1[n * 4 + 3],
                    a0, a1, a2, a3, b0, b1, sfa, sfb,
                )
                fc1[n * 4 + 0] = d0
                fc1[n * 4 + 1] = d1
                fc1[n * 4 + 2] = d2
                fc1[n * 4 + 3] = d3

        # ------------------------------------------------------------------
        # SwiGLU + per-K-block UE8M0 quant (unchanged from prepacked variant).
        # ------------------------------------------------------------------
        intermediate = [Float32(0.0) for _ in range(I_n_tiles * 4)]
        for n in cutlass.range_constexpr(I_n_tiles):
            for i in cutlass.range_constexpr(4):
                gate = fc1[n * 4 + i]
                up = fc1[(n + I_n_tiles) * 4 + i]
                sigmoid_up = Float32(1.0) / (Float32(1.0) + cute.math.exp(-up, fastmath=True))
                intermediate[n * 4 + i] = gate * up * sigmoid_up

        e4m3_bytes = [Uint32(0) for _ in range(I_n_tiles * 4)]
        sf_top_per_kb = [Int32(127) for _ in range(I_k_blocks)]
        sf_bot_per_kb = [Int32(127) for _ in range(I_k_blocks)]
        inv_max = Float32(1.0) / Float32(448.0)
        tiny = Float32(1.401298464e-45)

        for kb in cutlass.range_constexpr(I_k_blocks):
            n_start = kb * N_TILES_PER_K_BLOCK
            local_max_top = Float32(0.0)
            local_max_bot = Float32(0.0)
            for sub in cutlass.range_constexpr(N_TILES_PER_K_BLOCK):
                n = n_start + sub
                for i in cutlass.range_constexpr(2):
                    v_top = intermediate[n * 4 + i]
                    local_max_top = cute.arch.fmax(local_max_top, cute.arch.fmax(v_top, -v_top))
                    v_bot = intermediate[n * 4 + 2 + i]
                    local_max_bot = cute.arch.fmax(local_max_bot, cute.arch.fmax(v_bot, -v_bot))
            row_top_amax = warp_reduce(local_max_top, lambda a, b: cute.arch.fmax(a, b), width=4)
            row_bot_amax = warp_reduce(local_max_bot, lambda a, b: cute.arch.fmax(a, b), width=4)
            safe_top = cute.arch.fmax(row_top_amax * inv_max, tiny)
            safe_bot = cute.arch.fmax(row_bot_amax * inv_max, tiny)
            log_top = cute.math.log2(safe_top, fastmath=True)
            log_bot = cute.math.log2(safe_bot, fastmath=True)
            ceil_top = -floor_f32_to_s32(-log_top)
            ceil_bot = -floor_f32_to_s32(-log_bot)
            sf_top = imin_s32(Int32(254), imax_s32(Int32(0), ceil_top + Int32(127)))
            sf_bot = imin_s32(Int32(254), imax_s32(Int32(0), ceil_bot + Int32(127)))
            sf_top_per_kb[kb] = sf_top
            sf_bot_per_kb[kb] = sf_bot
            inv_scale_top = Float32(1.0) / cute.math.exp2(Float32(sf_top - Int32(127)), fastmath=True)
            inv_scale_bot = Float32(1.0) / cute.math.exp2(Float32(sf_bot - Int32(127)), fastmath=True)
            for sub in cutlass.range_constexpr(N_TILES_PER_K_BLOCK):
                n = n_start + sub
                for i in cutlass.range_constexpr(2):
                    e4m3_bytes[n * 4 + i] = cvt_f32_to_e4m3(intermediate[n * 4 + i] * inv_scale_top)
                    e4m3_bytes[n * 4 + 2 + i] = cvt_f32_to_e4m3(intermediate[n * 4 + 2 + i] * inv_scale_bot)

        # Write quantized intermediate to SMEM (D-fragment → row-major byte layout).
        for n in cutlass.range_constexpr(I_n_tiles):
            for i in cutlass.range_constexpr(2):
                col = lane * Int32(2) + Int32(n * 8) + Int32(i)
                sInt[g,            col] = Uint8(e4m3_bytes[n * 4 + i])
                sInt[g + Int32(8), col] = Uint8(e4m3_bytes[n * 4 + 2 + i])

        cute.arch.sync_warp()

        # ------------------------------------------------------------------
        # FC2 K-loop: reads A from SMEM, W2 from row-major global.
        # ------------------------------------------------------------------
        fc2 = [Float32(0.0) for _ in range(K_out_n_tiles * 4)]

        for kb_fc2 in cutlass.range_constexpr(I_k_blocks):
            col_off = Int32(kb_fc2 * 32)
            fc2_a0 = Uint32(0)
            fc2_a1 = Uint32(0)
            fc2_a2 = Uint32(0)
            fc2_a3 = Uint32(0)
            for i in cutlass.range_constexpr(4):
                col_lo = col_off + lane * Int32(8) + Int32(i)
                col_hi = col_off + lane * Int32(8) + Int32(4 + i)
                fc2_a0 = fc2_a0 | (Uint32(sInt[g,            col_lo]) << Uint32(i * 8))
                fc2_a1 = fc2_a1 | (Uint32(sInt[g + Int32(8), col_lo]) << Uint32(i * 8))
                fc2_a2 = fc2_a2 | (Uint32(sInt[g,            col_hi]) << Uint32(i * 8))
                fc2_a3 = fc2_a3 | (Uint32(sInt[g + Int32(8), col_hi]) << Uint32(i * 8))

            sfa_fc2 = Uint32(0x7F)
            if lane == Int32(0):
                sfa_fc2 = Uint32(sf_top_per_kb[kb_fc2])
            elif lane == Int32(1):
                sfa_fc2 = Uint32(sf_bot_per_kb[kb_fc2])

            for n in cutlass.range_constexpr(K_out_n_tiles):
                # W2 B fragment: 1 u32 at (W2_row = n*8 + g, col_u32 = kb_fc2*4 + lane).
                wrow_2 = Int32(n * 8) + g
                col_b2_u32 = Int32(kb_fc2 * 4) + lane
                bp = Uint32(mW2_u32[wrow_2, col_b2_u32])
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

                sfb_fc2 = Uint32(0x7F)
                if lane == Int32(0):
                    sfb_fc2 = Uint32(mW2_sf[wrow_2, kb_fc2])

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    fc2[n * 4 + 0], fc2[n * 4 + 1], fc2[n * 4 + 2], fc2[n * 4 + 3],
                    fc2_a0, fc2_a1, fc2_a2, fc2_a3, b0, b1, sfa_fc2, sfb_fc2,
                )
                fc2[n * 4 + 0] = d0
                fc2[n * 4 + 1] = d1
                fc2[n * 4 + 2] = d2
                fc2[n * 4 + 3] = d3

        # Write FC2 D fragments to (M_tiles, 16, K_out) bf16 output.
        for n in cutlass.range_constexpr(K_out_n_tiles):
            col0 = Int32(n * 8) + lane * Int32(2)
            mY[cta, g,            col0]            = BFloat16(fc2[n * 4 + 0])
            mY[cta, g,            col0 + Int32(1)] = BFloat16(fc2[n * 4 + 1])
            mY[cta, g + Int32(8), col0]            = BFloat16(fc2[n * 4 + 2])
            mY[cta, g + Int32(8), col0 + Int32(1)] = BFloat16(fc2[n * 4 + 3])

    @cute.jit
    def driver(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY, stream):
        M_tiles = mY.shape[0]
        kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY).launch(
            grid=(M_tiles, 1, 1), block=[32, 1, 1], stream=stream,
        )

    return driver


_compile_cache_global = {}


def _compiled_global(M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles):
    key = (M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles)
    if key in _compile_cache_global:
        return _compile_cache_global[key]
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    M_total = M_tiles * 16
    K_in = K_in_blocks * 32
    FC1_N = FC1_N_tiles * 8
    K_out = K_out_n_tiles * 8
    I = I_k_blocks * 32
    A = torch.zeros(M_total, K_in, dtype=torch.uint8, device=device)
    A_sf = torch.zeros(M_total, K_in_blocks, dtype=torch.uint8, device=device)
    W13 = torch.zeros(FC1_N, K_in // 2, dtype=torch.uint8, device=device)
    W13_sf = torch.zeros(FC1_N, K_in_blocks, dtype=torch.uint8, device=device)
    W2 = torch.zeros(K_out, I // 2, dtype=torch.uint8, device=device)
    W2_sf = torch.zeros(K_out, I_k_blocks, dtype=torch.uint8, device=device)
    Y = torch.zeros(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)
    driver = _make_kernel_global(K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles)
    compiled = cute.compile(
        driver,
        _to_cute(A, cutlass.Uint8),
        _to_cute(A_sf, cutlass.Uint8),
        _to_cute(W13, cutlass.Uint8),
        _to_cute(W13_sf, cutlass.Uint8),
        _to_cute(W2, cutlass.Uint8),
        _to_cute(W2_sf, cutlass.Uint8),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    _compile_cache_global[key] = compiled
    return compiled


def _make_kernel_cpasync(
    K_in_blocks: int,
    I_k_blocks: int,
    FC1_N_tiles: int,
    K_out_n_tiles: int,
):
    """cp.async + raw-SMEM-pointer staging for W13 and W2 K-blocks.

    Lessons from the v1 attempt:
      - cute SMEM tensor reads via `recast_tensor` had a shape-dependent layout
        quirk (separately reproducible in `benchmarks/dbg_cpasync.py`); v2 skips
        the abstraction and uses raw `ld.shared.s32` inline asm so cp.async
        byte writes and SMEM byte reads use a single consistent pointer model.

    Same as `_make_kernel_global` but stages W13 and W2 K-blocks via cp.async.

    Each FC1 K-iteration:
      1. cooperatively cp.async one K-block of W13 (FC1_N rows × 16 packed bytes)
         into SMEM
      2. cp.async.commit_group + wait_group(0)
      3. inner N-tile loop reads W13 fragments from SMEM (4-byte aligned u32 reads)

    Same pattern for FC2 K-loop with W2. A and the SF tensors are still loaded
    directly from global; A is small (16 × K_in bytes) and SF is tiny.

    Single-buffered for clarity; double-buffering would overlap async with
    compute and is the next perf step.

    Constraints: FC1_N must be a multiple of 32 (for clean cooperative load
    distribution; FC1_N >= 64 from SwiGLU + I >= 32 already satisfies this if
    I >= 16). Likewise K_out must be a multiple of 32.
    """
    I_n_tiles = I_k_blocks * N_TILES_PER_K_BLOCK
    if FC1_N_tiles != 2 * I_n_tiles:
        raise ValueError(
            f"SwiGLU requires FC1_N_tiles == 2 * I_n_tiles, got {FC1_N_tiles}"
        )
    FC1_N = FC1_N_tiles * 8
    K_out = K_out_n_tiles * 8
    if FC1_N % 32 != 0:
        raise ValueError(f"cp.async path requires FC1_N % 32 == 0; got {FC1_N}")
    if K_out % 32 != 0:
        raise ValueError(f"cp.async path requires K_out % 32 == 0; got {K_out}")
    fc1_loads_per_thread = FC1_N // 32
    fc2_loads_per_thread = K_out // 32

    @cute.kernel
    def kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY):
        cta = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        g = Int32(tidx) // Int32(4)
        lane = Int32(tidx) % Int32(4)

        row_top = cta * Int32(16) + g
        row_bot = cta * Int32(16) + g + Int32(8)

        mA_u32 = cute.recast_tensor(mA, cutlass.Uint32)

        # SMEM: intermediate (16, I_cols) bytes + W13/W2 K-block staging.
        # Re-bind shape constants as kernel-local names so the @cute.struct
        # class body resolves them via the kernel's lexical scope (closures
        # from the outer factory don't propagate into @cute.struct evaluation).
        smem = utils.SmemAllocator()
        I_cols = I_k_blocks * 32
        FC1_N_local = FC1_N_tiles * 8
        K_out_local = K_out_n_tiles * 8

        # Double-buffered SMEM for W13 and W2 K-block staging.
        # Buffer i (i ∈ {0, 1}) starts at base + i * (FC1_N_local or K_out_local) * 16.
        @cute.struct
        class Storage:
            sInt: cute.struct.MemRange[Uint8, M * I_cols]
            sW13_kb: cute.struct.MemRange[Uint8, 2 * FC1_N_local * 16]
            sW2_kb: cute.struct.MemRange[Uint8, 2 * K_out_local * 16]

        storage = smem.allocate(Storage)
        # sInt still goes through cute tensor — sInt is (16, I_cols) row-major
        # which matches the working `_make_kernel_global` SMEM-staged
        # intermediate path that's already validated.
        sInt = storage.sInt.get_tensor(cute.make_layout((M, I_cols)))
        # W13/W2 K-block staging uses raw SMEM byte addresses end-to-end
        # (bypassing the cute tensor abstraction for both writes and reads).
        sW13_smem_addr = shared_ptr_to_u32(storage.sW13_kb.data_ptr())
        sW2_smem_addr = shared_ptr_to_u32(storage.sW2_kb.data_ptr())

        # Each thread loads (FC1_N // 32) rows of W13 per K-block (16 bytes each
        # via one cp.async), distributed as: thread t in iter i loads row i*32+t.
        K_half = K_in_blocks * 16

        # ------------------------------------------------------------------
        # FC1 K-loop with double-buffered cp.async-staged W13.
        # Prologue prefetches K-block 0 into buf 0; each iteration prefetches
        # the next K-block into the alternate buffer while we MMA the current.
        # ------------------------------------------------------------------
        fc1 = [Float32(0.0) for _ in range(FC1_N_tiles * 4)]
        W13_BUF_BYTES = FC1_N_local * 16

        # Prologue: prefetch K-block 0 into buffer 0.
        for i in cutlass.range_constexpr(fc1_loads_per_thread):
            row = Int32(i * 32) + Int32(tidx)
            src_addr = get_ptr_as_int64(mW13, row * Int32(K_half) + Int32(0))
            dst_addr = sW13_smem_addr + row * Int32(16)
            _cp_async_16B(dst_addr, src_addr)
        cute.arch.cp_async_commit_group()

        for k in cutlass.range_constexpr(K_in_blocks):
            buf_cur = k % 2
            # Issue prefetch for next K-block (if any) into the alternate buffer.
            if k + 1 < K_in_blocks:
                buf_next = (k + 1) % 2
                for i in cutlass.range_constexpr(fc1_loads_per_thread):
                    row = Int32(i * 32) + Int32(tidx)
                    src_addr = get_ptr_as_int64(
                        mW13, row * Int32(K_half) + Int32((k + 1) * 16)
                    )
                    dst_addr = (sW13_smem_addr
                                + Int32(buf_next * W13_BUF_BYTES)
                                + row * Int32(16))
                    _cp_async_16B(dst_addr, src_addr)
                cute.arch.cp_async_commit_group()
                # Two groups in flight (current k, next k+1); wait for current.
                cute.arch.cp_async_wait_group(1)
            else:
                # Last K-iter: only the prologue/current group remains. Wait for it.
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()
            sW13_buf_addr = sW13_smem_addr + Int32(buf_cur * W13_BUF_BYTES)

            # A fragment from row-major global (small enough to skip staging).
            col_a_u32 = Int32(k * 8) + lane * Int32(2)
            a0 = Uint32(mA_u32[row_top, col_a_u32 + Int32(0)])
            a1 = Uint32(mA_u32[row_bot, col_a_u32 + Int32(0)])
            a2 = Uint32(mA_u32[row_top, col_a_u32 + Int32(1)])
            a3 = Uint32(mA_u32[row_bot, col_a_u32 + Int32(1)])

            sfa = Uint32(0x7F)
            if lane == Int32(0):
                sfa = Uint32(mA_sf[row_top, k])
            elif lane == Int32(1):
                sfa = Uint32(mA_sf[row_bot, k])

            for n in cutlass.range_constexpr(FC1_N_tiles):
                wrow = Int32(n * 8) + g
                # Raw SMEM byte address into the current double-buffer slot.
                bp_addr = sW13_buf_addr + wrow * Int32(16) + lane * Int32(4)
                bp = Uint32(ld_shared_i32(bp_addr))
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

                sfb = Uint32(0x7F)
                if lane == Int32(0):
                    sfb = Uint32(mW13_sf[wrow, k])

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    fc1[n * 4 + 0], fc1[n * 4 + 1], fc1[n * 4 + 2], fc1[n * 4 + 3],
                    a0, a1, a2, a3, b0, b1, sfa, sfb,
                )
                fc1[n * 4 + 0] = d0
                fc1[n * 4 + 1] = d1
                fc1[n * 4 + 2] = d2
                fc1[n * 4 + 3] = d3

        # ------------------------------------------------------------------
        # SwiGLU + per-K-block UE8M0 quant (unchanged).
        # ------------------------------------------------------------------
        intermediate = [Float32(0.0) for _ in range(I_n_tiles * 4)]
        for n in cutlass.range_constexpr(I_n_tiles):
            for i in cutlass.range_constexpr(4):
                gate = fc1[n * 4 + i]
                up = fc1[(n + I_n_tiles) * 4 + i]
                sigmoid_up = Float32(1.0) / (Float32(1.0) + cute.math.exp(-up, fastmath=True))
                intermediate[n * 4 + i] = gate * up * sigmoid_up

        e4m3_bytes = [Uint32(0) for _ in range(I_n_tiles * 4)]
        sf_top_per_kb = [Int32(127) for _ in range(I_k_blocks)]
        sf_bot_per_kb = [Int32(127) for _ in range(I_k_blocks)]
        inv_max = Float32(1.0) / Float32(448.0)
        tiny = Float32(1.401298464e-45)
        for kb in cutlass.range_constexpr(I_k_blocks):
            n_start = kb * N_TILES_PER_K_BLOCK
            local_max_top = Float32(0.0)
            local_max_bot = Float32(0.0)
            for sub in cutlass.range_constexpr(N_TILES_PER_K_BLOCK):
                n = n_start + sub
                for i in cutlass.range_constexpr(2):
                    v_top = intermediate[n * 4 + i]
                    local_max_top = cute.arch.fmax(local_max_top, cute.arch.fmax(v_top, -v_top))
                    v_bot = intermediate[n * 4 + 2 + i]
                    local_max_bot = cute.arch.fmax(local_max_bot, cute.arch.fmax(v_bot, -v_bot))
            row_top_amax = warp_reduce(local_max_top, lambda a, b: cute.arch.fmax(a, b), width=4)
            row_bot_amax = warp_reduce(local_max_bot, lambda a, b: cute.arch.fmax(a, b), width=4)
            safe_top = cute.arch.fmax(row_top_amax * inv_max, tiny)
            safe_bot = cute.arch.fmax(row_bot_amax * inv_max, tiny)
            log_top = cute.math.log2(safe_top, fastmath=True)
            log_bot = cute.math.log2(safe_bot, fastmath=True)
            ceil_top = -floor_f32_to_s32(-log_top)
            ceil_bot = -floor_f32_to_s32(-log_bot)
            sf_top = imin_s32(Int32(254), imax_s32(Int32(0), ceil_top + Int32(127)))
            sf_bot = imin_s32(Int32(254), imax_s32(Int32(0), ceil_bot + Int32(127)))
            sf_top_per_kb[kb] = sf_top
            sf_bot_per_kb[kb] = sf_bot
            inv_scale_top = Float32(1.0) / cute.math.exp2(Float32(sf_top - Int32(127)), fastmath=True)
            inv_scale_bot = Float32(1.0) / cute.math.exp2(Float32(sf_bot - Int32(127)), fastmath=True)
            for sub in cutlass.range_constexpr(N_TILES_PER_K_BLOCK):
                n = n_start + sub
                for i in cutlass.range_constexpr(2):
                    e4m3_bytes[n * 4 + i] = cvt_f32_to_e4m3(intermediate[n * 4 + i] * inv_scale_top)
                    e4m3_bytes[n * 4 + 2 + i] = cvt_f32_to_e4m3(intermediate[n * 4 + 2 + i] * inv_scale_bot)

        for n in cutlass.range_constexpr(I_n_tiles):
            for i in cutlass.range_constexpr(2):
                col = lane * Int32(2) + Int32(n * 8) + Int32(i)
                sInt[g,            col] = Uint8(e4m3_bytes[n * 4 + i])
                sInt[g + Int32(8), col] = Uint8(e4m3_bytes[n * 4 + 2 + i])

        cute.arch.sync_warp()

        # ------------------------------------------------------------------
        # FC2 K-loop with cp.async-staged W2.
        # ------------------------------------------------------------------
        fc2 = [Float32(0.0) for _ in range(K_out_n_tiles * 4)]
        I_half = I_k_blocks * 16
        W2_BUF_BYTES = K_out_local * 16

        # FC2 prologue: prefetch W2 K-block 0 into buffer 0.
        for i in cutlass.range_constexpr(fc2_loads_per_thread):
            row = Int32(i * 32) + Int32(tidx)
            src_addr = get_ptr_as_int64(mW2, row * Int32(I_half) + Int32(0))
            dst_addr = sW2_smem_addr + row * Int32(16)
            _cp_async_16B(dst_addr, src_addr)
        cute.arch.cp_async_commit_group()

        for kb_fc2 in cutlass.range_constexpr(I_k_blocks):
            buf_cur_2 = kb_fc2 % 2
            if kb_fc2 + 1 < I_k_blocks:
                buf_next_2 = (kb_fc2 + 1) % 2
                for i in cutlass.range_constexpr(fc2_loads_per_thread):
                    row = Int32(i * 32) + Int32(tidx)
                    src_addr = get_ptr_as_int64(
                        mW2, row * Int32(I_half) + Int32((kb_fc2 + 1) * 16)
                    )
                    dst_addr = (sW2_smem_addr
                                + Int32(buf_next_2 * W2_BUF_BYTES)
                                + row * Int32(16))
                    _cp_async_16B(dst_addr, src_addr)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()
            sW2_buf_addr = sW2_smem_addr + Int32(buf_cur_2 * W2_BUF_BYTES)

            col_off = Int32(kb_fc2 * 32)
            fc2_a0 = Uint32(0)
            fc2_a1 = Uint32(0)
            fc2_a2 = Uint32(0)
            fc2_a3 = Uint32(0)
            for i in cutlass.range_constexpr(4):
                col_lo = col_off + lane * Int32(8) + Int32(i)
                col_hi = col_off + lane * Int32(8) + Int32(4 + i)
                fc2_a0 = fc2_a0 | (Uint32(sInt[g,            col_lo]) << Uint32(i * 8))
                fc2_a1 = fc2_a1 | (Uint32(sInt[g + Int32(8), col_lo]) << Uint32(i * 8))
                fc2_a2 = fc2_a2 | (Uint32(sInt[g,            col_hi]) << Uint32(i * 8))
                fc2_a3 = fc2_a3 | (Uint32(sInt[g + Int32(8), col_hi]) << Uint32(i * 8))

            sfa_fc2 = Uint32(0x7F)
            if lane == Int32(0):
                sfa_fc2 = Uint32(sf_top_per_kb[kb_fc2])
            elif lane == Int32(1):
                sfa_fc2 = Uint32(sf_bot_per_kb[kb_fc2])

            for n in cutlass.range_constexpr(K_out_n_tiles):
                wrow_2 = Int32(n * 8) + g
                # Raw SMEM byte address into the current W2 double-buffer slot.
                bp_addr_2 = sW2_buf_addr + wrow_2 * Int32(16) + lane * Int32(4)
                bp = Uint32(ld_shared_i32(bp_addr_2))
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

                sfb_fc2 = Uint32(0x7F)
                if lane == Int32(0):
                    sfb_fc2 = Uint32(mW2_sf[wrow_2, kb_fc2])

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    fc2[n * 4 + 0], fc2[n * 4 + 1], fc2[n * 4 + 2], fc2[n * 4 + 3],
                    fc2_a0, fc2_a1, fc2_a2, fc2_a3, b0, b1, sfa_fc2, sfb_fc2,
                )
                fc2[n * 4 + 0] = d0
                fc2[n * 4 + 1] = d1
                fc2[n * 4 + 2] = d2
                fc2[n * 4 + 3] = d3

        for n in cutlass.range_constexpr(K_out_n_tiles):
            col0 = Int32(n * 8) + lane * Int32(2)
            mY[cta, g,            col0]            = BFloat16(fc2[n * 4 + 0])
            mY[cta, g,            col0 + Int32(1)] = BFloat16(fc2[n * 4 + 1])
            mY[cta, g + Int32(8), col0]            = BFloat16(fc2[n * 4 + 2])
            mY[cta, g + Int32(8), col0 + Int32(1)] = BFloat16(fc2[n * 4 + 3])

    @cute.jit
    def driver(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY, stream):
        M_tiles = mY.shape[0]
        kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY).launch(
            grid=(M_tiles, 1, 1), block=[32, 1, 1], stream=stream,
        )

    return driver


_compile_cache_cpasync = {}


def _compiled_cpasync(M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles):
    key = (M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles)
    if key in _compile_cache_cpasync:
        return _compile_cache_cpasync[key]
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    M_total = M_tiles * 16
    K_in = K_in_blocks * 32
    FC1_N = FC1_N_tiles * 8
    K_out = K_out_n_tiles * 8
    I = I_k_blocks * 32
    A = torch.zeros(M_total, K_in, dtype=torch.uint8, device=device)
    A_sf = torch.zeros(M_total, K_in_blocks, dtype=torch.uint8, device=device)
    W13 = torch.zeros(FC1_N, K_in // 2, dtype=torch.uint8, device=device)
    W13_sf = torch.zeros(FC1_N, K_in_blocks, dtype=torch.uint8, device=device)
    W2 = torch.zeros(K_out, I // 2, dtype=torch.uint8, device=device)
    W2_sf = torch.zeros(K_out, I_k_blocks, dtype=torch.uint8, device=device)
    Y = torch.zeros(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)
    driver = _make_kernel_cpasync(K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles)
    compiled = cute.compile(
        driver,
        _to_cute(A, cutlass.Uint8),
        _to_cute(A_sf, cutlass.Uint8),
        _to_cute(W13, cutlass.Uint8),
        _to_cute(W13_sf, cutlass.Uint8),
        _to_cute(W2, cutlass.Uint8),
        _to_cute(W2_sf, cutlass.Uint8),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    _compile_cache_cpasync[key] = compiled
    return compiled


def _make_kernel_cpasync_mw(
    warps_per_cta: int,
    K_in_blocks: int,
    I_k_blocks: int,
    FC1_N_tiles: int,
    K_out_n_tiles: int,
):
    """Multi-warp-per-CTA fused kernel with cp.async-staged W13/W2 (single-buffered).

    Each CTA contains `warps_per_cta` warps; each warp owns its own m16n8 row
    group. All warps in a CTA share W13/W2 SMEM buffers — that's the perf win:
    weights are loaded once per K-block per CTA and reused by W warps.

    - Block size = warps_per_cta * 32 threads.
    - Warp 0 does the cp.async loads (other warps wait at sync_threads).
    - Per-warp intermediate buffer in SMEM (M × I_cols bytes per warp).
    - All SMEM addressing is raw pointer arithmetic via `shared_ptr_to_u32`,
      `ld_shared_i32`, `st_shared_u8` — same approach as single-warp cpasync.
    - Single-buffered for now (double-buffer is a future optimization on top).
    """
    I_n_tiles = I_k_blocks * N_TILES_PER_K_BLOCK
    if FC1_N_tiles != 2 * I_n_tiles:
        raise ValueError(f"SwiGLU requires FC1_N_tiles == 2*I_n_tiles, got {FC1_N_tiles}")
    FC1_N = FC1_N_tiles * 8
    K_out = K_out_n_tiles * 8
    if FC1_N % 32 != 0:
        raise ValueError(f"requires FC1_N % 32 == 0; got {FC1_N}")
    if K_out % 32 != 0:
        raise ValueError(f"requires K_out % 32 == 0; got {K_out}")
    if warps_per_cta < 1 or warps_per_cta > 8:
        raise ValueError(f"warps_per_cta must be in [1, 8]; got {warps_per_cta}")
    fc1_loads_per_thread = FC1_N // 32
    fc2_loads_per_thread = K_out // 32
    W = warps_per_cta
    block_size = W * 32

    @cute.kernel
    def kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY):
        cta = cute.arch.block_idx()[0]
        block_tid = cute.arch.thread_idx()[0]
        warp_id = Int32(block_tid) // Int32(32)
        lane = Int32(block_tid) % Int32(32)
        g = lane // Int32(4)
        l = lane % Int32(4)

        # This warp's m16n8 row group.
        m_group = cta * Int32(W) + warp_id
        row_top = m_group * Int32(16) + g
        row_bot = m_group * Int32(16) + g + Int32(8)

        mA_u32 = cute.recast_tensor(mA, cutlass.Uint32)

        # SMEM: per-warp intermediate + shared W13/W2 K-block buffers.
        smem = utils.SmemAllocator()
        I_cols = I_k_blocks * 32
        FC1_N_local = FC1_N_tiles * 8
        K_out_local = K_out_n_tiles * 8
        W_local = W
        INT_BYTES_PER_WARP = M * I_cols

        @cute.struct
        class Storage:
            sInt: cute.struct.MemRange[Uint8, W_local * M * I_cols]
            sW13_kb: cute.struct.MemRange[Uint8, FC1_N_local * 16]
            sW2_kb: cute.struct.MemRange[Uint8, K_out_local * 16]

        storage = smem.allocate(Storage)
        sInt_base = shared_ptr_to_u32(storage.sInt.data_ptr())
        sInt_warp = sInt_base + warp_id * Int32(INT_BYTES_PER_WARP)
        sW13_smem_addr = shared_ptr_to_u32(storage.sW13_kb.data_ptr())
        sW2_smem_addr = shared_ptr_to_u32(storage.sW2_kb.data_ptr())

        K_half = K_in_blocks * 16

        # ------------------------------------------------------------------
        # FC1 K-loop with cp.async-staged W13 (single-buffered + multi-warp).
        # ------------------------------------------------------------------
        fc1 = [Float32(0.0) for _ in range(FC1_N_tiles * 4)]

        for k in cutlass.range_constexpr(K_in_blocks):
            # Warp 0 issues cp.async loads; all warps wait via sync_threads.
            if warp_id == Int32(0):
                for i in cutlass.range_constexpr(fc1_loads_per_thread):
                    row = Int32(i * 32) + lane
                    src_addr = get_ptr_as_int64(
                        mW13, row * Int32(K_half) + Int32(k * 16)
                    )
                    dst_addr = sW13_smem_addr + row * Int32(16)
                    _cp_async_16B(dst_addr, src_addr)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # A fragment from row-major global (each warp reads its own).
            col_a_u32 = Int32(k * 8) + l * Int32(2)
            a0 = Uint32(mA_u32[row_top, col_a_u32 + Int32(0)])
            a1 = Uint32(mA_u32[row_bot, col_a_u32 + Int32(0)])
            a2 = Uint32(mA_u32[row_top, col_a_u32 + Int32(1)])
            a3 = Uint32(mA_u32[row_bot, col_a_u32 + Int32(1)])

            sfa = Uint32(0x7F)
            if l == Int32(0):
                sfa = Uint32(mA_sf[row_top, k])
            elif l == Int32(1):
                sfa = Uint32(mA_sf[row_bot, k])

            for n in cutlass.range_constexpr(FC1_N_tiles):
                wrow = Int32(n * 8) + g
                bp_addr = sW13_smem_addr + wrow * Int32(16) + l * Int32(4)
                bp = Uint32(ld_shared_i32(bp_addr))
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

                sfb = Uint32(0x7F)
                if l == Int32(0):
                    sfb = Uint32(mW13_sf[wrow, k])

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    fc1[n * 4 + 0], fc1[n * 4 + 1], fc1[n * 4 + 2], fc1[n * 4 + 3],
                    a0, a1, a2, a3, b0, b1, sfa, sfb,
                )
                fc1[n * 4 + 0] = d0
                fc1[n * 4 + 1] = d1
                fc1[n * 4 + 2] = d2
                fc1[n * 4 + 3] = d3

        # ------------------------------------------------------------------
        # SwiGLU + per-K-block UE8M0 quant (per-warp; uses warp_reduce width=4).
        # ------------------------------------------------------------------
        intermediate = [Float32(0.0) for _ in range(I_n_tiles * 4)]
        for n in cutlass.range_constexpr(I_n_tiles):
            for i in cutlass.range_constexpr(4):
                gate = fc1[n * 4 + i]
                up = fc1[(n + I_n_tiles) * 4 + i]
                sigmoid_up = Float32(1.0) / (Float32(1.0) + cute.math.exp(-up, fastmath=True))
                intermediate[n * 4 + i] = gate * up * sigmoid_up

        e4m3_bytes = [Uint32(0) for _ in range(I_n_tiles * 4)]
        sf_top_per_kb = [Int32(127) for _ in range(I_k_blocks)]
        sf_bot_per_kb = [Int32(127) for _ in range(I_k_blocks)]
        inv_max = Float32(1.0) / Float32(448.0)
        tiny = Float32(1.401298464e-45)
        for kb in cutlass.range_constexpr(I_k_blocks):
            n_start = kb * N_TILES_PER_K_BLOCK
            local_max_top = Float32(0.0)
            local_max_bot = Float32(0.0)
            for sub in cutlass.range_constexpr(N_TILES_PER_K_BLOCK):
                n = n_start + sub
                for i in cutlass.range_constexpr(2):
                    v_top = intermediate[n * 4 + i]
                    local_max_top = cute.arch.fmax(local_max_top, cute.arch.fmax(v_top, -v_top))
                    v_bot = intermediate[n * 4 + 2 + i]
                    local_max_bot = cute.arch.fmax(local_max_bot, cute.arch.fmax(v_bot, -v_bot))
            row_top_amax = warp_reduce(local_max_top, lambda a, b: cute.arch.fmax(a, b), width=4)
            row_bot_amax = warp_reduce(local_max_bot, lambda a, b: cute.arch.fmax(a, b), width=4)
            safe_top = cute.arch.fmax(row_top_amax * inv_max, tiny)
            safe_bot = cute.arch.fmax(row_bot_amax * inv_max, tiny)
            log_top = cute.math.log2(safe_top, fastmath=True)
            log_bot = cute.math.log2(safe_bot, fastmath=True)
            ceil_top = -floor_f32_to_s32(-log_top)
            ceil_bot = -floor_f32_to_s32(-log_bot)
            sf_top = imin_s32(Int32(254), imax_s32(Int32(0), ceil_top + Int32(127)))
            sf_bot = imin_s32(Int32(254), imax_s32(Int32(0), ceil_bot + Int32(127)))
            sf_top_per_kb[kb] = sf_top
            sf_bot_per_kb[kb] = sf_bot
            inv_scale_top = Float32(1.0) / cute.math.exp2(Float32(sf_top - Int32(127)), fastmath=True)
            inv_scale_bot = Float32(1.0) / cute.math.exp2(Float32(sf_bot - Int32(127)), fastmath=True)
            for sub in cutlass.range_constexpr(N_TILES_PER_K_BLOCK):
                n = n_start + sub
                for i in cutlass.range_constexpr(2):
                    e4m3_bytes[n * 4 + i] = cvt_f32_to_e4m3(intermediate[n * 4 + i] * inv_scale_top)
                    e4m3_bytes[n * 4 + 2 + i] = cvt_f32_to_e4m3(intermediate[n * 4 + 2 + i] * inv_scale_bot)

        # Write quantized intermediate to per-warp SMEM via st.shared.u8.
        for n in cutlass.range_constexpr(I_n_tiles):
            for i in cutlass.range_constexpr(2):
                col = l * Int32(2) + Int32(n * 8) + Int32(i)
                top_addr = sInt_warp + g * Int32(I_cols) + col
                bot_addr = sInt_warp + (g + Int32(8)) * Int32(I_cols) + col
                st_shared_u8(top_addr, Uint8(e4m3_bytes[n * 4 + i]))
                st_shared_u8(bot_addr, Uint8(e4m3_bytes[n * 4 + 2 + i]))

        cute.arch.sync_warp()  # only this warp's intermediate writes need to be visible to itself

        # ------------------------------------------------------------------
        # FC2 K-loop with cp.async-staged W2 (warp 0 loads, all warps consume).
        # ------------------------------------------------------------------
        fc2 = [Float32(0.0) for _ in range(K_out_n_tiles * 4)]
        I_half = I_k_blocks * 16

        for kb_fc2 in cutlass.range_constexpr(I_k_blocks):
            if warp_id == Int32(0):
                for i in cutlass.range_constexpr(fc2_loads_per_thread):
                    row = Int32(i * 32) + lane
                    src_addr = get_ptr_as_int64(
                        mW2, row * Int32(I_half) + Int32(kb_fc2 * 16)
                    )
                    dst_addr = sW2_smem_addr + row * Int32(16)
                    _cp_async_16B(dst_addr, src_addr)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # FC2 A fragment from this warp's intermediate SMEM.
            col_off = Int32(kb_fc2 * 32)
            fc2_a0 = Uint32(0)
            fc2_a1 = Uint32(0)
            fc2_a2 = Uint32(0)
            fc2_a3 = Uint32(0)
            for i in cutlass.range_constexpr(4):
                col_lo = col_off + l * Int32(8) + Int32(i)
                col_hi = col_off + l * Int32(8) + Int32(4 + i)
                # Each thread reads 1 byte at a time via 4-byte loads + extract.
                # Simpler: 4 byte loads composed manually (use ld.shared.s32 with per-byte access).
                # Even simpler: read u32 covering each col_lo..col_lo+3 in one go.
                pass
            # Use 4-byte u32 reads instead of byte-by-byte.
            # SMEM intermediate is per-warp at sInt_warp, layout (M=16, I_cols).
            # Per-thread A frag for FC2: 4 b32, reading bytes at:
            #   a0: row=g,    cols [col_off + l*8, col_off + l*8 + 4)  -> 1 u32 read
            #   a1: row=g+8,  cols [col_off + l*8, col_off + l*8 + 4)
            #   a2: row=g,    cols [col_off + l*8 + 4, col_off + l*8 + 8)
            #   a3: row=g+8,  cols [col_off + l*8 + 4, col_off + l*8 + 8)
            base_lo_top = sInt_warp + g * Int32(I_cols) + col_off + l * Int32(8)
            base_lo_bot = sInt_warp + (g + Int32(8)) * Int32(I_cols) + col_off + l * Int32(8)
            fc2_a0 = Uint32(ld_shared_i32(base_lo_top + Int32(0)))
            fc2_a1 = Uint32(ld_shared_i32(base_lo_bot + Int32(0)))
            fc2_a2 = Uint32(ld_shared_i32(base_lo_top + Int32(4)))
            fc2_a3 = Uint32(ld_shared_i32(base_lo_bot + Int32(4)))

            sfa_fc2 = Uint32(0x7F)
            if l == Int32(0):
                sfa_fc2 = Uint32(sf_top_per_kb[kb_fc2])
            elif l == Int32(1):
                sfa_fc2 = Uint32(sf_bot_per_kb[kb_fc2])

            for n in cutlass.range_constexpr(K_out_n_tiles):
                wrow_2 = Int32(n * 8) + g
                bp_addr_2 = sW2_smem_addr + wrow_2 * Int32(16) + l * Int32(4)
                bp = Uint32(ld_shared_i32(bp_addr_2))
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

                sfb_fc2 = Uint32(0x7F)
                if l == Int32(0):
                    sfb_fc2 = Uint32(mW2_sf[wrow_2, kb_fc2])

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    fc2[n * 4 + 0], fc2[n * 4 + 1], fc2[n * 4 + 2], fc2[n * 4 + 3],
                    fc2_a0, fc2_a1, fc2_a2, fc2_a3, b0, b1, sfa_fc2, sfb_fc2,
                )
                fc2[n * 4 + 0] = d0
                fc2[n * 4 + 1] = d1
                fc2[n * 4 + 2] = d2
                fc2[n * 4 + 3] = d3

        # Write FC2 D fragments to global (M_tiles, 16, K_out) bf16.
        for n in cutlass.range_constexpr(K_out_n_tiles):
            col0 = Int32(n * 8) + l * Int32(2)
            mY[m_group, g,            col0]            = BFloat16(fc2[n * 4 + 0])
            mY[m_group, g,            col0 + Int32(1)] = BFloat16(fc2[n * 4 + 1])
            mY[m_group, g + Int32(8), col0]            = BFloat16(fc2[n * 4 + 2])
            mY[m_group, g + Int32(8), col0 + Int32(1)] = BFloat16(fc2[n * 4 + 3])

    @cute.jit
    def driver(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY, stream):
        M_tiles = mY.shape[0]
        # Each CTA owns W warps, each warp = one m16n8 row group.
        # M_tiles row groups total → M_tiles / W CTAs.
        kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY).launch(
            grid=(M_tiles // W, 1, 1), block=[W * 32, 1, 1], stream=stream,
        )

    return driver


_compile_cache_cpasync_mw = {}


def _compiled_cpasync_mw(M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles, warps_per_cta):
    key = (M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles, warps_per_cta)
    if key in _compile_cache_cpasync_mw:
        return _compile_cache_cpasync_mw[key]
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    M_total = M_tiles * 16
    K_in = K_in_blocks * 32
    FC1_N = FC1_N_tiles * 8
    K_out = K_out_n_tiles * 8
    I = I_k_blocks * 32
    A = torch.zeros(M_total, K_in, dtype=torch.uint8, device=device)
    A_sf = torch.zeros(M_total, K_in_blocks, dtype=torch.uint8, device=device)
    W13 = torch.zeros(FC1_N, K_in // 2, dtype=torch.uint8, device=device)
    W13_sf = torch.zeros(FC1_N, K_in_blocks, dtype=torch.uint8, device=device)
    W2 = torch.zeros(K_out, I // 2, dtype=torch.uint8, device=device)
    W2_sf = torch.zeros(K_out, I_k_blocks, dtype=torch.uint8, device=device)
    Y = torch.zeros(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)
    driver = _make_kernel_cpasync_mw(
        warps_per_cta, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles,
    )
    compiled = cute.compile(
        driver,
        _to_cute(A, cutlass.Uint8),
        _to_cute(A_sf, cutlass.Uint8),
        _to_cute(W13, cutlass.Uint8),
        _to_cute(W13_sf, cutlass.Uint8),
        _to_cute(W2, cutlass.Uint8),
        _to_cute(W2_sf, cutlass.Uint8),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    _compile_cache_cpasync_mw[key] = compiled
    return compiled


def run_fused_silu_cpasync_mw(
    x_e4m3: torch.Tensor,
    x_sf: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_sf: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_sf: torch.Tensor,
    *,
    warps_per_cta: int = 2,
) -> torch.Tensor:
    """Multi-warp-per-CTA cp.async fused kernel. Same calling shape as the
    single-warp variants. Constraint: M must be a multiple of `16 * warps_per_cta`.
    """
    M_actual, K_in = x_e4m3.shape
    FC1_N, K_half = w13_packed.shape
    K_out, I_half = w2_packed.shape
    if K_half * 2 != K_in:
        raise ValueError(f"w13 K/2={K_half} mismatches K_in={K_in}")
    I = I_half * 2
    K_in_blocks = K_in // 32
    I_k_blocks = I // 32
    FC1_N_tiles = FC1_N // 8
    K_out_n_tiles = K_out // 8

    # Pad M to multiple of 16 * warps_per_cta.
    align = 16 * warps_per_cta
    pad = (align - M_actual % align) % align
    if pad:
        x_e4m3 = torch.cat(
            [x_e4m3, torch.zeros(pad, K_in, dtype=x_e4m3.dtype, device=x_e4m3.device)],
            dim=0,
        )
        x_sf = torch.cat(
            [x_sf, torch.zeros(pad, K_in_blocks, dtype=x_sf.dtype, device=x_sf.device)],
            dim=0,
        )
    M_total = x_e4m3.shape[0]
    M_tiles = M_total // 16

    device = x_e4m3.device
    Y = torch.zeros(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)
    compiled = _compiled_cpasync_mw(
        M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles, warps_per_cta,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        _to_cute(x_e4m3, cutlass.Uint8),
        _to_cute(x_sf, cutlass.Uint8),
        _to_cute(w13_packed, cutlass.Uint8),
        _to_cute(w13_sf, cutlass.Uint8),
        _to_cute(w2_packed, cutlass.Uint8),
        _to_cute(w2_sf, cutlass.Uint8),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    torch.cuda.synchronize()
    out_flat = Y.reshape(M_total, K_out)
    if pad:
        out_flat = out_flat[:M_actual].clone()
    return out_flat


def run_fused_silu_cpasync(
    x_e4m3: torch.Tensor,
    x_sf: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_sf: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_sf: torch.Tensor,
) -> torch.Tensor:
    """cp.async-staged variant of run_fused_silu_global. Same calling shape."""
    M_actual, K_in = x_e4m3.shape
    FC1_N, K_half = w13_packed.shape
    K_out, I_half = w2_packed.shape
    if K_half * 2 != K_in:
        raise ValueError(f"w13 K/2={K_half} mismatches K_in={K_in}")
    I = I_half * 2
    K_in_blocks = K_in // 32
    I_k_blocks = I // 32
    FC1_N_tiles = FC1_N // 8
    K_out_n_tiles = K_out // 8

    pad = (16 - M_actual % 16) % 16
    if pad:
        x_e4m3 = torch.cat(
            [x_e4m3, torch.zeros(pad, K_in, dtype=x_e4m3.dtype, device=x_e4m3.device)],
            dim=0,
        )
        x_sf = torch.cat(
            [x_sf, torch.zeros(pad, K_in_blocks, dtype=x_sf.dtype, device=x_sf.device)],
            dim=0,
        )
    M_total = x_e4m3.shape[0]
    M_tiles = M_total // 16

    device = x_e4m3.device
    Y = torch.zeros(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)
    compiled = _compiled_cpasync(M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        _to_cute(x_e4m3,    cutlass.Uint8),
        _to_cute(x_sf,      cutlass.Uint8),
        _to_cute(w13_packed, cutlass.Uint8),
        _to_cute(w13_sf,    cutlass.Uint8),
        _to_cute(w2_packed, cutlass.Uint8),
        _to_cute(w2_sf,     cutlass.Uint8),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    torch.cuda.synchronize()
    out_flat = Y.reshape(M_total, K_out)
    if pad:
        out_flat = out_flat[:M_actual].clone()
    return out_flat


def run_fused_silu_global(
    x_e4m3: torch.Tensor,        # (M, K_in)         uint8
    x_sf: torch.Tensor,          # (M, K_in/32)      uint8
    w13_packed: torch.Tensor,    # (FC1_N, K_in/2)   uint8
    w13_sf: torch.Tensor,        # (FC1_N, K_in/32)  uint8
    w2_packed: torch.Tensor,     # (K_out, I/2)      uint8
    w2_sf: torch.Tensor,         # (K_out, I/32)     uint8
) -> torch.Tensor:
    """Fused single-launch FC1+SwiGLU+quant+FC2 with no host fragment packing.

    Returns (M, K_out) bf16. M is padded internally to a multiple of 16.
    """
    M_actual, K_in = x_e4m3.shape
    FC1_N, K_half = w13_packed.shape
    K_out, I_half = w2_packed.shape
    if K_half * 2 != K_in:
        raise ValueError(f"w13 K/2={K_half} mismatches K_in={K_in}")
    I = I_half * 2
    K_in_blocks = K_in // 32
    I_k_blocks = I // 32
    FC1_N_tiles = FC1_N // 8
    K_out_n_tiles = K_out // 8

    # Pad M to multiple of 16.
    pad = (16 - M_actual % 16) % 16
    if pad:
        x_e4m3 = torch.cat(
            [x_e4m3, torch.zeros(pad, K_in, dtype=x_e4m3.dtype, device=x_e4m3.device)],
            dim=0,
        )
        x_sf = torch.cat(
            [x_sf, torch.zeros(pad, K_in_blocks, dtype=x_sf.dtype, device=x_sf.device)],
            dim=0,
        )
    M_total = x_e4m3.shape[0]
    M_tiles = M_total // 16

    device = x_e4m3.device
    Y = torch.zeros(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)
    compiled = _compiled_global(M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        _to_cute(x_e4m3,    cutlass.Uint8),
        _to_cute(x_sf,      cutlass.Uint8),
        _to_cute(w13_packed, cutlass.Uint8),
        _to_cute(w13_sf,    cutlass.Uint8),
        _to_cute(w2_packed, cutlass.Uint8),
        _to_cute(w2_sf,     cutlass.Uint8),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    torch.cuda.synchronize()
    out_flat = Y.reshape(M_total, K_out)
    if pad:
        out_flat = out_flat[:M_actual].clone()
    return out_flat


def _make_kernel(
    K_in_blocks: int,
    I_k_blocks: int,
    FC1_N_tiles: int,
    K_out_n_tiles: int,
):
    """Build a fused kernel specialized for the given shape constants."""
    I_n_tiles = I_k_blocks * N_TILES_PER_K_BLOCK
    if FC1_N_tiles != 2 * I_n_tiles:
        raise ValueError(
            f"SwiGLU requires FC1_N_tiles == 2 * I_n_tiles "
            f"(= 2 * {I_n_tiles} = {2 * I_n_tiles}), got {FC1_N_tiles}"
        )

    @cute.kernel
    def kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY):
        cta = cute.arch.block_idx()[0]      # M-tile index in [0, M_tiles)
        tidx = cute.arch.thread_idx()[0]
        g = Int32(tidx) // Int32(4)
        lane = Int32(tidx) % Int32(4)

        # ------------------------------------------------------------------
        # SMEM allocation: intermediate matrix (M, I_k_blocks * 32) bytes
        # ------------------------------------------------------------------
        smem = utils.SmemAllocator()
        I_cols = I_k_blocks * 32

        @cute.struct
        class Storage:
            sInt: cute.struct.MemRange[Uint8, M * I_cols]

        storage = smem.allocate(Storage)
        sInt = storage.sInt.get_tensor(cute.make_layout((M, I_cols)))

        # ------------------------------------------------------------------
        # FC1 K-loop: accumulate FC1_N_tiles tiles of m16n8 D fragments.
        # ------------------------------------------------------------------
        fc1 = [Float32(0.0) for _ in range(FC1_N_tiles * 4)]

        for k in cutlass.range_constexpr(K_in_blocks):
            a0 = mA[cta, tidx, k, 0]
            a1 = mA[cta, tidx, k, 1]
            a2 = mA[cta, tidx, k, 2]
            a3 = mA[cta, tidx, k, 3]
            sfa = mA_sf[cta, tidx, k]
            for n in cutlass.range_constexpr(FC1_N_tiles):
                bp = mW13[tidx, n, k, 0]
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)
                sfb = mW13_sf[tidx, n, k]
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    fc1[n * 4 + 0], fc1[n * 4 + 1], fc1[n * 4 + 2], fc1[n * 4 + 3],
                    a0, a1, a2, a3, b0, b1, sfa, sfb,
                )
                fc1[n * 4 + 0] = d0
                fc1[n * 4 + 1] = d1
                fc1[n * 4 + 2] = d2
                fc1[n * 4 + 3] = d3

        # ------------------------------------------------------------------
        # SwiGLU: intermediate[r, j] = silu(up[r, j]) * gate[r, j]
        # gate is the first I_n_tiles N-tiles, up is the next I_n_tiles.
        # ------------------------------------------------------------------
        intermediate = [Float32(0.0) for _ in range(I_n_tiles * 4)]
        for n in cutlass.range_constexpr(I_n_tiles):
            for i in cutlass.range_constexpr(4):
                gate = fc1[n * 4 + i]
                up = fc1[(n + I_n_tiles) * 4 + i]
                sigmoid_up = Float32(1.0) / (Float32(1.0) + cute.math.exp(-up, fastmath=True))
                intermediate[n * 4 + i] = gate * up * sigmoid_up

        # ------------------------------------------------------------------
        # Per-K-block (block-32) amax → UE8M0 → quantized E4M3 byte
        # Per row group (top: rows 0..7 via d0/d1; bot: rows 8..15 via d2/d3).
        # ------------------------------------------------------------------
        e4m3_bytes = [Uint32(0) for _ in range(I_n_tiles * 4)]
        sf_top_per_kb = [Int32(127) for _ in range(I_k_blocks)]
        sf_bot_per_kb = [Int32(127) for _ in range(I_k_blocks)]

        inv_max = Float32(1.0) / Float32(448.0)
        tiny = Float32(1.401298464e-45)

        for kb in cutlass.range_constexpr(I_k_blocks):
            n_start = kb * N_TILES_PER_K_BLOCK
            local_max_top = Float32(0.0)
            local_max_bot = Float32(0.0)
            for sub in cutlass.range_constexpr(N_TILES_PER_K_BLOCK):
                n = n_start + sub
                for i in cutlass.range_constexpr(2):
                    v_top = intermediate[n * 4 + i]
                    local_max_top = cute.arch.fmax(
                        local_max_top, cute.arch.fmax(v_top, -v_top)
                    )
                    v_bot = intermediate[n * 4 + 2 + i]
                    local_max_bot = cute.arch.fmax(
                        local_max_bot, cute.arch.fmax(v_bot, -v_bot)
                    )

            row_top = warp_reduce(
                local_max_top, lambda a, b: cute.arch.fmax(a, b), width=4,
            )
            row_bot = warp_reduce(
                local_max_bot, lambda a, b: cute.arch.fmax(a, b), width=4,
            )

            safe_top = cute.arch.fmax(row_top * inv_max, tiny)
            safe_bot = cute.arch.fmax(row_bot * inv_max, tiny)
            log_top = cute.math.log2(safe_top, fastmath=True)
            log_bot = cute.math.log2(safe_bot, fastmath=True)
            ceil_top = -floor_f32_to_s32(-log_top)
            ceil_bot = -floor_f32_to_s32(-log_bot)
            sf_top = imin_s32(Int32(254), imax_s32(Int32(0), ceil_top + Int32(127)))
            sf_bot = imin_s32(Int32(254), imax_s32(Int32(0), ceil_bot + Int32(127)))
            sf_top_per_kb[kb] = sf_top
            sf_bot_per_kb[kb] = sf_bot

            inv_scale_top = (
                Float32(1.0) / cute.math.exp2(Float32(sf_top - Int32(127)), fastmath=True)
            )
            inv_scale_bot = (
                Float32(1.0) / cute.math.exp2(Float32(sf_bot - Int32(127)), fastmath=True)
            )

            for sub in cutlass.range_constexpr(N_TILES_PER_K_BLOCK):
                n = n_start + sub
                for i in cutlass.range_constexpr(2):
                    e4m3_bytes[n * 4 + i] = cvt_f32_to_e4m3(
                        intermediate[n * 4 + i] * inv_scale_top
                    )
                    e4m3_bytes[n * 4 + 2 + i] = cvt_f32_to_e4m3(
                        intermediate[n * 4 + 2 + i] * inv_scale_bot
                    )

        # ------------------------------------------------------------------
        # Write quantized intermediate to SMEM at row-major (M, I_cols).
        # Thread (g, l) holds 16 e4m3 bytes per (g, g+8) at cols
        #   col = lane*2 + n*8 + i  for n ∈ [0, I_n_tiles), i ∈ {0, 1}
        # ------------------------------------------------------------------
        for n in cutlass.range_constexpr(I_n_tiles):
            for i in cutlass.range_constexpr(2):
                col = lane * Int32(2) + Int32(n * 8) + Int32(i)
                sInt[g,            col] = Uint8(e4m3_bytes[n * 4 + i])
                sInt[g + Int32(8), col] = Uint8(e4m3_bytes[n * 4 + 2 + i])

        cute.arch.sync_warp()

        # ------------------------------------------------------------------
        # FC2 K-loop: for each I-K-block, read A fragment from SMEM and run
        # one MMA per output N-tile.
        # ------------------------------------------------------------------
        fc2 = [Float32(0.0) for _ in range(K_out_n_tiles * 4)]

        for kb_fc2 in cutlass.range_constexpr(I_k_blocks):
            col_off = Int32(kb_fc2 * 32)
            fc2_a0 = Uint32(0)
            fc2_a1 = Uint32(0)
            fc2_a2 = Uint32(0)
            fc2_a3 = Uint32(0)
            for i in cutlass.range_constexpr(4):
                col_lo = col_off + lane * Int32(8) + Int32(i)
                col_hi = col_off + lane * Int32(8) + Int32(4 + i)
                fc2_a0 = fc2_a0 | (Uint32(sInt[g,            col_lo]) << Uint32(i * 8))
                fc2_a1 = fc2_a1 | (Uint32(sInt[g + Int32(8), col_lo]) << Uint32(i * 8))
                fc2_a2 = fc2_a2 | (Uint32(sInt[g,            col_hi]) << Uint32(i * 8))
                fc2_a3 = fc2_a3 | (Uint32(sInt[g + Int32(8), col_hi]) << Uint32(i * 8))

            sfa_fc2 = Uint32(0x7F)
            if lane == Int32(0):
                sfa_fc2 = Uint32(sf_top_per_kb[kb_fc2])
            elif lane == Int32(1):
                sfa_fc2 = Uint32(sf_bot_per_kb[kb_fc2])

            for n in cutlass.range_constexpr(K_out_n_tiles):
                bp = mW2[tidx, n, kb_fc2, 0]
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)
                sfb_fc2 = mW2_sf[tidx, n, kb_fc2]
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    fc2[n * 4 + 0], fc2[n * 4 + 1], fc2[n * 4 + 2], fc2[n * 4 + 3],
                    fc2_a0, fc2_a1, fc2_a2, fc2_a3, b0, b1, sfa_fc2, sfb_fc2,
                )
                fc2[n * 4 + 0] = d0
                fc2[n * 4 + 1] = d1
                fc2[n * 4 + 2] = d2
                fc2[n * 4 + 3] = d3

        # ------------------------------------------------------------------
        # Write FC2 D fragments to global as bf16 (M_tiles, 16, K_out)
        # ------------------------------------------------------------------
        for n in cutlass.range_constexpr(K_out_n_tiles):
            col0 = Int32(n * 8) + lane * Int32(2)
            mY[cta, g,            col0]            = BFloat16(fc2[n * 4 + 0])
            mY[cta, g,            col0 + Int32(1)] = BFloat16(fc2[n * 4 + 1])
            mY[cta, g + Int32(8), col0]            = BFloat16(fc2[n * 4 + 2])
            mY[cta, g + Int32(8), col0 + Int32(1)] = BFloat16(fc2[n * 4 + 3])

    @cute.jit
    def driver(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY, stream):
        M_tiles = mA.shape[0]
        kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY).launch(
            grid=(M_tiles, 1, 1), block=[32, 1, 1], stream=stream,
        )

    return driver


_compile_cache_fused = {}


def _compiled_fused(M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles):
    # cute.compile bakes in tensor shapes, so M_tiles is part of the key.
    key = (M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles)
    if key in _compile_cache_fused:
        return _compile_cache_fused[key]
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    A = torch.zeros(M_tiles, 32, K_in_blocks, 4, dtype=torch.int32, device=device)
    A_sf = torch.zeros(M_tiles, 32, K_in_blocks, dtype=torch.int32, device=device)
    W13 = torch.zeros(32, FC1_N_tiles, K_in_blocks, 1, dtype=torch.int32, device=device)
    W13_sf = torch.zeros(32, FC1_N_tiles, K_in_blocks, dtype=torch.int32, device=device)
    W2 = torch.zeros(32, K_out_n_tiles, I_k_blocks, 1, dtype=torch.int32, device=device)
    W2_sf = torch.zeros(32, K_out_n_tiles, I_k_blocks, dtype=torch.int32, device=device)
    Y = torch.zeros(M_tiles, 16, K_out_n_tiles * 8, dtype=torch.bfloat16, device=device)
    driver = _make_kernel(K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles)
    compiled = cute.compile(
        driver,
        _to_cute(A, cutlass.Uint32),
        _to_cute(A_sf, cutlass.Uint32),
        _to_cute(W13, cutlass.Uint32),
        _to_cute(W13_sf, cutlass.Uint32),
        _to_cute(W2, cutlass.Uint32),
        _to_cute(W2_sf, cutlass.Uint32),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    _compile_cache_fused[key] = compiled
    return compiled


def run_fused_silu(
    A_frag: torch.Tensor,    # (M_tiles, 32, K_in_blocks, 4)            i32
    A_sf_frag: torch.Tensor, # (M_tiles, 32, K_in_blocks)               i32
    W13_frag: torch.Tensor,  # (32, FC1_N_tiles, K_in_blocks, 1)        i32
    W13_sf_frag: torch.Tensor,
    W2_frag: torch.Tensor,   # (32, K_out_n_tiles, I_k_blocks, 1)       i32
    W2_sf_frag: torch.Tensor,
) -> torch.Tensor:
    """Run the fused single-kernel FC1+SwiGLU+quant+FC2 across M_tiles CTAs.

    Returns (M_tiles, 16, K_out) bf16; reshape to (M_total, K_out) if needed.
    """
    if A_frag.dim() == 3:
        # Convenience: auto-promote single-tile (32, K_in_blocks, 4) to (1, ...).
        A_frag = A_frag.unsqueeze(0)
        A_sf_frag = A_sf_frag.unsqueeze(0)
    M_tiles, _, K_in_blocks, _ = A_frag.shape
    FC1_N_tiles = W13_frag.shape[1]
    K_out_n_tiles = W2_frag.shape[1]
    I_k_blocks = W2_frag.shape[2]
    if W13_frag.shape[2] != K_in_blocks:
        raise ValueError(
            f"W13 K dim {W13_frag.shape[2]} mismatches A K dim {K_in_blocks}"
        )
    if W2_sf_frag.shape != (32, K_out_n_tiles, I_k_blocks):
        raise ValueError(f"unexpected W2_sf shape {W2_sf_frag.shape}")
    if A_sf_frag.shape != (M_tiles, 32, K_in_blocks):
        raise ValueError(f"A_sf_frag shape {A_sf_frag.shape} mismatches A_frag")

    device = A_frag.device
    Y = torch.zeros(M_tiles, 16, K_out_n_tiles * 8, dtype=torch.bfloat16, device=device)
    compiled = _compiled_fused(M_tiles, K_in_blocks, I_k_blocks, FC1_N_tiles, K_out_n_tiles)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        _to_cute(A_frag, cutlass.Uint32),
        _to_cute(A_sf_frag, cutlass.Uint32),
        _to_cute(W13_frag, cutlass.Uint32),
        _to_cute(W13_sf_frag, cutlass.Uint32),
        _to_cute(W2_frag, cutlass.Uint32),
        _to_cute(W2_sf_frag, cutlass.Uint32),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    torch.cuda.synchronize()
    return Y


# Backwards-compatible aliases for the M=16 fixed-shape API used in tests.
FC1_N_TILES = 8
K_OUT_N_TILES = 4


def run_fused_silu_M16(
    A_frag: torch.Tensor,
    A_sf_frag: torch.Tensor,
    W13_frag: torch.Tensor,
    W13_sf_frag: torch.Tensor,
    W2_frag: torch.Tensor,
    W2_sf_frag: torch.Tensor,
) -> torch.Tensor:
    """Legacy entry. Promotes single-tile A frag and unwraps the M_tiles=1 output."""
    out = run_fused_silu(
        A_frag, A_sf_frag, W13_frag, W13_sf_frag, W2_frag, W2_sf_frag,
    )
    return out.reshape(-1, out.shape[-1])[:16]


def run_fused_silu_full(
    x_e4m3: torch.Tensor,        # (M, K_in)         e4m3 byte (uint8)
    x_sf: torch.Tensor,          # (M, K_in/32)      ue8m0   (uint8)
    w13_packed: torch.Tensor,    # (FC1_N, K_in/2)   packed e2m1 (uint8)
    w13_sf: torch.Tensor,        # (FC1_N, K_in/32)  ue8m0       (uint8)
    w2_packed: torch.Tensor,     # (K_out, I/2)      packed e2m1 (uint8)
    w2_sf: torch.Tensor,         # (K_out, I/32)     ue8m0       (uint8)
) -> torch.Tensor:
    """High-level entry: full-batch (M, K_in) inputs → (M, K_out) bf16 output.

    M is padded up to a multiple of 16 internally; output is sliced back. All
    per-CTA fragment packing is done host-side here; the kernel runs M/16 CTAs
    in a single launch over shared weights.
    """
    from b12x.moe.fused.mxfp4_mxfp8.single_tile import (
        pack_a_fragments, pack_b_fragments, pack_sfa, pack_sfb,
    )

    M_actual, K_in = x_e4m3.shape
    FC1_N, K_half = w13_packed.shape
    K_out, I_half = w2_packed.shape
    if K_half * 2 != K_in:
        raise ValueError(f"w13 K/2={K_half} mismatches K_in={K_in}")
    I = I_half * 2

    # Pad M up to next multiple of 16.
    pad = (16 - M_actual % 16) % 16
    if pad:
        x_e4m3 = torch.cat(
            [x_e4m3, torch.zeros(pad, K_in, dtype=x_e4m3.dtype, device=x_e4m3.device)],
            dim=0,
        )
        x_sf = torch.cat(
            [x_sf, torch.zeros(pad, K_in // 32, dtype=x_sf.dtype, device=x_sf.device)],
            dim=0,
        )
    M_total = x_e4m3.shape[0]
    M_tiles = M_total // 16

    # Per-CTA A fragments via pack_a_fragments on each 16-row slice.
    A_frag_per_cta = []
    A_sf_frag_per_cta = []
    for c in range(M_tiles):
        rows = slice(c * 16, (c + 1) * 16)
        A_frag_per_cta.append(pack_a_fragments(x_e4m3[rows]).unsqueeze(0))
        A_sf_frag_per_cta.append(pack_sfa(x_sf[rows]).unsqueeze(0))
    A_frag = torch.cat(A_frag_per_cta, dim=0)        # (M_tiles, 32, K_in/32, 4)
    A_sf_frag = torch.cat(A_sf_frag_per_cta, dim=0)  # (M_tiles, 32, K_in/32)

    # Weights are shared across CTAs — pack once, per N-tile.
    def _pack_w_per_tile(w_packed, w_sf, n_dim, k_dim):
        K_blocks = k_dim // 32
        N_tiles = n_dim // 8
        frags = torch.empty(32, N_tiles, K_blocks, 1,
                            dtype=torch.int32, device=w_packed.device)
        sf_frags = torch.empty(32, N_tiles, K_blocks,
                               dtype=torch.int32, device=w_packed.device)
        for n_idx, n_off in enumerate(range(0, n_dim, 8)):
            frags[:, n_idx] = pack_b_fragments(w_packed[n_off:n_off + 8])
            sf_frags[:, n_idx] = pack_sfb(w_sf[n_off:n_off + 8])
        return frags, sf_frags

    W13_frag, W13_sf_frag = _pack_w_per_tile(w13_packed, w13_sf, FC1_N, K_in)
    W2_frag, W2_sf_frag = _pack_w_per_tile(w2_packed, w2_sf, K_out, I)

    out = run_fused_silu(
        A_frag, A_sf_frag, W13_frag, W13_sf_frag, W2_frag, W2_sf_frag,
    )                                              # (M_tiles, 16, K_out)
    out_flat = out.reshape(M_total, K_out)
    if pad:
        out_flat = out_flat[:M_actual].clone()
    return out_flat


# =============================================================================
# Chunked single-warp kernel — unblocks arbitrary FC1_N and K_out
# =============================================================================


def _make_kernel_chunked(
    K_in_blocks: int,
    I_k_blocks: int,
    K_out_n_tiles: int,
    fc2_chunk_n_tiles: int = 4,
):
    """Single-warp single-CTA fused kernel that chunks FC1 and FC2 N-tiles.

    Unlike `_make_kernel_cpasync` (which holds ALL FC1 D fragments in registers
    simultaneously — limits FC1_N_tiles ≤ ~16 before spilling), this variant
    processes FC1 N-tiles in chunks of 4 (one intermediate K-block worth),
    and FC2 N-tiles in chunks of `fc2_chunk_n_tiles`. Each chunk's accumulator
    is its own register state, freed before the next chunk begins.

    FC1 chunk geometry (chunk = 1 intermediate K-block of 32 cols):
      - 4 gate N-tiles + 4 up N-tiles per chunk
      - per-thread fc1 accumulator: 8 N-tiles × 4 fp32 = 32 fp32

    FC2 chunk geometry (chunk = `fc2_chunk_n_tiles` × 8 cols of K_out):
      - per-thread fc2 accumulator: fc2_chunk_n_tiles × 4 fp32

    Constraints (validated up front):
      - K_out_n_tiles % fc2_chunk_n_tiles == 0
      - I_k_blocks ≥ 1 (each = 1 FC1 chunk)
      - W13 K-block size for chunk: 8 N-tiles × 8 rows/tile × 16 bytes = 1024 bytes
        (= 32 cp.async/16B per chunk per K-iter; 32 threads × 1 cp.async/thread)
      - W2 K-block size for chunk: fc2_chunk_n_tiles × 8 × 16 bytes
    """
    if K_out_n_tiles % fc2_chunk_n_tiles != 0:
        raise ValueError(
            f"K_out_n_tiles ({K_out_n_tiles}) must be divisible by "
            f"fc2_chunk_n_tiles ({fc2_chunk_n_tiles})"
        )
    fc2_n_chunks = K_out_n_tiles // fc2_chunk_n_tiles
    I_n_tiles = I_k_blocks * N_TILES_PER_K_BLOCK  # = 4 * I_k_blocks

    # FC1 chunk: 4 gate + 4 up tiles = 8 tiles, 8 rows/tile = 64 W13 rows per chunk per K-iter.
    # Each row is 16 bytes (1 K-block packed FP4) → 1024 bytes per chunk per K-iter.
    # 64 cp.async-16B issues = 32 threads × 2 each.
    FC1_W13_ROWS_PER_CHUNK = 8 * 8           # 64 rows
    FC1_W13_CHUNK_BYTES = FC1_W13_ROWS_PER_CHUNK * 16   # 1024 bytes
    FC1_W13_LOADS_PER_THREAD = FC1_W13_ROWS_PER_CHUNK // 32   # 2

    FC2_W2_ROWS_PER_CHUNK = fc2_chunk_n_tiles * 8
    FC2_W2_CHUNK_BYTES = FC2_W2_ROWS_PER_CHUNK * 16
    if FC2_W2_ROWS_PER_CHUNK % 32 != 0:
        raise ValueError(
            f"fc2_chunk_n_tiles ({fc2_chunk_n_tiles}) × 8 = {FC2_W2_ROWS_PER_CHUNK} "
            "must be divisible by 32 for clean cp.async distribution"
        )
    FC2_W2_LOADS_PER_THREAD = FC2_W2_ROWS_PER_CHUNK // 32

    @cute.kernel
    def kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY):
        cta = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        g = Int32(tidx) // Int32(4)
        lane = Int32(tidx) % Int32(4)

        row_top = cta * Int32(16) + g
        row_bot = cta * Int32(16) + g + Int32(8)

        mA_u32 = cute.recast_tensor(mA, cutlass.Uint32)

        smem = utils.SmemAllocator()
        I_cols = I_k_blocks * 32
        # Re-bind factory closures as kernel-local names so @cute.struct sees them.
        W13_CHUNK_BYTES_LOCAL = FC1_W13_CHUNK_BYTES
        W2_CHUNK_BYTES_LOCAL  = FC2_W2_CHUNK_BYTES

        @cute.struct
        class Storage:
            sInt: cute.struct.MemRange[Uint8, M * I_cols]
            sInt_sf: cute.struct.MemRange[Uint8, M * I_k_blocks]
            sW13_kb: cute.struct.MemRange[Uint8, W13_CHUNK_BYTES_LOCAL]
            sW2_kb:  cute.struct.MemRange[Uint8, W2_CHUNK_BYTES_LOCAL]

        storage = smem.allocate(Storage)
        sInt_addr = shared_ptr_to_u32(storage.sInt.data_ptr())
        sInt_sf_addr = shared_ptr_to_u32(storage.sInt_sf.data_ptr())
        sW13_addr = shared_ptr_to_u32(storage.sW13_kb.data_ptr())
        sW2_addr  = shared_ptr_to_u32(storage.sW2_kb.data_ptr())

        K_half = K_in_blocks * 16
        I_half = I_k_blocks * 16

        # ------------------------------------------------------------------
        # Outer FC1 chunk loop: each chunk = 1 intermediate K-block (32 cols).
        # Runtime loop (cutlass.range, unroll=1) keeps generated IR small at
        # large I_k_blocks (e.g. gpt-oss-20b's 90). Each iteration resets the
        # `fc1` accumulator from zeros, so no inter-iteration carry is needed.
        # The inner K loop stays constexpr-unrolled because its body mutates a
        # Python list of Float32 SSA values, a pattern that only works under
        # full unroll.
        # ------------------------------------------------------------------
        for chunk in cutlass.range(I_k_blocks, unroll=1):
            # Per-chunk fc1 accumulator: 4 gate + 4 up tiles, 4 fp32 each.
            fc1 = [Float32(0.0) for _ in range(8 * 4)]

            chunk_x32 = chunk * Int32(32)

            for k in cutlass.range_constexpr(K_in_blocks):
                # Load this chunk's W13 K-block portion (4 gate + 4 up rows × 8
                # rows/tile = 64 total rows × 16 bytes = 1024 bytes).
                # Layout in SMEM: rows 0..31 = gate (4 tiles × 8 rows),
                #                 rows 32..63 = up   (4 tiles × 8 rows).
                for i in cutlass.range_constexpr(FC1_W13_LOADS_PER_THREAD):
                    # i=0 → gate side; i=1 → up side. `chunk` is a runtime IV,
                    # so build the row base via Int32 arithmetic (not an Int32
                    # cast of a runtime expression). Use a ternary so the DSL
                    # AST preprocessor recognizes this as constexpr branching.
                    gmem_row_load = (
                        chunk_x32 + Int32(tidx)
                    ) if i == 0 else (
                        chunk_x32 + Int32(I_n_tiles * 8) + Int32(tidx)
                    )
                    smem_row = Int32(i * 32) + Int32(tidx)
                    src_addr = get_ptr_as_int64(
                        mW13, gmem_row_load * Int32(K_half) + Int32(k * 16),
                    )
                    dst_addr = sW13_addr + smem_row * Int32(16)
                    _cp_async_16B(dst_addr, src_addr)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_warp()

                # A fragment from row-major global.
                col_a_u32 = Int32(k * 8) + lane * Int32(2)
                a0 = Uint32(mA_u32[row_top, col_a_u32 + Int32(0)])
                a1 = Uint32(mA_u32[row_bot, col_a_u32 + Int32(0)])
                a2 = Uint32(mA_u32[row_top, col_a_u32 + Int32(1)])
                a3 = Uint32(mA_u32[row_bot, col_a_u32 + Int32(1)])

                sfa = Uint32(0x7F)
                if lane == Int32(0):
                    sfa = Uint32(mA_sf[row_top, k])
                elif lane == Int32(1):
                    sfa = Uint32(mA_sf[row_bot, k])

                # Inner: 4 gate tiles, then 4 up tiles. Each MMA reads its
                # bp from SMEM at the staged location.
                for n in cutlass.range_constexpr(8):  # 0..3 = gate, 4..7 = up
                    smem_wrow = Int32(n * 8) + g     # row inside chunk SMEM (0..63)
                    bp_addr = sW13_addr + smem_wrow * Int32(16) + lane * Int32(4)
                    bp = Uint32(ld_shared_i32(bp_addr))
                    b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

                    # SFB row index for this N-tile within the W13 matrix:
                    #   n<4 (gate): chunk*32 + n*8 + g
                    #   n>=4 (up):  I_n_tiles*8 + chunk*32 + (n-4)*8 + g
                    # Ternary form so the DSL preprocessor recognizes this as
                    # constexpr branching (constexpr `n` is checked at trace time).
                    gmem_wrow_sfb = (
                        chunk_x32 + Int32(n * 8) + g
                    ) if n < 4 else (
                        chunk_x32 + Int32(I_n_tiles * 8 + (n - 4) * 8) + g
                    )
                    sfb = Uint32(0x7F)
                    if lane == Int32(0):
                        sfb = Uint32(mW13_sf[gmem_wrow_sfb, k])

                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        fc1[n * 4 + 0], fc1[n * 4 + 1], fc1[n * 4 + 2], fc1[n * 4 + 3],
                        a0, a1, a2, a3, b0, b1, sfa, sfb,
                    )
                    fc1[n * 4 + 0] = d0
                    fc1[n * 4 + 1] = d1
                    fc1[n * 4 + 2] = d2
                    fc1[n * 4 + 3] = d3

            # SwiGLU + per-row amax + quant for this chunk's 32 cols.
            intermediate = [Float32(0.0) for _ in range(4 * 4)]   # 4 tiles × 4 fp32
            for n in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(4):
                    gate = fc1[n * 4 + i]
                    up   = fc1[(n + 4) * 4 + i]
                    sigmoid_up = Float32(1.0) / (Float32(1.0) + cute.math.exp(-up, fastmath=True))
                    intermediate[n * 4 + i] = gate * up * sigmoid_up

            local_max_top = Float32(0.0)
            local_max_bot = Float32(0.0)
            for n in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(2):
                    v_top = intermediate[n * 4 + i]
                    local_max_top = cute.arch.fmax(local_max_top, cute.arch.fmax(v_top, -v_top))
                    v_bot = intermediate[n * 4 + 2 + i]
                    local_max_bot = cute.arch.fmax(local_max_bot, cute.arch.fmax(v_bot, -v_bot))
            row_top_amax = warp_reduce(local_max_top, lambda a, b: cute.arch.fmax(a, b), width=4)
            row_bot_amax = warp_reduce(local_max_bot, lambda a, b: cute.arch.fmax(a, b), width=4)

            inv_max = Float32(1.0) / Float32(448.0)
            tiny = Float32(1.401298464e-45)
            safe_top = cute.arch.fmax(row_top_amax * inv_max, tiny)
            safe_bot = cute.arch.fmax(row_bot_amax * inv_max, tiny)
            log_top = cute.math.log2(safe_top, fastmath=True)
            log_bot = cute.math.log2(safe_bot, fastmath=True)
            ceil_top = -floor_f32_to_s32(-log_top)
            ceil_bot = -floor_f32_to_s32(-log_bot)
            sf_top = imin_s32(Int32(254), imax_s32(Int32(0), ceil_top + Int32(127)))
            sf_bot = imin_s32(Int32(254), imax_s32(Int32(0), ceil_bot + Int32(127)))
            inv_scale_top = Float32(1.0) / cute.math.exp2(Float32(sf_top - Int32(127)), fastmath=True)
            inv_scale_bot = Float32(1.0) / cute.math.exp2(Float32(sf_bot - Int32(127)), fastmath=True)

            # Quantize and write E4M3 bytes to sInt at chunk's 32 cols.
            for n in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(2):
                    e4m3_top = cvt_f32_to_e4m3(intermediate[n * 4 + i] * inv_scale_top)
                    e4m3_bot = cvt_f32_to_e4m3(intermediate[n * 4 + 2 + i] * inv_scale_bot)
                    col = chunk_x32 + lane * Int32(2) + Int32(n * 8) + Int32(i)
                    top_smem_addr = sInt_addr + g * Int32(I_cols) + col
                    bot_smem_addr = sInt_addr + (g + Int32(8)) * Int32(I_cols) + col
                    st_shared_u8(top_smem_addr, Uint8(e4m3_top))
                    st_shared_u8(bot_smem_addr, Uint8(e4m3_bot))

            # Per-row sf for this chunk's K-block: lane 0 writes top row's,
            # lane 1 writes bottom row's. `chunk` is a runtime IV — use it
            # directly in the address calculation.
            if lane == Int32(0):
                addr = sInt_sf_addr + g * Int32(I_k_blocks) + chunk
                st_shared_u8(addr, Uint8(sf_top))
            elif lane == Int32(1):
                addr = sInt_sf_addr + (g + Int32(8)) * Int32(I_k_blocks) + chunk
                st_shared_u8(addr, Uint8(sf_bot))

        cute.arch.sync_warp()

        # ------------------------------------------------------------------
        # Outer FC2 chunk loop. Runtime loop (cutlass.range, unroll=1) — see
        # the FC1 chunk-loop comment for why this avoids IR explosion at large
        # K_out / fc2_chunk counts. `fc2` accumulator resets every iteration.
        # ------------------------------------------------------------------
        for fc2_chunk in cutlass.range(fc2_n_chunks, unroll=1):
            # Per-chunk fc2 accumulator.
            fc2 = [Float32(0.0) for _ in range(fc2_chunk_n_tiles * 4)]

            fc2_chunk_row_off = fc2_chunk * Int32(fc2_chunk_n_tiles * 8)

            for kb_fc2 in cutlass.range_constexpr(I_k_blocks):
                # Load this FC2 chunk's W2 K-block portion into sW2_kb.
                for i in cutlass.range_constexpr(FC2_W2_LOADS_PER_THREAD):
                    smem_row = Int32(i * 32) + Int32(tidx)
                    gmem_row = fc2_chunk_row_off + smem_row
                    src_addr = get_ptr_as_int64(
                        mW2, gmem_row * Int32(I_half) + Int32(kb_fc2 * 16),
                    )
                    dst_addr = sW2_addr + smem_row * Int32(16)
                    _cp_async_16B(dst_addr, src_addr)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_warp()

                # FC2 A fragment: read from sInt at this kb_fc2's cols.
                col_off = Int32(kb_fc2 * 32)
                fc2_a0 = Uint32(0); fc2_a1 = Uint32(0)
                fc2_a2 = Uint32(0); fc2_a3 = Uint32(0)
                base_lo_top = sInt_addr + g * Int32(I_cols) + col_off + lane * Int32(8)
                base_lo_bot = sInt_addr + (g + Int32(8)) * Int32(I_cols) + col_off + lane * Int32(8)
                fc2_a0 = Uint32(ld_shared_i32(base_lo_top + Int32(0)))
                fc2_a1 = Uint32(ld_shared_i32(base_lo_bot + Int32(0)))
                fc2_a2 = Uint32(ld_shared_i32(base_lo_top + Int32(4)))
                fc2_a3 = Uint32(ld_shared_i32(base_lo_bot + Int32(4)))

                # FC2 SFA: read from sInt_sf at (g, kb_fc2) for lane 0 / (g+8, kb_fc2) for lane 1.
                # Use byte-aligned load since sInt_sf is per-row × per-K-block bytes
                # (offset isn't necessarily 4-byte aligned for I_k_blocks==1).
                sfa_fc2 = Uint32(0x7F)
                if lane == Int32(0):
                    addr = sInt_sf_addr + g * Int32(I_k_blocks) + Int32(kb_fc2)
                    sfa_fc2 = ld_shared_u8(addr)
                elif lane == Int32(1):
                    addr = sInt_sf_addr + (g + Int32(8)) * Int32(I_k_blocks) + Int32(kb_fc2)
                    sfa_fc2 = ld_shared_u8(addr)

                for n in cutlass.range_constexpr(fc2_chunk_n_tiles):
                    smem_wrow = Int32(n * 8) + g
                    bp_addr_2 = sW2_addr + smem_wrow * Int32(16) + lane * Int32(4)
                    bp = Uint32(ld_shared_i32(bp_addr_2))
                    b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

                    sfb_fc2 = Uint32(0x7F)
                    if lane == Int32(0):
                        gmem_wrow = fc2_chunk_row_off + Int32(n * 8) + g
                        sfb_fc2 = Uint32(mW2_sf[gmem_wrow, kb_fc2])

                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        fc2[n * 4 + 0], fc2[n * 4 + 1], fc2[n * 4 + 2], fc2[n * 4 + 3],
                        fc2_a0, fc2_a1, fc2_a2, fc2_a3, b0, b1, sfa_fc2, sfb_fc2,
                    )
                    fc2[n * 4 + 0] = d0
                    fc2[n * 4 + 1] = d1
                    fc2[n * 4 + 2] = d2
                    fc2[n * 4 + 3] = d3

            # Write D fragments to mY at this FC2 chunk's K_out cols.
            for n in cutlass.range_constexpr(fc2_chunk_n_tiles):
                col0 = fc2_chunk_row_off + Int32(n * 8) + lane * Int32(2)
                mY[cta, g,            col0]            = BFloat16(fc2[n * 4 + 0])
                mY[cta, g,            col0 + Int32(1)] = BFloat16(fc2[n * 4 + 1])
                mY[cta, g + Int32(8), col0]            = BFloat16(fc2[n * 4 + 2])
                mY[cta, g + Int32(8), col0 + Int32(1)] = BFloat16(fc2[n * 4 + 3])

    @cute.jit
    def driver(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY, stream):
        M_tiles = mY.shape[0]
        kernel(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY).launch(
            grid=(M_tiles, 1, 1), block=[32, 1, 1], stream=stream,
        )

    return driver


_compile_cache_chunked = {}


def _compiled_chunked(M_tiles, K_in_blocks, I_k_blocks, K_out_n_tiles, fc2_chunk):
    key = (M_tiles, K_in_blocks, I_k_blocks, K_out_n_tiles, fc2_chunk)
    if key in _compile_cache_chunked:
        return _compile_cache_chunked[key]
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    M_total = M_tiles * 16
    K_in = K_in_blocks * 32
    K_out = K_out_n_tiles * 8
    I = I_k_blocks * 32
    FC1_N = 2 * I
    A = torch.zeros(M_total, K_in, dtype=torch.uint8, device=device)
    A_sf = torch.zeros(M_total, K_in_blocks, dtype=torch.uint8, device=device)
    W13 = torch.zeros(FC1_N, K_in // 2, dtype=torch.uint8, device=device)
    W13_sf = torch.zeros(FC1_N, K_in_blocks, dtype=torch.uint8, device=device)
    W2 = torch.zeros(K_out, I // 2, dtype=torch.uint8, device=device)
    W2_sf = torch.zeros(K_out, I_k_blocks, dtype=torch.uint8, device=device)
    Y = torch.zeros(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)
    driver = _make_kernel_chunked(K_in_blocks, I_k_blocks, K_out_n_tiles, fc2_chunk)
    compiled = cute.compile(
        driver,
        _to_cute(A, cutlass.Uint8),
        _to_cute(A_sf, cutlass.Uint8),
        _to_cute(W13, cutlass.Uint8),
        _to_cute(W13_sf, cutlass.Uint8),
        _to_cute(W2, cutlass.Uint8),
        _to_cute(W2_sf, cutlass.Uint8),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    _compile_cache_chunked[key] = compiled
    return compiled


def run_fused_silu_chunked(
    x_e4m3: torch.Tensor,
    x_sf: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_sf: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_sf: torch.Tensor,
    *,
    fc2_chunk_n_tiles: int = 4,
) -> torch.Tensor:
    """Chunked variant. Same calling shape as run_fused_silu_global / cpasync.

    Handles arbitrary FC1_N = 2*I (chunked over I_k_blocks) and arbitrary
    K_out (chunked over K_out_n_tiles / fc2_chunk_n_tiles). Constraint:
    K_out_n_tiles must be divisible by fc2_chunk_n_tiles.
    """
    M_actual, K_in = x_e4m3.shape
    FC1_N, K_half = w13_packed.shape
    K_out, I_half = w2_packed.shape
    assert K_half * 2 == K_in
    I = I_half * 2
    assert FC1_N == 2 * I
    K_in_blocks = K_in // 32
    I_k_blocks = I // 32
    K_out_n_tiles = K_out // 8

    pad = (16 - M_actual % 16) % 16
    if pad:
        x_e4m3 = torch.cat(
            [x_e4m3, torch.zeros(pad, K_in, dtype=x_e4m3.dtype, device=x_e4m3.device)], dim=0,
        )
        x_sf = torch.cat(
            [x_sf, torch.zeros(pad, K_in_blocks, dtype=x_sf.dtype, device=x_sf.device)], dim=0,
        )
    M_total = x_e4m3.shape[0]
    M_tiles = M_total // 16

    device = x_e4m3.device
    Y = torch.zeros(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)
    compiled = _compiled_chunked(
        M_tiles, K_in_blocks, I_k_blocks, K_out_n_tiles, fc2_chunk_n_tiles,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        _to_cute(x_e4m3, cutlass.Uint8),
        _to_cute(x_sf, cutlass.Uint8),
        _to_cute(w13_packed, cutlass.Uint8),
        _to_cute(w13_sf, cutlass.Uint8),
        _to_cute(w2_packed, cutlass.Uint8),
        _to_cute(w2_sf, cutlass.Uint8),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    torch.cuda.synchronize()
    out_flat = Y.reshape(M_total, K_out)
    if pad:
        out_flat = out_flat[:M_actual].clone()
    return out_flat
