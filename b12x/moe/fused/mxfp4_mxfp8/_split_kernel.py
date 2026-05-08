"""Two-kernel split: FC1+SwiGLU+Quant kernel writes intermediate to HBM,
then FC2 kernel reads it. Both kernels are massively multi-CTA — each CTA
handles ONE (M-tile, chunk) work unit, so the device's SM grid is well
populated even at small M.

Goal: outperform flashinfer at gpt-oss-20b shapes (M=128, K=I=K_out=2880),
where the single-CTA-per-M-tile chunked kernel is ~50× too slow.

Grid layout at gpt-oss-20b (M_tiles=8, I_k_blocks=90, K_out_n_tiles=360):
  - Kernel A grid = (8, 90)   = 720 CTAs (8× more parallelism than chunked)
  - Kernel B grid = (8, 360 // fc2_chunk_n_tiles) = up to 720 CTAs
  - On RTX PRO 6000 Blackwell (~140 SMs), both saturate the device.

Intermediate (between the two kernels) is uint8 e4m3 of size M × I plus
M × I_k_blocks UE8M0 scales. At gpt-oss-20b (M=128, I=2880):
  ~380 KB total round-trip. <1 µs on HBM.
"""
from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x.cute.fp4 import (
    cvt_e2m1x8_to_e4m3x8,
    cvt_f32_to_e4m3,
    get_ptr_as_int64,
    ld_shared_i32,
    mxfp8_mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    st_global_u8,
    warp_reduce,
)
from b12x.moe.fused.mxfp4_mxfp8._quant_kernel import (
    floor_f32_to_s32,
    imax_s32,
    imin_s32,
)


@dsl_user_op
def _cp_async_16B(smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None):
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


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


# ---------------------------------------------------------------------------
# Kernel A: FC1 + SwiGLU + MXFP8 quant. Multi-CTA over (M_tiles, I_k_blocks).
# Each CTA produces one chunk (32 cols of intermediate) for one M-row group.
# ---------------------------------------------------------------------------
def _make_kernel_fc1_silu_quant(K_in_blocks: int, I_k_blocks: int):
    I_n_tiles = I_k_blocks * 4
    FC1_W13_ROWS_PER_CHUNK = 8 * 8           # 4 gate + 4 up tiles × 8 rows
    FC1_W13_CHUNK_BYTES = FC1_W13_ROWS_PER_CHUNK * 16  # 1024
    FC1_W13_LOADS_PER_THREAD = FC1_W13_ROWS_PER_CHUNK // 32  # 2

    @cute.kernel
    def kernel(mA, mA_sf, mW13, mW13_sf, mInt, mInt_sf):
        cta_m = cute.arch.block_idx()[0]
        cta_c = cute.arch.block_idx()[1]
        tidx = cute.arch.thread_idx()[0]
        g = Int32(tidx) // Int32(4)
        lane = Int32(tidx) % Int32(4)

        row_top = cta_m * Int32(16) + g
        row_bot = cta_m * Int32(16) + g + Int32(8)

        mA_u32 = cute.recast_tensor(mA, cutlass.Uint32)

        smem = utils.SmemAllocator()
        W13_CHUNK_BYTES_LOCAL = FC1_W13_CHUNK_BYTES

        @cute.struct
        class Storage:
            sW13_kb: cute.struct.MemRange[Uint8, W13_CHUNK_BYTES_LOCAL]

        storage = smem.allocate(Storage)
        sW13_addr = shared_ptr_to_u32(storage.sW13_kb.data_ptr())

        K_half = K_in_blocks * 16
        chunk_x32 = cta_c * Int32(32)

        fc1 = [Float32(0.0) for _ in range(8 * 4)]

        for k in cutlass.range_constexpr(K_in_blocks):
            # cp.async distributes across all 32 threads of the warp (use tidx,
            # not lane=tidx%4). 32 threads × 2 loads = 64 rows × 16B = 1024B chunk.
            for i in cutlass.range_constexpr(FC1_W13_LOADS_PER_THREAD):
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

            for n in cutlass.range_constexpr(8):
                smem_wrow = Int32(n * 8) + g
                bp_addr = sW13_addr + smem_wrow * Int32(16) + lane * Int32(4)
                bp = Uint32(ld_shared_i32(bp_addr))
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

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

        intermediate = [Float32(0.0) for _ in range(4 * 4)]
        for n in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(4):
                gate = fc1[n * 4 + i]
                up = fc1[(n + 4) * 4 + i]
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

        I_cols = I_k_blocks * 32
        for n in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(2):
                e4m3_top = cvt_f32_to_e4m3(intermediate[n * 4 + i] * inv_scale_top)
                e4m3_bot = cvt_f32_to_e4m3(intermediate[n * 4 + 2 + i] * inv_scale_bot)
                col = chunk_x32 + lane * Int32(2) + Int32(n * 8) + Int32(i)
                top_addr = get_ptr_as_int64(mInt, row_top * Int32(I_cols) + col)
                bot_addr = get_ptr_as_int64(mInt, row_bot * Int32(I_cols) + col)
                st_global_u8(top_addr, Uint8(e4m3_top))
                st_global_u8(bot_addr, Uint8(e4m3_bot))

        if lane == Int32(0):
            sf_addr = get_ptr_as_int64(mInt_sf, row_top * Int32(I_k_blocks) + cta_c)
            st_global_u8(sf_addr, Uint8(sf_top))
        elif lane == Int32(1):
            sf_addr = get_ptr_as_int64(mInt_sf, row_bot * Int32(I_k_blocks) + cta_c)
            st_global_u8(sf_addr, Uint8(sf_bot))

    @cute.jit
    def driver(mA, mA_sf, mW13, mW13_sf, mInt, mInt_sf, stream):
        M_tiles = mInt.shape[0] // 16
        kernel(mA, mA_sf, mW13, mW13_sf, mInt, mInt_sf).launch(
            grid=(M_tiles, I_k_blocks, 1), block=[32, 1, 1], stream=stream,
        )

    return driver


# ---------------------------------------------------------------------------
# Kernel B: FC2. Multi-CTA over (M_tiles, K_out_chunks). Each CTA handles
# fc2_chunk_n_tiles × 8 cols of K_out for one M-row group.
# ---------------------------------------------------------------------------
def _make_kernel_fc2_mcta(I_k_blocks: int, K_out_n_tiles: int, fc2_chunk_n_tiles: int = 4):
    if K_out_n_tiles % fc2_chunk_n_tiles != 0:
        raise ValueError(
            f"K_out_n_tiles ({K_out_n_tiles}) must be divisible by "
            f"fc2_chunk_n_tiles ({fc2_chunk_n_tiles})"
        )
    fc2_n_chunks = K_out_n_tiles // fc2_chunk_n_tiles
    FC2_W2_ROWS_PER_CHUNK = fc2_chunk_n_tiles * 8
    FC2_W2_CHUNK_BYTES = FC2_W2_ROWS_PER_CHUNK * 16
    if FC2_W2_ROWS_PER_CHUNK % 32 != 0:
        raise ValueError(
            f"fc2_chunk_n_tiles*8={FC2_W2_ROWS_PER_CHUNK} must be divisible by 32"
        )
    FC2_W2_LOADS_PER_THREAD = FC2_W2_ROWS_PER_CHUNK // 32

    @cute.kernel
    def kernel(mInt, mInt_sf, mW2, mW2_sf, mY):
        cta_m = cute.arch.block_idx()[0]
        cta_k = cute.arch.block_idx()[1]
        tidx = cute.arch.thread_idx()[0]
        g = Int32(tidx) // Int32(4)
        lane = Int32(tidx) % Int32(4)

        row_top = cta_m * Int32(16) + g
        row_bot = cta_m * Int32(16) + g + Int32(8)

        mInt_u32 = cute.recast_tensor(mInt, cutlass.Uint32)

        smem = utils.SmemAllocator()
        W2_CHUNK_BYTES_LOCAL = FC2_W2_CHUNK_BYTES

        @cute.struct
        class Storage:
            sW2_kb: cute.struct.MemRange[Uint8, W2_CHUNK_BYTES_LOCAL]

        storage = smem.allocate(Storage)
        sW2_addr = shared_ptr_to_u32(storage.sW2_kb.data_ptr())

        I_half = I_k_blocks * 16
        chunk_row_off = cta_k * Int32(fc2_chunk_n_tiles * 8)

        fc2 = [Float32(0.0) for _ in range(fc2_chunk_n_tiles * 4)]

        for kb_fc2 in cutlass.range_constexpr(I_k_blocks):
            for i in cutlass.range_constexpr(FC2_W2_LOADS_PER_THREAD):
                smem_row = Int32(i * 32) + Int32(tidx)
                gmem_row = chunk_row_off + smem_row
                src_addr = get_ptr_as_int64(
                    mW2, gmem_row * Int32(I_half) + Int32(kb_fc2 * 16),
                )
                dst_addr = sW2_addr + smem_row * Int32(16)
                _cp_async_16B(dst_addr, src_addr)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()

            # FC2 A fragment from mInt (global). m16n8k32 with 32 elements per
            # K-block: each thread covers 8 e4m3 elements per (top/bot) row.
            # In u32 view (4 e4m3 per u32), per thread reads 2 u32 per row.
            col_u32_base = kb_fc2 * Int32(8) + lane * Int32(2)
            fc2_a0 = Uint32(mInt_u32[row_top, col_u32_base + Int32(0)])
            fc2_a1 = Uint32(mInt_u32[row_bot, col_u32_base + Int32(0)])
            fc2_a2 = Uint32(mInt_u32[row_top, col_u32_base + Int32(1)])
            fc2_a3 = Uint32(mInt_u32[row_bot, col_u32_base + Int32(1)])

            sfa = Uint32(0x7F)
            if lane == Int32(0):
                sfa = Uint32(mInt_sf[row_top, kb_fc2])
            elif lane == Int32(1):
                sfa = Uint32(mInt_sf[row_bot, kb_fc2])

            for n in cutlass.range_constexpr(fc2_chunk_n_tiles):
                smem_wrow = Int32(n * 8) + g
                bp_addr = sW2_addr + smem_wrow * Int32(16) + lane * Int32(4)
                bp = Uint32(ld_shared_i32(bp_addr))
                b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)

                sfb = Uint32(0x7F)
                if lane == Int32(0):
                    gmem_wrow = chunk_row_off + Int32(n * 8) + g
                    sfb = Uint32(mW2_sf[gmem_wrow, kb_fc2])

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    fc2[n * 4 + 0], fc2[n * 4 + 1], fc2[n * 4 + 2], fc2[n * 4 + 3],
                    fc2_a0, fc2_a1, fc2_a2, fc2_a3, b0, b1, sfa, sfb,
                )
                fc2[n * 4 + 0] = d0
                fc2[n * 4 + 1] = d1
                fc2[n * 4 + 2] = d2
                fc2[n * 4 + 3] = d3

        for n in cutlass.range_constexpr(fc2_chunk_n_tiles):
            col0 = chunk_row_off + Int32(n * 8) + lane * Int32(2)
            mY[cta_m, g,            col0]            = BFloat16(fc2[n * 4 + 0])
            mY[cta_m, g,            col0 + Int32(1)] = BFloat16(fc2[n * 4 + 1])
            mY[cta_m, g + Int32(8), col0]            = BFloat16(fc2[n * 4 + 2])
            mY[cta_m, g + Int32(8), col0 + Int32(1)] = BFloat16(fc2[n * 4 + 3])

    @cute.jit
    def driver(mInt, mInt_sf, mW2, mW2_sf, mY, stream):
        M_tiles = mY.shape[0]
        kernel(mInt, mInt_sf, mW2, mW2_sf, mY).launch(
            grid=(M_tiles, fc2_n_chunks, 1), block=[32, 1, 1], stream=stream,
        )

    return driver


# ---------------------------------------------------------------------------
# Compile caches and Python wrapper.
# ---------------------------------------------------------------------------
_compile_cache_fc1 = {}
_compile_cache_fc2 = {}


def _compiled_fc1(M_tiles, K_in_blocks, I_k_blocks):
    key = (M_tiles, K_in_blocks, I_k_blocks)
    if key in _compile_cache_fc1:
        return _compile_cache_fc1[key]
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    M_total = M_tiles * 16
    K_in = K_in_blocks * 32
    I = I_k_blocks * 32
    FC1_N = 2 * I
    A = torch.zeros(M_total, K_in, dtype=torch.uint8, device=device)
    A_sf = torch.zeros(M_total, K_in_blocks, dtype=torch.uint8, device=device)
    W13 = torch.zeros(FC1_N, K_in // 2, dtype=torch.uint8, device=device)
    W13_sf = torch.zeros(FC1_N, K_in_blocks, dtype=torch.uint8, device=device)
    Int_buf = torch.zeros(M_total, I, dtype=torch.uint8, device=device)
    Int_sf = torch.zeros(M_total, I_k_blocks, dtype=torch.uint8, device=device)
    driver = _make_kernel_fc1_silu_quant(K_in_blocks, I_k_blocks)
    compiled = cute.compile(
        driver,
        _to_cute(A, cutlass.Uint8),
        _to_cute(A_sf, cutlass.Uint8),
        _to_cute(W13, cutlass.Uint8),
        _to_cute(W13_sf, cutlass.Uint8),
        _to_cute(Int_buf, cutlass.Uint8),
        _to_cute(Int_sf, cutlass.Uint8),
        stream,
    )
    _compile_cache_fc1[key] = compiled
    return compiled


def _compiled_fc2(M_tiles, I_k_blocks, K_out_n_tiles, fc2_chunk):
    key = (M_tiles, I_k_blocks, K_out_n_tiles, fc2_chunk)
    if key in _compile_cache_fc2:
        return _compile_cache_fc2[key]
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    M_total = M_tiles * 16
    I = I_k_blocks * 32
    K_out = K_out_n_tiles * 8
    Int_buf = torch.zeros(M_total, I, dtype=torch.uint8, device=device)
    Int_sf = torch.zeros(M_total, I_k_blocks, dtype=torch.uint8, device=device)
    W2 = torch.zeros(K_out, I // 2, dtype=torch.uint8, device=device)
    W2_sf = torch.zeros(K_out, I_k_blocks, dtype=torch.uint8, device=device)
    Y = torch.zeros(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)
    driver = _make_kernel_fc2_mcta(I_k_blocks, K_out_n_tiles, fc2_chunk)
    compiled = cute.compile(
        driver,
        _to_cute(Int_buf, cutlass.Uint8),
        _to_cute(Int_sf, cutlass.Uint8),
        _to_cute(W2, cutlass.Uint8),
        _to_cute(W2_sf, cutlass.Uint8),
        _to_cute(Y, cutlass.BFloat16),
        stream,
    )
    _compile_cache_fc2[key] = compiled
    return compiled


def run_fused_silu_split(
    x_e4m3: torch.Tensor,
    x_sf: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_sf: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_sf: torch.Tensor,
    *,
    fc2_chunk_n_tiles: int = 4,
) -> torch.Tensor:
    """Two-kernel split: FC1+SwiGLU+Quant kernel writes intermediate to HBM,
    then FC2 kernel reads it. Both kernels are massively multi-CTA.

    Same calling shape as run_fused_silu_chunked.
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
    Int_buf = torch.empty(M_total, I, dtype=torch.uint8, device=device)
    Int_sf = torch.empty(M_total, I_k_blocks, dtype=torch.uint8, device=device)
    Y = torch.empty(M_tiles, 16, K_out, dtype=torch.bfloat16, device=device)

    compiled_fc1 = _compiled_fc1(M_tiles, K_in_blocks, I_k_blocks)
    compiled_fc2 = _compiled_fc2(M_tiles, I_k_blocks, K_out_n_tiles, fc2_chunk_n_tiles)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_fc1(
        _to_cute(x_e4m3, cutlass.Uint8),
        _to_cute(x_sf, cutlass.Uint8),
        _to_cute(w13_packed, cutlass.Uint8),
        _to_cute(w13_sf, cutlass.Uint8),
        _to_cute(Int_buf, cutlass.Uint8),
        _to_cute(Int_sf, cutlass.Uint8),
        stream,
    )
    compiled_fc2(
        _to_cute(Int_buf, cutlass.Uint8),
        _to_cute(Int_sf, cutlass.Uint8),
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
