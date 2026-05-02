from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import (
    bfloat2_to_float2_scaled,
    f8f6f4_mma_m16n8k32_f32_e4m3,
    frag_layout_swizzle_16b_to_8b,
    fp8x4_e4m3_to_bfloat2x2,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    st_shared_v4_u32,
)
from b12x.cute.utils import current_cuda_stream

@dataclass(frozen=True)
class _CompileKey:
    m: int
    n: int
    k: int
    a_stride: tuple[int, ...]
    b_stride: tuple[int, ...]
    c_stride: tuple[int, ...]
    threads: int

_COMPILED: dict[_CompileKey, object] = {}
_COMPILED_MMA: dict[_CompileKey, object] = {}

def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor

class _Fp8DenseDotKernel:
    def __init__(self, m: int, n: int, k: int, threads_per_cta: int) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.threads_per_cta = threads_per_cta

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(mA.element_type != cutlass.Float8E4M3FN):
            raise TypeError("A must be Float8E4M3FN")
        if cutlass.const_expr(mB.element_type != cutlass.Float8E4M3FN):
            raise TypeError("B must be Float8E4M3FN")
        if cutlass.const_expr(mScaleA.element_type != cutlass.Float32):
            raise TypeError("scale_a must be Float32")
        if cutlass.const_expr(mScaleB.element_type != cutlass.Float32):
            raise TypeError("scale_b must be Float32")
        if cutlass.const_expr(mC.element_type != cutlass.BFloat16):
            raise TypeError("C must be BFloat16")
        if cutlass.const_expr(mA.shape != (self.m, self.k)):
            raise ValueError("A shape mismatch")
        if cutlass.const_expr(mB.shape != (self.n, self.k)):
            raise ValueError("B shape mismatch")
        if cutlass.const_expr(mC.shape != (self.m, self.n)):
            raise ValueError("C shape mismatch")
        if cutlass.const_expr(self.k % 4 != 0):
            raise ValueError("K must be divisible by 4")

        grid = (cute.ceil_div(self.m * self.n, self.threads_per_cta), 1, 1)
        self.kernel(mA, mB, mScaleA, mScaleB, mC).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mC: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        linear = bidx * self.threads_per_cta + tidx
        total = self.m * self.n

        if linear < total:
            row = linear // self.n
            col = linear - row * self.n
            mAu8 = cute.recast_tensor(mA, cutlass.Uint8)
            mBu8 = cute.recast_tensor(mB, cutlass.Uint8)
            accum = Float32(0.0)
            one = Float32(1.0)

            for k4 in cutlass.range(self.k // 4, unroll=1):
                base = k4 * Int32(4)
                a_pack = (
                    Uint32(mAu8[row, base + Int32(0)])
                    | (Uint32(mAu8[row, base + Int32(1)]) << Uint32(8))
                    | (Uint32(mAu8[row, base + Int32(2)]) << Uint32(16))
                    | (Uint32(mAu8[row, base + Int32(3)]) << Uint32(24))
                )
                b_pack = (
                    Uint32(mBu8[col, base + Int32(0)])
                    | (Uint32(mBu8[col, base + Int32(1)]) << Uint32(8))
                    | (Uint32(mBu8[col, base + Int32(2)]) << Uint32(16))
                    | (Uint32(mBu8[col, base + Int32(3)]) << Uint32(24))
                )
                a_bf2_01, a_bf2_23 = fp8x4_e4m3_to_bfloat2x2(a_pack)
                b_bf2_01, b_bf2_23 = fp8x4_e4m3_to_bfloat2x2(b_pack)
                a0, a1 = bfloat2_to_float2_scaled(a_bf2_01, one)
                a2, a3 = bfloat2_to_float2_scaled(a_bf2_23, one)
                b0, b1 = bfloat2_to_float2_scaled(b_bf2_01, one)
                b2, b3 = bfloat2_to_float2_scaled(b_bf2_23, one)
                accum = accum + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3

            scale = mScaleA[0] * mScaleB[0]
            mC[row, col] = (accum * scale).to(cutlass.BFloat16)

@cute.jit
def _permuted_offset_128b(row_idx, vec_idx, stride_128b):
    return row_idx * stride_128b + (vec_idx ^ (row_idx % 8))

@cute.jit
def _smem_addr_from_b128_offset(base_addr: Int32, offset_128b):
    return base_addr + Int32(offset_128b * 16)

@cute.jit
def _advance_offset_by_row_128b(offset_128b, step_size, row_stride_128b):
    return offset_128b + step_size * row_stride_128b

@cute.jit
def _advance_offset_by_column_128b_2(offset_128b, step_size):
    return (offset_128b + step_size * Int32(2)) ^ Int32(2 * (step_size % 4))

@cute.jit
def _store_mma_16x16_tile(out_acc: cute.Tensor, mC: cute.Tensor, mScaleA: cute.Tensor, mScaleB: cute.Tensor, row_base: Int32, col_base: Int32, lane: Int32):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    scale = mScaleA[0] * mScaleB[0]
    for reg_id in cutlass.range_constexpr(8):
        row_slot = (reg_id % 4) // 2
        row = row_base + lane_group + Int32(8) * row_slot
        col = col_base + lane_pair_base + Int32(8) * (reg_id // 4) + Int32(reg_id % 2)
        mC[row, col] = (out_acc[reg_id] * scale).to(cutlass.BFloat16)

class _Fp8DenseMmaKernel:
    def __init__(self, m: int, n: int, k: int) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.num_threads = 32
        self.k_block = 32

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(mA.element_type != cutlass.Float8E4M3FN):
            raise TypeError("A must be Float8E4M3FN")
        if cutlass.const_expr(mB.element_type != cutlass.Float8E4M3FN):
            raise TypeError("B must be Float8E4M3FN")
        if cutlass.const_expr(mC.element_type != cutlass.BFloat16):
            raise TypeError("C must be BFloat16")
        if cutlass.const_expr(mA.shape != (self.m, self.k)):
            raise ValueError("A shape mismatch")
        if cutlass.const_expr(mB.shape != (self.n, self.k)):
            raise ValueError("B shape mismatch")
        if cutlass.const_expr(mC.shape != (self.m, self.n)):
            raise ValueError("C shape mismatch")
        if cutlass.const_expr(self.m % 16 != 0 or self.n % 32 != 0 or self.k % 32 != 0):
            raise ValueError("M must be divisible by 16, N by 32, and K by 32")

        grid = (self.m // 16, self.n // 32, 1)
        self.kernel(mA, mB, mScaleA, mScaleB, mC).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mC: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        block_m = cute.arch.block_idx()[0]
        block_n = cute.arch.block_idx()[1]
        row_base = block_m * Int32(16)
        col_base = block_n * Int32(32)
        unit_scale = Uint32(0x7F7F7F7F)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(element_type=cutlass.Uint8, layout=cute.make_layout((16, 128), stride=(128, 1)), byte_alignment=128)
        sB = smem.allocate_tensor(element_type=cutlass.Uint8, layout=cute.make_layout((32, 128), stride=(128, 1)), byte_alignment=128)
        a_base_addr = shared_ptr_to_u32(sA.iterator)
        b_base_addr = shared_ptr_to_u32(sB.iterator)
        out_acc0 = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        out_acc1 = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        for reg_id in cutlass.range_constexpr(8):
            out_acc0[reg_id] = Float32(0.0)
            out_acc1[reg_id] = Float32(0.0)

        for k_block in cutlass.range(self.k // 32, unroll=1):
            k_base = k_block * Int32(32)
            linear = lane
            while linear < Int32(16 * 32 // 16):
                row = linear // Int32(2)
                vec_idx = linear - row * Int32(2)
                src_col = k_base + vec_idx * Int32(16)
                dst_addr = a_base_addr + row * Int32(128) + vec_idx * Int32(16)
                a_u8 = cute.recast_tensor(mA, cutlass.Uint8)
                st_shared_v4_u32(
                    dst_addr,
                    Uint32(a_u8[row_base + row, src_col + Int32(0)])
                    | (Uint32(a_u8[row_base + row, src_col + Int32(1)]) << Uint32(8))
                    | (Uint32(a_u8[row_base + row, src_col + Int32(2)]) << Uint32(16))
                    | (Uint32(a_u8[row_base + row, src_col + Int32(3)]) << Uint32(24)),
                    Uint32(a_u8[row_base + row, src_col + Int32(4)])
                    | (Uint32(a_u8[row_base + row, src_col + Int32(5)]) << Uint32(8))
                    | (Uint32(a_u8[row_base + row, src_col + Int32(6)]) << Uint32(16))
                    | (Uint32(a_u8[row_base + row, src_col + Int32(7)]) << Uint32(24)),
                    Uint32(a_u8[row_base + row, src_col + Int32(8)])
                    | (Uint32(a_u8[row_base + row, src_col + Int32(9)]) << Uint32(8))
                    | (Uint32(a_u8[row_base + row, src_col + Int32(10)]) << Uint32(16))
                    | (Uint32(a_u8[row_base + row, src_col + Int32(11)]) << Uint32(24)),
                    Uint32(a_u8[row_base + row, src_col + Int32(12)])
                    | (Uint32(a_u8[row_base + row, src_col + Int32(13)]) << Uint32(8))
                    | (Uint32(a_u8[row_base + row, src_col + Int32(14)]) << Uint32(16))
                    | (Uint32(a_u8[row_base + row, src_col + Int32(15)]) << Uint32(24)),
                )
                linear += Int32(self.num_threads)

            linear = lane
            while linear < Int32(32 * 32 // 16):
                row = linear // Int32(2)
                vec_idx = linear - row * Int32(2)
                src_col = k_base + vec_idx * Int32(16)
                dst_addr = _smem_addr_from_b128_offset(b_base_addr, _permuted_offset_128b(row, vec_idx, Int32(8)))
                b_u8 = cute.recast_tensor(mB, cutlass.Uint8)
                st_shared_v4_u32(
                    dst_addr,
                    Uint32(b_u8[col_base + row, src_col + Int32(0)])
                    | (Uint32(b_u8[col_base + row, src_col + Int32(1)]) << Uint32(8))
                    | (Uint32(b_u8[col_base + row, src_col + Int32(2)]) << Uint32(16))
                    | (Uint32(b_u8[col_base + row, src_col + Int32(3)]) << Uint32(24)),
                    Uint32(b_u8[col_base + row, src_col + Int32(4)])
                    | (Uint32(b_u8[col_base + row, src_col + Int32(5)]) << Uint32(8))
                    | (Uint32(b_u8[col_base + row, src_col + Int32(6)]) << Uint32(16))
                    | (Uint32(b_u8[col_base + row, src_col + Int32(7)]) << Uint32(24)),
                    Uint32(b_u8[col_base + row, src_col + Int32(8)])
                    | (Uint32(b_u8[col_base + row, src_col + Int32(9)]) << Uint32(8))
                    | (Uint32(b_u8[col_base + row, src_col + Int32(10)]) << Uint32(16))
                    | (Uint32(b_u8[col_base + row, src_col + Int32(11)]) << Uint32(24)),
                    Uint32(b_u8[col_base + row, src_col + Int32(12)])
                    | (Uint32(b_u8[col_base + row, src_col + Int32(13)]) << Uint32(8))
                    | (Uint32(b_u8[col_base + row, src_col + Int32(14)]) << Uint32(16))
                    | (Uint32(b_u8[col_base + row, src_col + Int32(15)]) << Uint32(24)),
                )
                linear += Int32(self.num_threads)
            cute.arch.sync_threads()

            a_row = lane // Int32(4)
            a_base_col = (lane % Int32(4)) * Int32(2)
            a_regs = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
            a_regs[0] = (
                Uint32(sA[a_row, a_base_col + Int32(0)])
                | (Uint32(sA[a_row, a_base_col + Int32(1)]) << Int32(8))
                | (Uint32(sA[a_row, a_base_col + Int32(8)]) << Int32(16))
                | (Uint32(sA[a_row, a_base_col + Int32(9)]) << Int32(24))
            )
            a_regs[1] = (
                Uint32(sA[a_row + Int32(8), a_base_col + Int32(0)])
                | (Uint32(sA[a_row + Int32(8), a_base_col + Int32(1)]) << Int32(8))
                | (Uint32(sA[a_row + Int32(8), a_base_col + Int32(8)]) << Int32(16))
                | (Uint32(sA[a_row + Int32(8), a_base_col + Int32(9)]) << Int32(24))
            )
            a_regs[2] = (
                Uint32(sA[a_row, a_base_col + Int32(16)])
                | (Uint32(sA[a_row, a_base_col + Int32(17)]) << Int32(8))
                | (Uint32(sA[a_row, a_base_col + Int32(24)]) << Int32(16))
                | (Uint32(sA[a_row, a_base_col + Int32(25)]) << Int32(24))
            )
            a_regs[3] = (
                Uint32(sA[a_row + Int32(8), a_base_col + Int32(16)])
                | (Uint32(sA[a_row + Int32(8), a_base_col + Int32(17)]) << Int32(8))
                | (Uint32(sA[a_row + Int32(8), a_base_col + Int32(24)]) << Int32(16))
                | (Uint32(sA[a_row + Int32(8), a_base_col + Int32(25)]) << Int32(24))
            )

            b_offset = _permuted_offset_128b(
                lane % Int32(8),
                (lane % Int32(16)) // Int32(8),
                Int32(8),
            ) + Int32(8) * (lane // Int32(16)) * Int32(8)
            b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(_smem_addr_from_b128_offset(b_base_addr, b_offset))
            b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(_smem_addr_from_b128_offset(b_base_addr, b_offset))
            b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)
            b_offset_1 = b_offset + Int32(16 * 8)
            b2_k0, b3_k0 = ldmatrix_m8n8x4_left_half_b16(_smem_addr_from_b128_offset(b_base_addr, b_offset_1))
            b2_k1, b3_k1 = ldmatrix_m8n8x4_right_half_b16(_smem_addr_from_b128_offset(b_base_addr, b_offset_1))
            b2_k0 = frag_layout_swizzle_16b_to_8b(b2_k0)
            b3_k0 = frag_layout_swizzle_16b_to_8b(b3_k0)
            b2_k1 = frag_layout_swizzle_16b_to_8b(b2_k1)
            b3_k1 = frag_layout_swizzle_16b_to_8b(b3_k1)

            d0, d1, d2, d3 = f8f6f4_mma_m16n8k32_f32_e4m3(
                out_acc0[0], out_acc0[1], out_acc0[2], out_acc0[3],
                a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                b0_k0, b0_k1,
            )
            d4, d5, d6, d7 = f8f6f4_mma_m16n8k32_f32_e4m3(
                out_acc0[4], out_acc0[5], out_acc0[6], out_acc0[7],
                a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                b1_k0, b1_k1,
            )
            e0, e1, e2, e3 = f8f6f4_mma_m16n8k32_f32_e4m3(
                out_acc1[0], out_acc1[1], out_acc1[2], out_acc1[3],
                a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                b2_k0, b2_k1,
            )
            e4, e5, e6, e7 = f8f6f4_mma_m16n8k32_f32_e4m3(
                out_acc1[4], out_acc1[5], out_acc1[6], out_acc1[7],
                a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                b3_k0, b3_k1,
            )
            out_acc0[0] = d0
            out_acc0[1] = d1
            out_acc0[2] = d2
            out_acc0[3] = d3
            out_acc0[4] = d4
            out_acc0[5] = d5
            out_acc0[6] = d6
            out_acc0[7] = d7
            out_acc1[0] = e0
            out_acc1[1] = e1
            out_acc1[2] = e2
            out_acc1[3] = e3
            out_acc1[4] = e4
            out_acc1[5] = e5
            out_acc1[6] = e6
            out_acc1[7] = e7
            cute.arch.sync_threads()

        _store_mma_16x16_tile(out_acc0, mC, mScaleA, mScaleB, row_base, col_base, lane)
        _store_mma_16x16_tile(out_acc1, mC, mScaleA, mScaleB, row_base, col_base + Int32(16), lane)

class _Fp8DenseMmaStagedKernel:
    def __init__(self, m: int, n: int, k: int) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.num_threads = 128
        self.stage_k = 256

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(mA.element_type != cutlass.Float8E4M3FN):
            raise TypeError("A must be Float8E4M3FN")
        if cutlass.const_expr(mB.element_type != cutlass.Float8E4M3FN):
            raise TypeError("B must be Float8E4M3FN")
        if cutlass.const_expr(mC.element_type != cutlass.BFloat16):
            raise TypeError("C must be BFloat16")
        if cutlass.const_expr(mA.shape != (self.m, self.k)):
            raise ValueError("A shape mismatch")
        if cutlass.const_expr(mB.shape != (self.n, self.k)):
            raise ValueError("B shape mismatch")
        if cutlass.const_expr(mC.shape != (self.m, self.n)):
            raise ValueError("C shape mismatch")
        if cutlass.const_expr(self.m % 32 != 0 or self.n % 32 != 0 or self.k % self.stage_k != 0):
            raise ValueError("staged mma prototype requires M divisible by 32, N by 32, and K divisible by stage_k")

        grid = (self.m // 32, self.n // 32, 1)
        self.kernel(mA, mB, mScaleA, mScaleB, mC).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mC: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_m = cute.arch.block_idx()[0]
        block_n = cute.arch.block_idx()[1]
        cta_row_base = block_m * Int32(32)
        cta_col_base = block_n * Int32(32)
        row_base = cta_row_base + warp_idx * Int32(16)
        unit_scale = Uint32(0x7F7F7F7F)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(element_type=cutlass.Uint8, layout=cute.make_layout((32, 256), stride=(256, 1)), byte_alignment=128)
        sB = smem.allocate_tensor(element_type=cutlass.Uint8, layout=cute.make_layout((32, 256), stride=(256, 1)), byte_alignment=128)
        a_base_addr = shared_ptr_to_u32(sA.iterator)
        b_base_addr = shared_ptr_to_u32(sB.iterator)
        a_u8 = cute.recast_tensor(mA, cutlass.Uint8)
        b_u8 = cute.recast_tensor(mB, cutlass.Uint8)

        out_acc0 = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        out_acc1 = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        for reg_id in cutlass.range_constexpr(8):
            out_acc0[reg_id] = Float32(0.0)
            out_acc1[reg_id] = Float32(0.0)

        for k_stage_idx in cutlass.range(self.k // self.stage_k, unroll=1):
            k_stage = k_stage_idx * Int32(self.stage_k)
            linear = tidx
            while linear < Int32(32 * 256 // 16):
                row = linear // Int32(16)
                vec_idx = linear - row * Int32(16)
                src_col = k_stage + vec_idx * Int32(16)
                dst_addr = a_base_addr + row * Int32(256) + vec_idx * Int32(16)
                st_shared_v4_u32(
                    dst_addr,
                    Uint32(a_u8[cta_row_base + row, src_col + Int32(0)])
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(1)]) << Uint32(8))
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(2)]) << Uint32(16))
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(3)]) << Uint32(24)),
                    Uint32(a_u8[cta_row_base + row, src_col + Int32(4)])
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(5)]) << Uint32(8))
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(6)]) << Uint32(16))
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(7)]) << Uint32(24)),
                    Uint32(a_u8[cta_row_base + row, src_col + Int32(8)])
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(9)]) << Uint32(8))
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(10)]) << Uint32(16))
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(11)]) << Uint32(24)),
                    Uint32(a_u8[cta_row_base + row, src_col + Int32(12)])
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(13)]) << Uint32(8))
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(14)]) << Uint32(16))
                    | (Uint32(a_u8[cta_row_base + row, src_col + Int32(15)]) << Uint32(24)),
                )
                linear += Int32(self.num_threads)

            linear = tidx
            while linear < Int32(32 * 256 // 16):
                row = linear // Int32(16)
                vec_idx = linear - row * Int32(16)
                src_col = k_stage + vec_idx * Int32(16)
                dst_addr = _smem_addr_from_b128_offset(b_base_addr, _permuted_offset_128b(row, vec_idx, Int32(16)))
                st_shared_v4_u32(
                    dst_addr,
                    Uint32(b_u8[cta_col_base + row, src_col + Int32(0)])
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(1)]) << Uint32(8))
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(2)]) << Uint32(16))
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(3)]) << Uint32(24)),
                    Uint32(b_u8[cta_col_base + row, src_col + Int32(4)])
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(5)]) << Uint32(8))
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(6)]) << Uint32(16))
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(7)]) << Uint32(24)),
                    Uint32(b_u8[cta_col_base + row, src_col + Int32(8)])
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(9)]) << Uint32(8))
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(10)]) << Uint32(16))
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(11)]) << Uint32(24)),
                    Uint32(b_u8[cta_col_base + row, src_col + Int32(12)])
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(13)]) << Uint32(8))
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(14)]) << Uint32(16))
                    | (Uint32(b_u8[cta_col_base + row, src_col + Int32(15)]) << Uint32(24)),
                )
                linear += Int32(self.num_threads)
            cute.arch.sync_threads()

            if warp_idx < Int32(2):
                for k_inner in cutlass.range_constexpr(0, 256, 32):
                    a_row = warp_idx * Int32(16) + lane // Int32(4)
                    a_base_col = (lane % Int32(4)) * Int32(2)
                    a_regs = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                    a_regs[0] = (
                        Uint32(sA[a_row, Int32(k_inner) + a_base_col + Int32(0)])
                        | (Uint32(sA[a_row, Int32(k_inner) + a_base_col + Int32(1)]) << Int32(8))
                        | (Uint32(sA[a_row, Int32(k_inner) + a_base_col + Int32(8)]) << Int32(16))
                        | (Uint32(sA[a_row, Int32(k_inner) + a_base_col + Int32(9)]) << Int32(24))
                    )
                    a_regs[1] = (
                        Uint32(sA[a_row + Int32(8), Int32(k_inner) + a_base_col + Int32(0)])
                        | (Uint32(sA[a_row + Int32(8), Int32(k_inner) + a_base_col + Int32(1)]) << Int32(8))
                        | (Uint32(sA[a_row + Int32(8), Int32(k_inner) + a_base_col + Int32(8)]) << Int32(16))
                        | (Uint32(sA[a_row + Int32(8), Int32(k_inner) + a_base_col + Int32(9)]) << Int32(24))
                    )
                    a_regs[2] = (
                        Uint32(sA[a_row, Int32(k_inner) + a_base_col + Int32(16)])
                        | (Uint32(sA[a_row, Int32(k_inner) + a_base_col + Int32(17)]) << Int32(8))
                        | (Uint32(sA[a_row, Int32(k_inner) + a_base_col + Int32(24)]) << Int32(16))
                        | (Uint32(sA[a_row, Int32(k_inner) + a_base_col + Int32(25)]) << Int32(24))
                    )
                    a_regs[3] = (
                        Uint32(sA[a_row + Int32(8), Int32(k_inner) + a_base_col + Int32(16)])
                        | (Uint32(sA[a_row + Int32(8), Int32(k_inner) + a_base_col + Int32(17)]) << Int32(8))
                        | (Uint32(sA[a_row + Int32(8), Int32(k_inner) + a_base_col + Int32(24)]) << Int32(16))
                        | (Uint32(sA[a_row + Int32(8), Int32(k_inner) + a_base_col + Int32(25)]) << Int32(24))
                    )

                    b_offset = _permuted_offset_128b(
                        lane % Int32(8),
                        Int32(k_inner // 16) + (lane % Int32(16)) // Int32(8),
                        Int32(16),
                    ) + Int32(16) * (lane // Int32(16)) * Int32(8)
                    b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(_smem_addr_from_b128_offset(b_base_addr, b_offset))
                    b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(_smem_addr_from_b128_offset(b_base_addr, b_offset))
                    b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
                    b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
                    b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
                    b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)
                    b_offset_1 = b_offset + Int32(16 * 16)
                    b2_k0, b3_k0 = ldmatrix_m8n8x4_left_half_b16(_smem_addr_from_b128_offset(b_base_addr, b_offset_1))
                    b2_k1, b3_k1 = ldmatrix_m8n8x4_right_half_b16(_smem_addr_from_b128_offset(b_base_addr, b_offset_1))
                    b2_k0 = frag_layout_swizzle_16b_to_8b(b2_k0)
                    b3_k0 = frag_layout_swizzle_16b_to_8b(b3_k0)
                    b2_k1 = frag_layout_swizzle_16b_to_8b(b2_k1)
                    b3_k1 = frag_layout_swizzle_16b_to_8b(b3_k1)

                    d0, d1, d2, d3 = f8f6f4_mma_m16n8k32_f32_e4m3(
                        out_acc0[0], out_acc0[1], out_acc0[2], out_acc0[3],
                        a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                        b0_k0, b0_k1,
                    )
                    d4, d5, d6, d7 = f8f6f4_mma_m16n8k32_f32_e4m3(
                        out_acc0[4], out_acc0[5], out_acc0[6], out_acc0[7],
                        a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                        b1_k0, b1_k1,
                    )
                    e0, e1, e2, e3 = f8f6f4_mma_m16n8k32_f32_e4m3(
                        out_acc1[0], out_acc1[1], out_acc1[2], out_acc1[3],
                        a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                        b2_k0, b2_k1,
                    )
                    e4, e5, e6, e7 = f8f6f4_mma_m16n8k32_f32_e4m3(
                        out_acc1[4], out_acc1[5], out_acc1[6], out_acc1[7],
                        a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                        b3_k0, b3_k1,
                    )
                    out_acc0[0] = d0
                    out_acc0[1] = d1
                    out_acc0[2] = d2
                    out_acc0[3] = d3
                    out_acc0[4] = d4
                    out_acc0[5] = d5
                    out_acc0[6] = d6
                    out_acc0[7] = d7
                    out_acc1[0] = e0
                    out_acc1[1] = e1
                    out_acc1[2] = e2
                    out_acc1[3] = e3
                    out_acc1[4] = e4
                    out_acc1[5] = e5
                    out_acc1[6] = e6
                    out_acc1[7] = e7
            cute.arch.sync_threads()

        if warp_idx < Int32(2):
            _store_mma_16x16_tile(out_acc0, mC, mScaleA, mScaleB, row_base, cta_col_base, lane)
            _store_mma_16x16_tile(out_acc1, mC, mScaleA, mScaleB, row_base, cta_col_base + Int32(16), lane)

def fp8_dense_gemm_mma(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_fp8_dense_inputs(a, b, scale_a, scale_b)
    m = int(a.shape[0])
    k = int(a.shape[1])
    n = int(b.shape[0])
    if m % 16 != 0 or n % 32 != 0 or k % 32 != 0:
        raise ValueError("mma prototype requires M divisible by 16, N by 32, and K by 32")
    out = _prepare_output(a, m, n, out)
    scale_a = scale_a.reshape(1).contiguous()
    scale_b = scale_b.reshape(1).contiguous()
    stream = current_cuda_stream()

    a_cute = _to_cute_tensor(a, cutlass.Float8E4M3FN)
    b_cute = _to_cute_tensor(b, cutlass.Float8E4M3FN)
    scale_a_cute = _to_cute_tensor(scale_a, cutlass.Float32)
    scale_b_cute = _to_cute_tensor(scale_b, cutlass.Float32)
    out_cute = _to_cute_tensor(out, cutlass.BFloat16)

    use_staged = m % 32 == 0 and n % 32 == 0 and k % 256 == 0
    key = _CompileKey(m, n, k, tuple(a.stride()), tuple(b.stride()), tuple(out.stride()), 128 if use_staged else 32)
    compiled = _COMPILED_MMA.get(key)
    if compiled is None:
        kernel = _Fp8DenseMmaStagedKernel(m, n, k) if use_staged else _Fp8DenseMmaKernel(m, n, k)
        compiled = cute.compile(kernel, a_cute, b_cute, scale_a_cute, scale_b_cute, out_cute, stream)
        _COMPILED_MMA[key] = compiled
    compiled(a_cute, b_cute, scale_a_cute, scale_b_cute, out_cute, stream)
    return out

def _validate_fp8_dense_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> None:
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
    if a.shape[1] % 4 != 0:
        raise ValueError("K must be divisible by 4")
    if not a.is_cuda or not b.is_cuda or not scale_a.is_cuda or not scale_b.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous for this prototype")

def _prepare_output(a: torch.Tensor, m: int, n: int, out: Optional[torch.Tensor]) -> torch.Tensor:
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if out.shape != (m, n) or out.dtype is not torch.bfloat16 or not out.is_cuda:
        raise ValueError("out must be a CUDA bfloat16 tensor with shape (M, N)")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous for this prototype")
    return out

def fp8_dense_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    threads_per_cta: int = 256,
) -> torch.Tensor:
    _validate_fp8_dense_inputs(a, b, scale_a, scale_b)

    m = int(a.shape[0])
    k = int(a.shape[1])
    n = int(b.shape[0])
    out = _prepare_output(a, m, n, out)

    scale_a = scale_a.reshape(1).contiguous()
    scale_b = scale_b.reshape(1).contiguous()
    stream = current_cuda_stream()

    a_cute = _to_cute_tensor(a, cutlass.Float8E4M3FN)
    b_cute = _to_cute_tensor(b, cutlass.Float8E4M3FN)
    scale_a_cute = _to_cute_tensor(scale_a, cutlass.Float32)
    scale_b_cute = _to_cute_tensor(scale_b, cutlass.Float32)
    out_cute = _to_cute_tensor(out, cutlass.BFloat16)

    key = _CompileKey(m, n, k, tuple(a.stride()), tuple(b.stride()), tuple(out.stride()), threads_per_cta)
    compiled = _COMPILED.get(key)
    if compiled is None:
        kernel = _Fp8DenseDotKernel(m, n, k, threads_per_cta)
        compiled = cute.compile(kernel, a_cute, b_cute, scale_a_cute, scale_b_cute, out_cute, stream)
        _COMPILED[key] = compiled

    compiled(a_cute, b_cute, scale_a_cute, scale_b_cute, out_cute, stream)
    return out

__all__ = ["fp8_dense_gemm", "fp8_dense_gemm_mma"]
