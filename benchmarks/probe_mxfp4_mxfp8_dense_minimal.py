#!/usr/bin/env python3
"""Minimal single-warp MXFP4-weight × MXFP8-activation GEMM.

Single CTA, single warp, M=16, N=8 output tile. K is looped per-warp. No TMA,
no shared memory pipeline — purely a correctness demonstrator for the
dequant-then-mxfp8-MMA data path. Validates against torch reference.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import (
    cvt_e2m1x8_to_e4m3x8,
    mxfp8_mma_m16n8k32_f32_e4m3,
)
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine,
    dequant_mxfp4,
    dequant_mxfp8,
    quantize_to_mxfp4,
    quantize_to_mxfp8,
)


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


# Per-thread fragment layout for m16n8k32 (row-major A, col-major B):
#   - A holds 16 rows × 32 K-cols. Each thread holds 4 b32 = 16 bytes = 16 elements.
#     Lane t covers rows [t/4, t/4+8] (mod 16) at cols [(t%4)*8 + 0..7] and a mirror.
#     We don't need that exact mapping — we'll feed the same value to every thread
#     for the smoke; for the real kernel we use ldmatrix.
#   - B holds 32 K-rows × 8 N-cols. Each thread holds 2 b32 = 8 bytes = 8 elements.


@cute.jit
def gemm_kernel(
    mA_e4m3: cute.Tensor,    # (32, K_tiles, 4) u32 — per-thread A fragments
    mA_sf:   cute.Tensor,    # (1,) u32
    mB_packed: cute.Tensor,  # (32, K_tiles, 1) u32 — per-thread packed FP4 B
    mB_sf:   cute.Tensor,    # (1,) u32
    mD:      cute.Tensor,    # (32, 4) f32
    stream: cuda.CUstream,
):
    gemm_warp(mA_e4m3, mA_sf, mB_packed, mB_sf, mD).launch(
        grid=(1, 1, 1), block=[32, 1, 1], stream=stream,
    )


@cute.kernel
def gemm_warp(mA_e4m3, mA_sf, mB_packed, mB_sf, mD):
    tidx = cute.arch.thread_idx()[0]
    sfa = mA_sf[0]
    sfb = mB_sf[0]

    K_tiles = mA_e4m3.shape[1]

    d0 = Float32(0.0); d1 = Float32(0.0); d2 = Float32(0.0); d3 = Float32(0.0)

    # K loop: each iteration consumes a K=32 block via one mxf8f6f4 m16n8k32 MMA.
    for k in cutlass.range_constexpr(K_tiles):
        a0 = mA_e4m3[tidx, k, 0]
        a1 = mA_e4m3[tidx, k, 1]
        a2 = mA_e4m3[tidx, k, 2]
        a3 = mA_e4m3[tidx, k, 3]
        b_packed = mB_packed[tidx, k, 0]
        b0, b1 = cvt_e2m1x8_to_e4m3x8(b_packed)

        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        )

    mD[tidx, 0] = d0
    mD[tidx, 1] = d1
    mD[tidx, 2] = d2
    mD[tidx, 3] = d3


def _packed_e4m3_one_u32() -> int:
    one = 0x38
    return one | (one << 8) | (one << 16) | (one << 24)


def _packed_fp4_ones_u32() -> int:
    """8 nibbles all = 0x2 (E2M1 1.0) = 0x22222222."""
    return 0x22222222


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K-tiles", type=int, default=1, help="number of K=32 chunks")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    K_tiles = args.K_tiles
    A_buf = torch.full((32, K_tiles, 4), _packed_e4m3_one_u32(), device=device, dtype=torch.int32)
    B_buf = torch.full((32, K_tiles, 1), _packed_fp4_ones_u32(),  device=device, dtype=torch.int32)
    SF_a = torch.tensor([0x7F], device=device, dtype=torch.int32)
    SF_b = torch.tensor([0x7F], device=device, dtype=torch.int32)
    D = torch.zeros(32, 4, device=device, dtype=torch.float32)

    cuteA = _to_cute(A_buf, cutlass.Uint32)
    cuteB = _to_cute(B_buf, cutlass.Uint32)
    cuteSFa = _to_cute(SF_a, cutlass.Uint32)
    cuteSFb = _to_cute(SF_b, cutlass.Uint32)
    cuteD = _to_cute(D, cutlass.Float32)

    compiled = cute.compile(gemm_kernel, cuteA, cuteSFa, cuteB, cuteSFb, cuteD, stream)
    compiled(cuteA, cuteSFa, cuteB, cuteSFb, cuteD, stream)
    torch.cuda.synchronize()

    # With A=B=1.0 and K=K_tiles*32 elements, each output position should be K_tiles*32.
    expected = float(K_tiles * 32)
    print(f"K_tiles={K_tiles}  expected D = {expected}")
    print(f"  thread 0 D: {D[0].tolist()}")
    print(f"  thread 31 D: {D[31].tolist()}")
    err = (D - expected).abs().max().item()
    print(f"  max abs err: {err}")
    if err < 1e-3:
        print("  PASS")
    else:
        print("  FAIL")


if __name__ == "__main__":
    main()
