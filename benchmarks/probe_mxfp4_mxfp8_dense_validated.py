#!/usr/bin/env python3
"""Validated single-tile MXFP4×MXFP8 GEMM (M=16, N=8, K=multiple of 32).

Single CTA, single warp. Each thread holds the m16n8k32 standard fragment.
Inputs are random uniform in the FP4 representable set, scales fixed to 1.0
so the kernel result must match a plain `A @ B.T` in fp32 exactly.

Per PTX m16n8k32 (row.col, byte-stored elements):

  A (16×K, row-major), per-thread 4 b32:
    a0: bytes (row=t/4,     col=(t%4)*8 + 0..3) for K-block 0
    a1: bytes (row=t/4 + 8, col=(t%4)*8 + 0..3) for K-block 0
    a2: bytes (row=t/4,     col=(t%4)*8 + 4..7) for K-block 0
    a3: bytes (row=t/4 + 8, col=(t%4)*8 + 4..7) for K-block 0
  B (K×8, col-major), per-thread 2 b32:
    b0: bytes (col=t/4, k=(t%4)*8 + 0..3) — bytes are e4m3
    b1: bytes (col=t/4, k=(t%4)*8 + 4..7)
  But B is packed MXFP4 (2 nibbles/byte), so on disk:
    bp[byte i]: B_packed[col=t/4, k_byte=(t%4)*4 + i] for i in 0..3
    where each byte holds (low nibble = k=2*i, high nibble = k=2*i+1) within the lane's 8-K window.
  D (16×8) per-thread 4 f32:
    d0: (row=t/4,     col=(t%4)*2 + 0)
    d1: (row=t/4,     col=(t%4)*2 + 1)
    d2: (row=t/4 + 8, col=(t%4)*2 + 0)
    d3: (row=t/4 + 8, col=(t%4)*2 + 1)
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


_FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


@cute.jit
def gemm_kernel(
    mA: cute.Tensor,    # (32, K_blocks, 4) u32 — per-thread A fragments
    mBp: cute.Tensor,   # (32, K_blocks, 1) u32 — per-thread packed FP4 B
    mSFa: cute.Tensor,  # (1,) u32  scalar
    mSFb: cute.Tensor,  # (1,) u32  scalar
    mD: cute.Tensor,    # (32, 4) f32 — per-thread D
    stream: cuda.CUstream,
):
    gemm_warp(mA, mBp, mSFa, mSFb, mD).launch(
        grid=(1, 1, 1), block=[32, 1, 1], stream=stream,
    )


@cute.kernel
def gemm_warp(mA, mBp, mSFa, mSFb, mD):
    tidx = cute.arch.thread_idx()[0]
    sfa = mSFa[0]
    sfb = mSFb[0]
    K_blocks = mA.shape[1]

    d0 = Float32(0.0); d1 = Float32(0.0); d2 = Float32(0.0); d3 = Float32(0.0)

    for k in cutlass.range_constexpr(K_blocks):
        a0 = mA[tidx, k, 0]
        a1 = mA[tidx, k, 1]
        a2 = mA[tidx, k, 2]
        a3 = mA[tidx, k, 3]
        bp = mBp[tidx, k, 0]
        b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        )

    mD[tidx, 0] = d0
    mD[tidx, 1] = d1
    mD[tidx, 2] = d2
    mD[tidx, 3] = d3


def _e4m3_byte_lut(device):
    vals = torch.tensor(_FP4_LUT, dtype=torch.float32, device=device)
    bytes_ = vals.to(torch.float8_e4m3fn).view(torch.uint8)
    return bytes_


def _pack_a_fragments(A_e4m3: torch.Tensor) -> torch.Tensor:
    """Pack (16, K) E4M3 byte tensor into (32, K_blocks, 4) per-thread u32 frags."""
    M, K = A_e4m3.shape
    assert M == 16 and K % 32 == 0
    K_blocks = K // 32
    out = torch.zeros(32, K_blocks, 4, dtype=torch.int32, device=A_e4m3.device)
    A_int = A_e4m3.to(torch.int32)
    for t in range(32):
        group = t // 4
        lane = t % 4
        for kb in range(K_blocks):
            kb_off = kb * 32
            # a0: row=group, cols=lane*8 + 0..3
            for i in range(4):
                out[t, kb, 0] |= (A_int[group, kb_off + lane*8 + i] & 0xFF) << (i*8)
            # a1: row=group+8, cols=lane*8 + 0..3
            for i in range(4):
                out[t, kb, 1] |= (A_int[group + 8, kb_off + lane*8 + i] & 0xFF) << (i*8)
            # a2: row=group, cols=lane*8 + 4..7
            for i in range(4):
                out[t, kb, 2] |= (A_int[group, kb_off + lane*8 + 4 + i] & 0xFF) << (i*8)
            # a3: row=group+8, cols=lane*8 + 4..7
            for i in range(4):
                out[t, kb, 3] |= (A_int[group + 8, kb_off + lane*8 + 4 + i] & 0xFF) << (i*8)
    return out


def _pack_b_fragments(B_packed_fp4: torch.Tensor) -> torch.Tensor:
    """Pack (8, K/2) col-major-on-disk packed FP4 → (32, K_blocks, 1) per-thread u32."""
    N, K_half = B_packed_fp4.shape
    assert N == 8
    K = K_half * 2
    assert K % 32 == 0
    K_blocks = K // 32
    out = torch.zeros(32, K_blocks, 1, dtype=torch.int32, device=B_packed_fp4.device)
    B_int = B_packed_fp4.to(torch.int32)
    for t in range(32):
        col = t // 4
        lane = t % 4
        for kb in range(K_blocks):
            # Per K-block (32 K positions), this thread holds 8 K positions:
            # k = lane*8 + 0..7. As packed FP4 (2 nibbles per byte), that's 4 bytes.
            kb_byte_off = (kb * 32) // 2
            for i in range(4):
                byte_idx = kb_byte_off + lane * 4 + i
                out[t, kb, 0] |= (B_int[col, byte_idx] & 0xFF) << (i*8)
    return out


def _unpack_d(D_per_thread: torch.Tensor) -> torch.Tensor:
    """(32, 4) per-thread fp32 → (16, 8) D."""
    D = torch.zeros(16, 8, dtype=torch.float32, device=D_per_thread.device)
    for t in range(32):
        group = t // 4
        lane = t % 4
        D[group,     lane*2 + 0] = D_per_thread[t, 0]
        D[group,     lane*2 + 1] = D_per_thread[t, 1]
        D[group + 8, lane*2 + 0] = D_per_thread[t, 2]
        D[group + 8, lane*2 + 1] = D_per_thread[t, 3]
    return D


def _quantize_to_fp4_packed_lut_index(idx: torch.Tensor) -> torch.Tensor:
    """idx: (N, K) int in [0..15] FP4 LUT index → (N, K/2) packed uint8 (low|high)."""
    N, K = idx.shape
    assert K % 2 == 0
    pairs = idx.reshape(N, K // 2, 2).to(torch.int32)
    packed = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)) & 0xFF
    return packed.to(torch.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=32, help="K dim, multiple of 32")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    torch.manual_seed(args.seed)

    K = args.K
    M, N = 16, 8
    assert K % 32 == 0

    # Generate random A,B with values from the FP4 LUT (so quantize-dequant is identity).
    a_idx = torch.randint(0, 16, (M, K), device=device)  # 0..15 → LUT
    b_idx = torch.randint(0, 16, (N, K), device=device)
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=device)
    A_f32 = lut[a_idx]
    B_f32 = lut[b_idx]

    # Reference D = A @ B.T (in fp32; since values are exact, this is the gold).
    D_ref = (A_f32 @ B_f32.T)  # (16, 8)

    # Quantize A → E4M3 byte (since LUT vals are exact in E4M3 too).
    A_e4m3 = A_f32.to(torch.float8_e4m3fn).view(torch.uint8)
    # Quantize B (packed FP4) using the LUT-index path (exact).
    B_packed = _quantize_to_fp4_packed_lut_index(b_idx)  # (8, K/2)

    A_frag = _pack_a_fragments(A_e4m3)
    B_frag = _pack_b_fragments(B_packed)
    SFa = torch.tensor([0x7F], device=device, dtype=torch.int32)  # scale = 1.0
    SFb = torch.tensor([0x7F], device=device, dtype=torch.int32)
    D_per_thread = torch.zeros(32, 4, device=device, dtype=torch.float32)

    cuteA = _to_cute(A_frag, cutlass.Uint32)
    cuteBp = _to_cute(B_frag, cutlass.Uint32)
    cuteSFa = _to_cute(SFa, cutlass.Uint32)
    cuteSFb = _to_cute(SFb, cutlass.Uint32)
    cuteD = _to_cute(D_per_thread, cutlass.Float32)

    compiled = cute.compile(gemm_kernel, cuteA, cuteBp, cuteSFa, cuteSFb, cuteD, stream)
    compiled(cuteA, cuteBp, cuteSFa, cuteSFb, cuteD, stream)
    torch.cuda.synchronize()

    D_kernel = _unpack_d(D_per_thread)

    err = (D_kernel - D_ref).abs()
    max_abs = err.max().item()
    print(f"K={K}  max abs err = {max_abs}")
    if max_abs < 1e-3:
        print("  PASS")
        return 0
    else:
        print("  FAIL")
        print(f"  D_kernel[:4, :4] =\n{D_kernel[:4, :4]}")
        print(f"  D_ref[:4, :4] =\n{D_ref[:4, :4]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
