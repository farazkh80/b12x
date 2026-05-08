#!/usr/bin/env python3
"""Single-tile MXFP4×MXFP8 GEMM with per-block UE8M0 scales.

Extends `probe_mxfp4_mxfp8_dense_validated.py` to use real per-row sfa and
per-col sfb scales (vs fixed 1.0), and random fp32 inputs that go through
proper MXFP4/MXFP8 quantization (vs LUT-exact). Reference is the torch
reference's dequant matmul.

Per-block scale layout for m16n8k32 with `scale_vec::1X`:
  SFA: 16 bytes per K-block (one per A row).
  SFB:  8 bytes per K-block (one per B col).
Per-thread scale registers (1 b32 each):
  sfa[t] holds the byte for row t/4 (the lower row group); the upper row
    group's byte for thread t is at row t/4 + 8 — but the same `1X` slot
    ALSO carries that byte. Per CUTLASS docs the convention is:
      sfa byte 0 = row=t/4
      sfa byte 1 = row=t/4 + 8
    (only low 2 bytes used; bytes 2..3 set to 0)
  Likewise sfb byte 0 = col=t/4, but since B has only 8 cols and thread t/4 < 8
    only one byte is meaningful per thread; we set byte 1..3 = byte 0 for safety
    (the existing nvfp4 path uses just low byte).
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


@cute.jit
def gemm_kernel(
    mA: cute.Tensor,    # (32, K_blocks, 4) u32
    mBp: cute.Tensor,   # (32, K_blocks, 1) u32
    mSFa: cute.Tensor,  # (32, K_blocks) u32  per-thread per-block scale register
    mSFb: cute.Tensor,  # (32, K_blocks) u32
    mD: cute.Tensor,    # (32, 4) f32
    stream: cuda.CUstream,
):
    gemm_warp(mA, mBp, mSFa, mSFb, mD).launch(
        grid=(1, 1, 1), block=[32, 1, 1], stream=stream,
    )


@cute.kernel
def gemm_warp(mA, mBp, mSFa, mSFb, mD):
    tidx = cute.arch.thread_idx()[0]
    K_blocks = mA.shape[1]

    d0 = Float32(0.0); d1 = Float32(0.0); d2 = Float32(0.0); d3 = Float32(0.0)

    for k in cutlass.range_constexpr(K_blocks):
        a0 = mA[tidx, k, 0]
        a1 = mA[tidx, k, 1]
        a2 = mA[tidx, k, 2]
        a3 = mA[tidx, k, 3]
        bp = mBp[tidx, k, 0]
        b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)
        sfa = mSFa[tidx, k]
        sfb = mSFb[tidx, k]
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        )

    mD[tidx, 0] = d0
    mD[tidx, 1] = d1
    mD[tidx, 2] = d2
    mD[tidx, 3] = d3


def _pack_a_fragments(A_e4m3: torch.Tensor) -> torch.Tensor:
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
            for i in range(4):
                out[t, kb, 0] |= (A_int[group, kb_off + lane*8 + i] & 0xFF) << (i*8)
            for i in range(4):
                out[t, kb, 1] |= (A_int[group + 8, kb_off + lane*8 + i] & 0xFF) << (i*8)
            for i in range(4):
                out[t, kb, 2] |= (A_int[group, kb_off + lane*8 + 4 + i] & 0xFF) << (i*8)
            for i in range(4):
                out[t, kb, 3] |= (A_int[group + 8, kb_off + lane*8 + 4 + i] & 0xFF) << (i*8)
    return out


def _pack_b_fragments(B_packed: torch.Tensor) -> torch.Tensor:
    N, K_half = B_packed.shape
    assert N == 8
    K_blocks = (K_half * 2) // 32
    out = torch.zeros(32, K_blocks, 1, dtype=torch.int32, device=B_packed.device)
    B_int = B_packed.to(torch.int32)
    for t in range(32):
        col = t // 4
        lane = t % 4
        for kb in range(K_blocks):
            kb_byte_off = (kb * 32) // 2
            for i in range(4):
                byte_idx = kb_byte_off + lane * 4 + i
                out[t, kb, 0] |= (B_int[col, byte_idx] & 0xFF) << (i*8)
    return out


def _pack_sfa(sfa_bytes: torch.Tensor) -> torch.Tensor:
    """sfa_bytes: (16, K_blocks) uint8 → (32, K_blocks) u32 per-thread scale register.

    Probed layout (m16n8k32, scale_vec::1X):
      thread t in group g=t/4, lane l=t%4
        l==0  →  byte0 = sfa[row = g     ]   (top row group)
        l==1  →  byte0 = sfa[row = g + 8 ]   (bottom row group)
        l>=2  →  byte0 ignored (set to 0x7F = 1.0 to keep it neutral if it leaks)
    """
    M, K_blocks = sfa_bytes.shape
    assert M == 16
    out = torch.full((32, K_blocks), 0x7F, dtype=torch.int32, device=sfa_bytes.device)
    for t in range(32):
        g = t // 4
        l = t % 4
        if l == 0:
            row = g
        elif l == 1:
            row = g + 8
        else:
            continue
        for kb in range(K_blocks):
            byte = int(sfa_bytes[row, kb].item()) & 0xFF
            out[t, kb] = byte
    return out


def _pack_sfb(sfb_bytes: torch.Tensor) -> torch.Tensor:
    """sfb_bytes: (8, K_blocks) uint8 → (32, K_blocks) u32 per-thread scale register.

    Mirroring SFA's pattern but for B's 8 columns. Probed: lane 0 of each group
    holds the scale for col=group; lanes 1..3 are ignored. Since B has only 8
    cols (one per group), we don't need the +8 trick.
    """
    N, K_blocks = sfb_bytes.shape
    assert N == 8
    out = torch.full((32, K_blocks), 0x7F, dtype=torch.int32, device=sfb_bytes.device)
    for t in range(32):
        g = t // 4
        l = t % 4
        if l != 0:
            continue
        col = g
        for kb in range(K_blocks):
            byte = int(sfb_bytes[col, kb].item()) & 0xFF
            out[t, kb] = byte
    return out


def _unpack_d(D_per_thread: torch.Tensor) -> torch.Tensor:
    D = torch.zeros(16, 8, dtype=torch.float32, device=D_per_thread.device)
    for t in range(32):
        group = t // 4
        lane = t % 4
        D[group,     lane*2 + 0] = D_per_thread[t, 0]
        D[group,     lane*2 + 1] = D_per_thread[t, 1]
        D[group + 8, lane*2 + 0] = D_per_thread[t, 2]
        D[group + 8, lane*2 + 1] = D_per_thread[t, 3]
    return D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-uniform", action="store_true",
                        help="set all scales to 1.0 (sanity)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    torch.manual_seed(args.seed)

    K = args.K
    M, N = 16, 8
    assert K % 32 == 0

    A_f32 = torch.randn(M, K, dtype=torch.float32, device=device) * 0.5
    B_f32 = torch.randn(N, K, dtype=torch.float32, device=device) * 0.5

    # MXFP8 quant for A, MXFP4 quant for B
    A_e4m3, A_sf = quantize_to_mxfp8(A_f32)            # (16, K), (16, K/32)
    B_packed, B_sf = quantize_to_mxfp4(B_f32)          # (8, K/2), (8, K/32)

    # Reference: dequant + matmul. Mirrors what the kernel computes.
    A_dq = dequant_mxfp8(A_e4m3, A_sf, rows=M, cols=K)
    B_dq = dequant_mxfp4(B_packed, B_sf, rows=N, cols=K)
    D_ref = (A_dq @ B_dq.T)

    if args.scale_uniform:
        A_sf = torch.full_like(A_sf, 0x7F)
        B_sf = torch.full_like(B_sf, 0x7F)
        # Recompute reference accordingly
        A_dq = dequant_mxfp8(A_e4m3, A_sf, rows=M, cols=K)
        B_dq = dequant_mxfp4(B_packed, B_sf, rows=N, cols=K)
        D_ref = (A_dq @ B_dq.T)

    A_frag = _pack_a_fragments(A_e4m3)
    B_frag = _pack_b_fragments(B_packed)
    SFa_frag = _pack_sfa(A_sf)
    SFb_frag = _pack_sfb(B_sf)
    D_per_thread = torch.zeros(32, 4, device=device, dtype=torch.float32)

    cuteA = _to_cute(A_frag, cutlass.Uint32)
    cuteBp = _to_cute(B_frag, cutlass.Uint32)
    cuteSFa = _to_cute(SFa_frag, cutlass.Uint32)
    cuteSFb = _to_cute(SFb_frag, cutlass.Uint32)
    cuteD = _to_cute(D_per_thread, cutlass.Float32)

    compiled = cute.compile(gemm_kernel, cuteA, cuteBp, cuteSFa, cuteSFb, cuteD, stream)
    compiled(cuteA, cuteBp, cuteSFa, cuteSFb, cuteD, stream)
    torch.cuda.synchronize()

    D_kernel = _unpack_d(D_per_thread)

    err = (D_kernel - D_ref).abs()
    cos = cosine(D_kernel, D_ref)
    print(f"K={K} scale_uniform={args.scale_uniform}")
    print(f"  max abs err: {err.max().item():.6f}")
    print(f"  rmse:        {err.square().mean().sqrt().item():.6f}")
    print(f"  cosine:      {cos:.6f}")
    print(f"  D_ref      [0, :8]: {D_ref[0].tolist()}")
    print(f"  D_kernel   [0, :8]: {D_kernel[0].tolist()}")
    return 0 if cos > 0.999 else 1


if __name__ == "__main__":
    sys.exit(main())
