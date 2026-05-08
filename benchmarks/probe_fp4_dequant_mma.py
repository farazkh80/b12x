#!/usr/bin/env python3
"""Probe MXFP4-weight × MXFP8-activation MMA via in-register FP4→E4M3 dequant.

We don't have a confirmed mxf4×mxf8 native path (mxf8f6f4 .e4m3.e2m1 shows a
factor-4 layout mismatch we haven't pinned down). Instead, we exploit the fact
that every E2M1 representable value is exactly representable in E4M3, so we
widen FP4→E4M3 in registers and use the proven `mxfp8_mma_m16n8k32_f32_e4m3`
warp-MMA. This wastes some register space (4-bit weights become 8-bit) but is
provably correct and uses an instruction we trust.

Smoke:
  - Take a packed FP4 register containing 8 nibbles, all encoding 1.0 (= 0x22 22 22 22).
  - Convert to two packed E4M3 registers via `cvt_e2m1x8_to_e4m3x8`.
  - Feed (a*4, b0, b1) where A is E4M3 1.0 and B is the converted FP4-as-E4M3.
  - Sweep ue8m0 scales; expect D = 32 * 2^(sf-127) * 2^(sf-127) (same as raw mxfp8 case).
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


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


@cute.jit
def probe(
    mA: cute.Tensor,         # (32, 4) u32 — E4M3 A
    mB_packed: cute.Tensor,  # (32, 1) u32 — packed FP4 B (8 nibbles per b32)
    mSF: cute.Tensor,        # (1,)
    mD: cute.Tensor,         # (32, 4) f32
    stream: cuda.CUstream,
):
    probe_kernel(mA, mB_packed, mSF, mD).launch(grid=(1, 1, 1), block=[32, 1, 1], stream=stream)


@cute.kernel
def probe_kernel(mA, mB_packed, mSF, mD):
    tidx = cute.arch.thread_idx()[0]
    a0 = mA[tidx, 0]; a1 = mA[tidx, 1]; a2 = mA[tidx, 2]; a3 = mA[tidx, 3]
    packed = mB_packed[tidx, 0]
    b0, b1 = cvt_e2m1x8_to_e4m3x8(packed)
    sfa = mSF[0]; sfb = mSF[0]
    d0 = Float32(0.0); d1 = Float32(0.0); d2 = Float32(0.0); d3 = Float32(0.0)
    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
        d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
    )
    mD[tidx, 0] = d0; mD[tidx, 1] = d1; mD[tidx, 2] = d2; mD[tidx, 3] = d3


def _u32_packed(byte_val: int) -> int:
    return byte_val | (byte_val << 8) | (byte_val << 16) | (byte_val << 24)


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    e4m3_one = _u32_packed(0x38)              # 4 × E4M3 1.0
    fp4_packed_ones = _u32_packed(0x22)       # 4 bytes × 2 nibbles each = 8 packed FP4 1.0

    A = torch.full((32, 4), e4m3_one, device=device, dtype=torch.int32)
    Bp = torch.full((32, 1), fp4_packed_ones, device=device, dtype=torch.int32)
    D = torch.zeros(32, 4, device=device, dtype=torch.float32)

    cuteA = _to_cute(A, cutlass.Uint32)
    cuteBp = _to_cute(Bp, cutlass.Uint32)
    cuteD = _to_cute(D, cutlass.Float32)

    print("FP4-dequant + MXFP8 MMA path (B = packed FP4 ones, A = E4M3 ones, K=32):")
    for sf_byte in [0x7d, 0x7e, 0x7f, 0x80, 0x81]:
        SF = torch.tensor([sf_byte], device=device, dtype=torch.int32)
        cuteSF = _to_cute(SF, cutlass.Uint32)
        D.zero_()
        compiled = cute.compile(probe, cuteA, cuteBp, cuteSF, cuteD, stream)
        compiled(cuteA, cuteBp, cuteSF, cuteD, stream)
        torch.cuda.synchronize()
        exp = 2.0 ** (sf_byte - 127)
        expected = 32.0 * exp * exp
        print(f"  sf=0x{sf_byte:02x}  D[0,0]={D[0,0].item():>10.4f}  expected={expected:>10.4f}  ratio={D[0,0].item()/max(expected,1e-9):.4f}")


if __name__ == "__main__":
    main()
