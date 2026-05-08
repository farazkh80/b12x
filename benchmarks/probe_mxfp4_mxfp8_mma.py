#!/usr/bin/env python3
"""Probe SM120 MXFP4×MXFP8 m16n8k32 MMA `kind::mxf8f6f4` E4M3/E2M1.

Mirrors `probe_mxfp8_scale.py` but flips B operand from E4M3 to E2M1.
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
    mxfp4_mxfp8_mma_m16n8k32_f32_e4m3_e2m1,
    mxfp8_mma_m16n8k32_f32_e4m3,
)


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


@cute.jit
def probe_mxfp4_mxfp8(
    mA: cute.Tensor, mB: cute.Tensor, mSF: cute.Tensor, mD: cute.Tensor,
    stream: cuda.CUstream,
):
    probe_kernel(mA, mB, mSF, mD).launch(grid=(1, 1, 1), block=[32, 1, 1], stream=stream)


@cute.kernel
def probe_kernel(mA, mB, mSF, mD):
    tidx = cute.arch.thread_idx()[0]
    a0 = mA[tidx, 0]; a1 = mA[tidx, 1]; a2 = mA[tidx, 2]; a3 = mA[tidx, 3]
    b0 = mB[tidx, 0]; b1 = mB[tidx, 1]
    sfa = mSF[0]; sfb = mSF[0]
    d0 = Float32(0.0); d1 = Float32(0.0); d2 = Float32(0.0); d3 = Float32(0.0)
    d0, d1, d2, d3 = mxfp4_mxfp8_mma_m16n8k32_f32_e4m3_e2m1(
        d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
    )
    mD[tidx, 0] = d0; mD[tidx, 1] = d1; mD[tidx, 2] = d2; mD[tidx, 3] = d3


@cute.jit
def probe_mxfp8_only(
    mA: cute.Tensor, mB: cute.Tensor, mSF: cute.Tensor, mD: cute.Tensor,
    stream: cuda.CUstream,
):
    probe_kernel_mxfp8(mA, mB, mSF, mD).launch(grid=(1, 1, 1), block=[32, 1, 1], stream=stream)


@cute.kernel
def probe_kernel_mxfp8(mA, mB, mSF, mD):
    tidx = cute.arch.thread_idx()[0]
    a0 = mA[tidx, 0]; a1 = mA[tidx, 1]; a2 = mA[tidx, 2]; a3 = mA[tidx, 3]
    b0 = mB[tidx, 0]; b1 = mB[tidx, 1]
    sfa = mSF[0]; sfb = mSF[0]
    d0 = Float32(0.0); d1 = Float32(0.0); d2 = Float32(0.0); d3 = Float32(0.0)
    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
        d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
    )
    mD[tidx, 0] = d0; mD[tidx, 1] = d1; mD[tidx, 2] = d2; mD[tidx, 3] = d3


def _u32_packed(byte_val: int) -> int:
    return byte_val | (byte_val << 8) | (byte_val << 16) | (byte_val << 24)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mxfp4", "mxfp8", "compare"], default="compare")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Reference: existing mxfp8 path with B=E4M3 1.0 (byte 0x38) → expect D=32 at sfa=sfb=0x7f.
    a_e4m3 = _u32_packed(0x38)
    b_e4m3 = _u32_packed(0x38)
    # Test FP4 byte-padded conventions: low-nibble vs high-nibble (1.0 = LUT[2] = 0x2).
    b_fp4_low = _u32_packed(0x02)   # value in low nibble
    b_fp4_high = _u32_packed(0x20)  # value in high nibble

    sf_only_low = 0x0000007F   # only low byte carries scale (canonical)

    A = torch.full((32, 4), a_e4m3, device=device, dtype=torch.int32)
    B_e4m3 = torch.full((32, 2), b_e4m3, device=device, dtype=torch.int32)
    B_fp4_low = torch.full((32, 2), b_fp4_low, device=device, dtype=torch.int32)
    B_fp4_high = torch.full((32, 2), b_fp4_high, device=device, dtype=torch.int32)
    SF = torch.tensor([sf_only_low], device=device, dtype=torch.int32)
    D = torch.zeros(32, 4, device=device, dtype=torch.float32)

    cuteA = _to_cute(A, cutlass.Uint32)
    cuteSF = _to_cute(SF, cutlass.Uint32)
    cuteD = _to_cute(D, cutlass.Float32)

    if args.mode in ("mxfp8", "compare"):
        cuteB = _to_cute(B_e4m3, cutlass.Uint32)
        D.zero_()
        compiled = cute.compile(probe_mxfp8_only, cuteA, cuteB, cuteSF, cuteD, stream)
        compiled(cuteA, cuteB, cuteSF, cuteD, stream)
        torch.cuda.synchronize()
        print(f"[mxfp8 e4m3.e4m3]  D[0,0]={D[0,0].item():.4f}  D[0,1]={D[0,1].item():.4f}  D[0,2]={D[0,2].item():.4f}  D[0,3]={D[0,3].item():.4f}  (expect ~32)")

    if args.mode in ("mxfp4", "compare"):
        # Test: zero out b1 to see if it matters. If D unchanged, b1 is unused
        # (i.e., FP4 K_eff per call is 8 not 32).
        for label, b0, b1 in [
            ("b0=0x02..., b1=0x02...", _u32_packed(0x02), _u32_packed(0x02)),
            ("b0=0x02..., b1=0",        _u32_packed(0x02), 0),
            ("b0=0,       b1=0x02...", 0,                  _u32_packed(0x02)),
            ("b0=0x22..., b1=0x22...", _u32_packed(0x22), _u32_packed(0x22)),
        ]:
            B = torch.tensor([[b0, b1]] * 32, device=device, dtype=torch.int32)
            cuteB = _to_cute(B, cutlass.Uint32)
            D.zero_()
            compiled = cute.compile(probe_mxfp4_mxfp8, cuteA, cuteB, cuteSF, cuteD, stream)
            compiled(cuteA, cuteB, cuteSF, cuteD, stream)
            torch.cuda.synchronize()
            print(f"[{label}]  D[0,0]={D[0,0].item():>9.4f}  D[0,1]={D[0,1].item():>9.4f}")


if __name__ == "__main__":
    main()
