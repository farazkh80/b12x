#!/usr/bin/env python3
"""Pin down which byte of the per-thread sfa u32 corresponds to which A row.

Strategy: A and B are all 1.0. Set sfa byte 0 to a non-trivial scale s0 (e.g.
0x80 = 2^1 = 2.0); set sfa bytes 1, 2, 3 each to a distinct value sN (other
non-trivial scales). Read back the kernel's 4 D outputs per thread:
  d0 (row=t/4, col=t%4*2+0)
  d1 (row=t/4, col=t%4*2+1)
  d2 (row=t/4+8, col=t%4*2+0)
  d3 (row=t/4+8, col=t%4*2+1)
Whichever byte affected d0/d1 is the byte for row t/4. Whichever byte
affected d2/d3 is the byte for row t/4+8.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import mxfp8_mma_m16n8k32_f32_e4m3


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


@cute.jit
def probe(mA, mB, mSFa, mSFb, mD, stream: cuda.CUstream):
    probe_kernel(mA, mB, mSFa, mSFb, mD).launch(grid=(1,1,1), block=[32,1,1], stream=stream)


@cute.kernel
def probe_kernel(mA, mB, mSFa, mSFb, mD):
    tidx = cute.arch.thread_idx()[0]
    a0 = mA[tidx, 0]; a1 = mA[tidx, 1]; a2 = mA[tidx, 2]; a3 = mA[tidx, 3]
    b0 = mB[tidx, 0]; b1 = mB[tidx, 1]
    sfa = mSFa[tidx]
    sfb = mSFb[tidx]
    d0 = Float32(0.0); d1 = Float32(0.0); d2 = Float32(0.0); d3 = Float32(0.0)
    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
        d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
    )
    mD[tidx, 0] = d0; mD[tidx, 1] = d1; mD[tidx, 2] = d2; mD[tidx, 3] = d3


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    one = 0x38  # E4M3 1.0
    a_packed = one | (one << 8) | (one << 16) | (one << 24)
    A = torch.full((32, 4), a_packed, device=device, dtype=torch.int32)
    B = torch.full((32, 2), a_packed, device=device, dtype=torch.int32)
    D = torch.zeros(32, 4, device=device, dtype=torch.float32)

    cuteA = _to_cute(A, cutlass.Uint32)
    cuteB = _to_cute(B, cutlass.Uint32)
    cuteD = _to_cute(D, cutlass.Float32)

    # Sweep: which byte of sfa scales d0..d3? Set ONE byte to 0x80 (=2.0), rest to 0x7F (=1.0).
    print("Setting sfb=0x7F broadcast (=1.0) for all threads.")
    SFb = torch.full((32,), 0x7F, device=device, dtype=torch.int32)
    cuteSFb = _to_cute(SFb, cutlass.Uint32)

    print("\nProbing which thread's sfa byte0 affects which D position (A=B=1.0, K=32 base D=32):")
    print("Set ONE thread's sfa to 0x80 (=2.0); rest stay 0x7F. Read which D positions doubled.")
    base_sfa = torch.full((32,), 0x7F, device=device, dtype=torch.int32)
    for set_thread in [0, 1, 4, 8, 12, 16, 20, 24, 28]:
        SFa = base_sfa.clone()
        SFa[set_thread] = 0x80
        cuteSFa = _to_cute(SFa, cutlass.Uint32)
        D.zero_()
        compiled = cute.compile(probe, cuteA, cuteB, cuteSFa, cuteSFb, cuteD, stream)
        compiled(cuteA, cuteB, cuteSFa, cuteSFb, cuteD, stream)
        torch.cuda.synchronize()
        # find which (thread, d_idx) positions are not the base 32.
        D_cpu = D.cpu()
        affected = []
        for tid in range(32):
            for di in range(4):
                v = D_cpu[tid, di].item()
                if abs(v - 32.0) > 0.5:
                    affected.append((tid, di, v))
        print(f"  thread {set_thread:>2} sfa=0x80 → {len(affected):>3} positions changed: {affected[:6]}")


if __name__ == "__main__":
    main()
