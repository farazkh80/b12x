"""Debug: short-circuit fused kernel to write FC1 N-tile 0 output directly."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import (
    cvt_e2m1x8_to_e4m3x8,
    mxfp8_mma_m16n8k32_f32_e4m3,
)
from b12x.moe.fused.mxfp4_mxfp8.reference import quantize_to_mxfp4, quantize_to_mxfp8
from b12x.moe.fused.mxfp4_mxfp8.single_tile import (
    pack_a_fragments, pack_b_fragments, pack_sfa, pack_sfb,
)
from b12x.moe.fused.mxfp4_mxfp8._kernel import run_single_tile

def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t

@cute.jit
def dbg(mA, mA_sf, mW13, mW13_sf, mY, stream):
    dbg_warp(mA, mA_sf, mW13, mW13_sf, mY).launch(grid=(1,1,1), block=[32,1,1], stream=stream)

@cute.kernel
def dbg_warp(mA, mA_sf, mW13, mW13_sf, mY):
    tidx = cute.arch.thread_idx()[0]
    a0 = mA[tidx, 0, 0]; a1 = mA[tidx, 0, 1]; a2 = mA[tidx, 0, 2]; a3 = mA[tidx, 0, 3]
    sfa = mA_sf[tidx, 0]
    bp = mW13[tidx, 0, 0, 0]
    b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)
    sfb = mW13_sf[tidx, 0, 0]
    d0 = Float32(0.0); d1 = Float32(0.0); d2 = Float32(0.0); d3 = Float32(0.0)
    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb)
    g = Int32(tidx) // Int32(4)
    lane = Int32(tidx) % Int32(4)
    col = lane * Int32(2)
    mY[g, col] = BFloat16(d0)
    mY[g, col + Int32(1)] = BFloat16(d1)
    mY[g + Int32(8), col] = BFloat16(d2)
    mY[g + Int32(8), col + Int32(1)] = BFloat16(d3)


def main():
    torch.manual_seed(3)
    device = torch.device("cuda")
    M, K_in = 16, 32
    x_f32 = torch.randn(M, K_in, dtype=torch.float32, device=device) * 0.5
    w13_tile_f32 = torch.randn(8, K_in, dtype=torch.float32, device=device) * 0.3

    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_tile_f32)

    A_frag = pack_a_fragments(x_e4m3)
    A_sf_frag = pack_sfa(x_sf)
    W13_frag = pack_b_fragments(w13_p).unsqueeze(1)
    W13_sf_frag = pack_sfb(w13_sf).unsqueeze(1)

    Y = torch.zeros(16, 8, dtype=torch.bfloat16, device=device)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        dbg,
        _to_cute(A_frag, cutlass.Uint32), _to_cute(A_sf_frag, cutlass.Uint32),
        _to_cute(W13_frag, cutlass.Uint32), _to_cute(W13_sf_frag, cutlass.Uint32),
        _to_cute(Y, cutlass.BFloat16), stream,
    )
    compiled(
        _to_cute(A_frag, cutlass.Uint32), _to_cute(A_sf_frag, cutlass.Uint32),
        _to_cute(W13_frag, cutlass.Uint32), _to_cute(W13_sf_frag, cutlass.Uint32),
        _to_cute(Y, cutlass.BFloat16), stream,
    )
    torch.cuda.synchronize()

    ref = run_single_tile(x_e4m3, x_sf, w13_p, w13_sf)
    print("ref     [0,:8]:", [f"{v:7.3f}" for v in ref[0].tolist()])
    print("kernel  [0,:8]:", [f"{v:7.3f}" for v in Y[0].tolist()])
    err = (Y.to(torch.float32) - ref).abs().max().item()
    print(f"max abs err: {err:.4f}")


if __name__ == "__main__":
    main()
