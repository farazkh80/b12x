"""Debug: same fused kernel structure but bypass SwiGLU+amax+quant.
Just cast FC1 D fragments (the FIRST 4 of the 8 N-tiles, treating them as
the intermediate without SwiGLU) directly to E4M3 with scale=1.0, then
do FC2. Validates the SMEM round-trip + FC2 path independently of the
activation/quant complexity.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Uint8, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import cvt_e2m1x8_to_e4m3x8, cvt_f32_to_e4m3, mxfp8_mma_m16n8k32_f32_e4m3
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine, dequant_mxfp4, dequant_mxfp8, quantize_to_mxfp4, quantize_to_mxfp8,
)
from b12x.moe.fused.mxfp4_mxfp8.single_tile import (
    pack_a_fragments, pack_b_fragments, pack_sfa, pack_sfb,
)


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


I_N_TILES = 4
K_OUT_N_TILES = 4


@cute.jit
def fused_bypass(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY, stream):
    fused_warp(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY).launch(
        grid=(1,1,1), block=[32,1,1], stream=stream,
    )


@cute.kernel
def fused_warp(mA, mA_sf, mW13, mW13_sf, mW2, mW2_sf, mY):
    tidx = cute.arch.thread_idx()[0]

    smem = utils.SmemAllocator()

    @cute.struct
    class Storage:
        sInt: cute.struct.MemRange[Uint8, 16 * 32]

    storage = smem.allocate(Storage)
    sInt = storage.sInt.get_tensor(cute.make_layout((16, 32)))

    # Full FC1 across 8 N-tiles (gate + up).
    a0 = mA[tidx, 0, 0]; a1 = mA[tidx, 0, 1]; a2 = mA[tidx, 0, 2]; a3 = mA[tidx, 0, 3]
    sfa = mA_sf[tidx, 0]

    fc1 = [Float32(0.0) for _ in range(8 * 4)]
    for n in cutlass.range_constexpr(8):
        bp = mW13[tidx, n, 0, 0]
        b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)
        sfb = mW13_sf[tidx, n, 0]
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            fc1[n*4+0], fc1[n*4+1], fc1[n*4+2], fc1[n*4+3],
            a0, a1, a2, a3, b0, b1, sfa, sfb,
        )
        fc1[n*4+0] = d0; fc1[n*4+1] = d1; fc1[n*4+2] = d2; fc1[n*4+3] = d3

    # SwiGLU: intermediate = silu(up) * gate.
    intermediate = [Float32(0.0) for _ in range(I_N_TILES * 4)]
    for n in cutlass.range_constexpr(I_N_TILES):
        for i in cutlass.range_constexpr(4):
            gate = fc1[n*4+i]
            up = fc1[(n + I_N_TILES)*4+i]
            sigmoid_up = Float32(1.0) / (Float32(1.0) + cute.math.exp(-up, fastmath=True))
            intermediate[n*4+i] = gate * up * sigmoid_up

    # NEW: compute amax + UE8M0 scale + scaled quant.
    from b12x.moe.fused.mxfp4_mxfp8._quant_kernel import floor_f32_to_s32, imax_s32, imin_s32
    from b12x.cute.fp4 import warp_reduce
    local_max_top = Float32(0.0)
    local_max_bot = Float32(0.0)
    for n in cutlass.range_constexpr(I_N_TILES):
        for i in cutlass.range_constexpr(2):
            v_top = intermediate[n*4 + i]
            local_max_top = cute.arch.fmax(local_max_top, cute.arch.fmax(v_top, -v_top))
            v_bot = intermediate[n*4 + 2 + i]
            local_max_bot = cute.arch.fmax(local_max_bot, cute.arch.fmax(v_bot, -v_bot))
    row_top_amax = warp_reduce(local_max_top, lambda a, b: cute.arch.fmax(a, b), width=4)
    row_bot_amax = warp_reduce(local_max_bot, lambda a, b: cute.arch.fmax(a, b), width=4)

    inv_max = Float32(1.0) / Float32(448.0)
    safe_top = cute.arch.fmax(row_top_amax * inv_max, Float32(1.401298464e-45))
    safe_bot = cute.arch.fmax(row_bot_amax * inv_max, Float32(1.401298464e-45))
    log_top = cute.math.log2(safe_top, fastmath=True)
    log_bot = cute.math.log2(safe_bot, fastmath=True)
    ceil_top = -floor_f32_to_s32(-log_top)
    ceil_bot = -floor_f32_to_s32(-log_bot)
    sf_top = imin_s32(Int32(254), imax_s32(Int32(0), ceil_top + Int32(127)))
    sf_bot = imin_s32(Int32(254), imax_s32(Int32(0), ceil_bot + Int32(127)))
    inv_scale_top = Float32(1.0) / cute.math.exp2(Float32(sf_top - Int32(127)), fastmath=True)
    inv_scale_bot = Float32(1.0) / cute.math.exp2(Float32(sf_bot - Int32(127)), fastmath=True)

    e4m3_bytes = [Uint32(0) for _ in range(I_N_TILES * 4)]
    for n in cutlass.range_constexpr(I_N_TILES):
        for i in cutlass.range_constexpr(2):
            e4m3_bytes[n*4 + i]     = cvt_f32_to_e4m3(intermediate[n*4 + i] * inv_scale_top)
            e4m3_bytes[n*4 + 2 + i] = cvt_f32_to_e4m3(intermediate[n*4 + 2 + i] * inv_scale_bot)

    # Write to SMEM in (16, 32) row-major layout.
    g = Int32(tidx) // Int32(4)
    lane = Int32(tidx) % Int32(4)
    for n in cutlass.range_constexpr(I_N_TILES):
        for i in cutlass.range_constexpr(2):
            col = lane * Int32(2) + Int32(n * 8) + Int32(i)
            sInt[g,            col] = Uint8(e4m3_bytes[n*4+i])
            sInt[g + Int32(8), col] = Uint8(e4m3_bytes[n*4+2+i])

    cute.arch.sync_warp()

    # Read FC2 A fragment.
    fc2_a0 = Uint32(0); fc2_a1 = Uint32(0); fc2_a2 = Uint32(0); fc2_a3 = Uint32(0)
    for i in cutlass.range_constexpr(4):
        col_lo = lane * Int32(8) + Int32(i)
        col_hi = lane * Int32(8) + Int32(4 + i)
        fc2_a0 = fc2_a0 | (Uint32(sInt[g,            col_lo]) << Uint32(i*8))
        fc2_a1 = fc2_a1 | (Uint32(sInt[g + Int32(8), col_lo]) << Uint32(i*8))
        fc2_a2 = fc2_a2 | (Uint32(sInt[g,            col_hi]) << Uint32(i*8))
        fc2_a3 = fc2_a3 | (Uint32(sInt[g + Int32(8), col_hi]) << Uint32(i*8))

    # FC2 SFA: per-row UE8M0 (lane 0 → row g, lane 1 → row g+8).
    sfa_fc2 = Uint32(0x7F)
    if lane == Int32(0):
        sfa_fc2 = Uint32(sf_top)
    elif lane == Int32(1):
        sfa_fc2 = Uint32(sf_bot)

    fc2 = [Float32(0.0) for _ in range(K_OUT_N_TILES * 4)]
    for n in cutlass.range_constexpr(K_OUT_N_TILES):
        bp = mW2[tidx, n, 0, 0]
        b0, b1 = cvt_e2m1x8_to_e4m3x8(bp)
        sfb = mW2_sf[tidx, n, 0]
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            fc2[n*4+0], fc2[n*4+1], fc2[n*4+2], fc2[n*4+3],
            fc2_a0, fc2_a1, fc2_a2, fc2_a3, b0, b1, sfa_fc2, sfb,
        )
        fc2[n*4+0] = d0; fc2[n*4+1] = d1; fc2[n*4+2] = d2; fc2[n*4+3] = d3

    for n in cutlass.range_constexpr(K_OUT_N_TILES):
        col0 = Int32(n*8) + lane * Int32(2)
        mY[g,            col0]            = BFloat16(fc2[n*4+0])
        mY[g,            col0 + Int32(1)] = BFloat16(fc2[n*4+1])
        mY[g + Int32(8), col0]            = BFloat16(fc2[n*4+2])
        mY[g + Int32(8), col0 + Int32(1)] = BFloat16(fc2[n*4+3])


def main():
    torch.manual_seed(3)
    device = torch.device("cuda")
    M, K_in, I, K_out, FC1_N = 16, 32, 32, 32, 64  # full SwiGLU now
    x_f32 = torch.randn(M, K_in, dtype=torch.float32, device=device) * 0.5
    w13_f32 = torch.randn(FC1_N, K_in, dtype=torch.float32, device=device) * 0.3
    w2_f32 = torch.randn(K_out, I, dtype=torch.float32, device=device) * 0.3

    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_f32)
    w2_p, w2_sf = quantize_to_mxfp4(w2_f32)

    A_frag = pack_a_fragments(x_e4m3)
    A_sf_frag = pack_sfa(x_sf)
    W13_frag = torch.cat([pack_b_fragments(w13_p[i:i+8]).unsqueeze(1) for i in range(0, FC1_N, 8)], dim=1)
    W13_sf_frag = torch.cat([pack_sfb(w13_sf[i:i+8]).unsqueeze(1) for i in range(0, FC1_N, 8)], dim=1)
    W2_frag = torch.cat([pack_b_fragments(w2_p[i:i+8]).unsqueeze(1) for i in range(0, K_out, 8)], dim=1)
    W2_sf_frag = torch.cat([pack_sfb(w2_sf[i:i+8]).unsqueeze(1) for i in range(0, K_out, 8)], dim=1)

    Y = torch.zeros(16, K_out, dtype=torch.bfloat16, device=device)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        fused_bypass,
        _to_cute(A_frag, cutlass.Uint32), _to_cute(A_sf_frag, cutlass.Uint32),
        _to_cute(W13_frag, cutlass.Uint32), _to_cute(W13_sf_frag, cutlass.Uint32),
        _to_cute(W2_frag, cutlass.Uint32), _to_cute(W2_sf_frag, cutlass.Uint32),
        _to_cute(Y, cutlass.BFloat16), stream,
    )
    compiled(
        _to_cute(A_frag, cutlass.Uint32), _to_cute(A_sf_frag, cutlass.Uint32),
        _to_cute(W13_frag, cutlass.Uint32), _to_cute(W13_sf_frag, cutlass.Uint32),
        _to_cute(W2_frag, cutlass.Uint32), _to_cute(W2_sf_frag, cutlass.Uint32),
        _to_cute(Y, cutlass.BFloat16), stream,
    )
    torch.cuda.synchronize()

    print("Y any nan:", Y.isnan().any().item(), " max abs:", Y.abs().max().item())
    print("Y[0, :8]:", Y[0, :8].tolist())

    # Reference: full chain via the validated unfused expert_chain_b12x.
    from b12x.moe.fused.mxfp4_mxfp8.expert_chain import expert_chain_b12x
    chain_res = expert_chain_b12x(
        x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf, activation="silu",
    )
    ref = chain_res.out_bf16

    cos = cosine(Y, ref)
    err = (Y.to(torch.float32) - ref.to(torch.float32)).abs()
    print(f"cosine vs bypass-ref: {cos:.6f}  max_abs: {err.max().item():.4f}")


if __name__ == "__main__":
    main()
