"""Compiled CuTe DSL kernel callers for MXFP4×MXFP8.

Wraps `cute.compile + invoke` for the single-tile (M=16, N=8) GEMM so callers
can treat it as a regular Python function. Uses the validated cvt+mxfp8 path.
"""

from __future__ import annotations

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import (
    cvt_e2m1x8_to_e4m3x8,
    mxfp8_mma_m16n8k32_f32_e4m3,
)
from b12x.moe.fused.mxfp4_mxfp8.single_tile import (
    pack_a_fragments,
    pack_b_fragments,
    pack_sfa,
    pack_sfb,
    unpack_d,
)


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


@cute.jit
def _gemm_kernel(mA, mBp, mSFa, mSFb, mD, stream: cuda.CUstream):
    _gemm_warp(mA, mBp, mSFa, mSFb, mD).launch(
        grid=(1, 1, 1), block=[32, 1, 1], stream=stream,
    )


@cute.kernel
def _gemm_warp(mA, mBp, mSFa, mSFb, mD):
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
    mD[tidx, 0] = d0; mD[tidx, 1] = d1; mD[tidx, 2] = d2; mD[tidx, 3] = d3


# Cache compiled kernel by K_blocks since cute bakes shape into the JIT.
_compile_cache = {}


def _compiled_for(K_blocks: int):
    if K_blocks in _compile_cache:
        return _compile_cache[K_blocks]
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    A = torch.zeros(32, K_blocks, 4, dtype=torch.int32, device=device)
    Bp = torch.zeros(32, K_blocks, 1, dtype=torch.int32, device=device)
    SFa = torch.zeros(32, K_blocks, dtype=torch.int32, device=device)
    SFb = torch.zeros(32, K_blocks, dtype=torch.int32, device=device)
    D = torch.zeros(32, 4, dtype=torch.float32, device=device)
    compiled = cute.compile(
        _gemm_kernel,
        _to_cute(A, cutlass.Uint32),
        _to_cute(Bp, cutlass.Uint32),
        _to_cute(SFa, cutlass.Uint32),
        _to_cute(SFb, cutlass.Uint32),
        _to_cute(D, cutlass.Float32),
        stream,
    )
    _compile_cache[K_blocks] = compiled
    return compiled


def run_single_tile(
    A_e4m3: torch.Tensor,
    A_sf: torch.Tensor,
    B_packed: torch.Tensor,
    B_sf: torch.Tensor,
) -> torch.Tensor:
    """Run one m16n8 GEMM tile. Returns (16, 8) fp32 output.

    Args:
        A_e4m3: (16, K) E4M3 byte (uint8 view).
        A_sf:   (16, K_blocks) UE8M0 byte (uint8).
        B_packed: (8, K/2) packed E2M1 (uint8).
        B_sf:     (8, K_blocks) UE8M0 byte (uint8).
    """
    M, K = A_e4m3.shape
    assert M == 16
    K_blocks = K // 32

    A_frag = pack_a_fragments(A_e4m3)
    B_frag = pack_b_fragments(B_packed)
    SFa_frag = pack_sfa(A_sf)
    SFb_frag = pack_sfb(B_sf)
    D_per_thread = torch.zeros(32, 4, dtype=torch.float32, device=A_e4m3.device)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _compiled_for(K_blocks)
    compiled(
        _to_cute(A_frag, cutlass.Uint32),
        _to_cute(B_frag, cutlass.Uint32),
        _to_cute(SFa_frag, cutlass.Uint32),
        _to_cute(SFb_frag, cutlass.Uint32),
        _to_cute(D_per_thread, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize()
    return unpack_d(D_per_thread)
