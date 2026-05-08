"""Single-kernel fused FC1+SwiGLU+MXFP8-quant+FC2 — generalized shapes.

Validates the fused kernel against the validated unfused expert_chain across
a (K_in, I, K_out) sweep at M=16.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused.mxfp4_mxfp8._fused_kernel import run_fused_silu
from b12x.moe.fused.mxfp4_mxfp8.expert_chain import expert_chain_b12x
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine,
    quantize_to_mxfp4,
    quantize_to_mxfp8,
)
from b12x.moe.fused.mxfp4_mxfp8.single_tile import (
    pack_a_fragments,
    pack_b_fragments,
    pack_sfa,
    pack_sfb,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _pack_w_per_tile(w_packed, w_sf, n_dim, k_dim):
    """(N, K/2), (N, K/32) → per-thread fragments per N-tile."""
    K_blocks = k_dim // 32
    N_tiles = n_dim // 8
    K_blocks_chk = w_sf.shape[1]
    assert K_blocks_chk == K_blocks, f"w_sf K_blocks {K_blocks_chk} vs expected {K_blocks}"

    frags = torch.empty(32, N_tiles, K_blocks, 1,
                        dtype=torch.int32, device=w_packed.device)
    sf_frags = torch.empty(32, N_tiles, K_blocks,
                           dtype=torch.int32, device=w_packed.device)
    for n_idx, n_off in enumerate(range(0, n_dim, 8)):
        tile_packed = w_packed[n_off:n_off + 8]      # (8, K/2)
        tile_sf = w_sf[n_off:n_off + 8]              # (8, K/32)
        frags[:, n_idx] = pack_b_fragments(tile_packed)         # (32, K_blocks, 1)
        sf_frags[:, n_idx] = pack_sfb(tile_sf)                  # (32, K_blocks)
    return frags, sf_frags


@pytest.mark.parametrize("K_in,I,K_out", [
    ( 32,  32,  32),  # original tiny case
    ( 64,  32,  32),  # K_in > 1 block
    ( 32,  64,  32),  # I > 1 block — exercises per-K-block intermediate scales
    ( 32,  32,  64),  # K_out > 4 N-tiles
    ( 64,  64,  64),  # all three > 1 block
    (128,  64,  64),  # bigger K_in
    (256, 128, 128),  # stresses register pressure (FC1 fragment = 32 N-tiles × 4 fp32 = 128/thread)
    (128,  32, 128),  # tall N (16 K_out N-tiles)
])
def test_fused_silu_shape_sweep(device, K_in, I, K_out):
    torch.manual_seed(5)
    M = 16
    FC1_N = 2 * I
    assert K_in  % 32 == 0
    assert I     % 32 == 0
    assert K_out %  8 == 0

    x_f32 = torch.randn(M, K_in, dtype=torch.float32, device=device) * 0.5
    w13_f32 = torch.randn(FC1_N, K_in, dtype=torch.float32, device=device) * 0.3
    w2_f32 = torch.randn(K_out, I, dtype=torch.float32, device=device) * 0.3

    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_f32)
    w2_p, w2_sf = quantize_to_mxfp4(w2_f32)

    # Reference: validated unfused chain.
    ref = expert_chain_b12x(
        x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf, activation="silu",
    )

    # Pack fragments for the fused kernel.
    A_frag = pack_a_fragments(x_e4m3)         # (32, K_in/32, 4)
    A_sf_frag = pack_sfa(x_sf)                # (32, K_in/32)
    W13_frag, W13_sf_frag = _pack_w_per_tile(w13_p, w13_sf, FC1_N, K_in)
    W2_frag, W2_sf_frag = _pack_w_per_tile(w2_p, w2_sf, K_out, I)

    out_fused = run_fused_silu(
        A_frag, A_sf_frag, W13_frag, W13_sf_frag, W2_frag, W2_sf_frag,
    )

    cos = cosine(out_fused, ref.out_bf16)
    err = (out_fused.to(torch.float32) - ref.out_bf16.to(torch.float32)).abs()
    print(f"\n[fused vs chain  K_in={K_in} I={I} K_out={K_out}]")
    print(f"  cos={cos:.6f}  max_abs={err.max().item():.4f}  rmse={err.square().mean().sqrt().item():.4f}")
    assert cos > 0.998, f"fused drifted from chain: cos={cos:.6f}"
