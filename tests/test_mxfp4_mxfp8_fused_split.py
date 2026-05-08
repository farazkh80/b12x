"""Two-kernel split (FC1 + SwiGLU + Quant kernel, then FC2 kernel).

Multi-CTA over (M_tiles, I_k_blocks) for FC1 and (M_tiles, K_out_chunks) for
FC2. Must match the chunked single-warp baseline bit-exact at moderate shapes.
"""
from __future__ import annotations

import pytest
import torch

from b12x.moe.fused.mxfp4_mxfp8._fused_kernel import run_fused_silu_chunked
from b12x.moe.fused.mxfp4_mxfp8._split_kernel import run_fused_silu_split
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine,
    quantize_to_mxfp4,
    quantize_to_mxfp8,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.mark.parametrize("M,K_in,I,K_out,fc2_chunk", [
    ( 16, 32, 32, 32, 4),
    ( 64, 32, 32, 32, 4),
    ( 16, 32, 64, 32, 4),
    ( 16, 32, 128, 32, 4),
    ( 16, 32, 32, 64, 4),
    ( 16, 32, 32, 128, 4),
    ( 16, 32, 64, 64, 4),
    (128, 64, 64, 64, 4),
    ( 16, 32, 32, 64, 8),
])
def test_split_matches_chunked(device, M, K_in, I, K_out, fc2_chunk):
    """Split (2-kernel) must match the chunked single-warp baseline bit-exact."""
    torch.manual_seed(31)
    FC1_N = 2 * I
    x_f32 = torch.randn(M, K_in, dtype=torch.float32, device=device) * 0.5
    w13_f32 = torch.randn(FC1_N, K_in, dtype=torch.float32, device=device) * 0.3
    w2_f32 = torch.randn(K_out, I, dtype=torch.float32, device=device) * 0.3

    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_f32)
    w2_p, w2_sf = quantize_to_mxfp4(w2_f32)

    ref = run_fused_silu_chunked(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf, fc2_chunk_n_tiles=fc2_chunk)
    out = run_fused_silu_split(
        x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf,
        fc2_chunk_n_tiles=fc2_chunk,
    )

    assert out.shape == ref.shape == (M, K_out)
    err = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    cos = cosine(out, ref)
    print(f"\n[split  M={M} K_in={K_in} I={I} K_out={K_out} fc2_chunk={fc2_chunk}]")
    print(f"  cos={cos:.6f}  max_abs={err.max().item():.4f}")
    assert cos > 0.9999, f"split drifted: cos={cos:.6f}"
    assert err.max().item() == 0.0, f"split must be bit-exact: max_abs={err.max().item()}"
