"""N-tile-chunked single-warp fused kernel — bit-exact vs single-warp cp.async."""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused.mxfp4_mxfp8._fused_kernel import (
    run_fused_silu_chunked,
    run_fused_silu_cpasync,
)
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
    # Smallest single-chunk shape (chunk_count=1 in both FC1 and FC2):
    ( 16, 32, 32, 32, 4),
    ( 64, 32, 32, 32, 4),
    # Multiple FC1 chunks (I_k_blocks > 1):
    ( 16, 32, 64, 32, 4),     # 2 FC1 chunks
    ( 16, 32, 128, 32, 4),    # 4 FC1 chunks
    # Multiple FC2 chunks (K_out_n_tiles > fc2_chunk):
    ( 16, 32, 32, 64, 4),     # 2 FC2 chunks (fc2_chunk=4 → K_out=64 = 8 N-tiles → 2 chunks)
    ( 16, 32, 32, 128, 4),    # 4 FC2 chunks
    # Both >1:
    ( 16, 32, 64, 64, 4),
    (128, 64, 64, 64, 4),
    # Larger fc2 chunk:
    ( 16, 32, 32, 64, 8),
])
def test_chunked_matches_cpasync(device, M, K_in, I, K_out, fc2_chunk):
    """Chunked variant must match single-warp cp.async path bit-exact."""
    torch.manual_seed(31)
    FC1_N = 2 * I
    x_f32 = torch.randn(M, K_in, dtype=torch.float32, device=device) * 0.5
    w13_f32 = torch.randn(FC1_N, K_in, dtype=torch.float32, device=device) * 0.3
    w2_f32 = torch.randn(K_out, I, dtype=torch.float32, device=device) * 0.3

    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_f32)
    w2_p, w2_sf = quantize_to_mxfp4(w2_f32)

    ref = run_fused_silu_cpasync(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)
    out = run_fused_silu_chunked(
        x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf,
        fc2_chunk_n_tiles=fc2_chunk,
    )

    assert out.shape == ref.shape == (M, K_out)
    err = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    cos = cosine(out, ref)
    print(f"\n[chunked  M={M} K_in={K_in} I={I} K_out={K_out} fc2_chunk={fc2_chunk}]")
    print(f"  cos={cos:.6f}  max_abs={err.max().item():.4f}")
    assert cos > 0.9999, f"chunked drifted: cos={cos:.6f}"
    assert err.max().item() == 0.0, f"chunked must be bit-exact: max_abs={err.max().item()}"
