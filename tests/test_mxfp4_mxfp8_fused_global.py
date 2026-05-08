"""Fused single-launch kernel with row-major global loads (no host packing)."""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused.mxfp4_mxfp8._fused_kernel import (
    run_fused_silu_full,
    run_fused_silu_global,
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


@pytest.mark.parametrize("M,K_in,I,K_out", [
    ( 16, 32, 32, 32),
    ( 64, 32, 32, 32),
    ( 80, 64, 64, 64),   # padding path
    (128, 64, 64, 64),
    (256, 64, 64, 64),
    ( 64, 128, 64, 64),  # K_in > 1 block
])
def test_fused_global_matches_prepack(device, M, K_in, I, K_out):
    """Global-load variant should match the prepacked-fragment variant exactly."""
    torch.manual_seed(11)
    FC1_N = 2 * I
    x_f32 = torch.randn(M, K_in, dtype=torch.float32, device=device) * 0.5
    w13_f32 = torch.randn(FC1_N, K_in, dtype=torch.float32, device=device) * 0.3
    w2_f32 = torch.randn(K_out, I, dtype=torch.float32, device=device) * 0.3

    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_f32)
    w2_p, w2_sf = quantize_to_mxfp4(w2_f32)

    ref = run_fused_silu_full(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)
    out = run_fused_silu_global(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)

    assert out.shape == ref.shape == (M, K_out)
    err = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    cos = cosine(out, ref)
    print(f"\n[global vs prepack  M={M} K_in={K_in} I={I} K_out={K_out}]")
    print(f"  cos={cos:.6f}  max_abs={err.max().item():.4f}")
    # Both run the same kernel logic — the only difference is where fragments
    # are assembled (host vs in-kernel). Should be bit-exact.
    assert cos > 0.9999, f"global drifted from prepack: cos={cos:.6f}"
    assert err.max().item() == 0.0, f"global vs prepack should be bit-exact, max_abs={err.max().item()}"
