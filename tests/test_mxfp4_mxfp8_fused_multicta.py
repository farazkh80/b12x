"""Multi-CTA fused kernel (M ≥ 16) — validate against the unfused chain."""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused.mxfp4_mxfp8._fused_kernel import run_fused_silu_full
from b12x.moe.fused.mxfp4_mxfp8.expert_chain import expert_chain_b12x
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


def _expert_chain_full(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf, *, activation="silu"):
    """Reference: chunk M into 16-row pieces and use the validated single-tile chain."""
    M, K_in = x_e4m3.shape
    K_out = w2_p.shape[0]
    out = torch.zeros(M, K_out, dtype=torch.bfloat16, device=x_e4m3.device)
    for c in range(0, M, 16):
        end = min(c + 16, M)
        chunk_x = x_e4m3[c:end]
        chunk_sf = x_sf[c:end]
        res = expert_chain_b12x(
            chunk_x, chunk_sf, w13_p, w13_sf, w2_p, w2_sf, activation=activation,
        )
        out[c:end] = res.out_bf16
    return out


@pytest.mark.parametrize("M,K_in,I,K_out", [
    ( 16, 32, 32, 32),   # baseline single CTA
    ( 32, 32, 32, 32),   # 2 CTAs
    ( 64, 32, 32, 32),   # 4 CTAs
    (128, 64, 32, 32),   # 8 CTAs, K_in > 1 block
    ( 80, 64, 64, 64),   # M not divisible by 16 → padding path
    (256, 64, 64, 64),   # 16 CTAs, all dims > 1 block
])
def test_fused_multicta(device, M, K_in, I, K_out):
    torch.manual_seed(7)
    FC1_N = 2 * I

    x_f32 = torch.randn(M, K_in, dtype=torch.float32, device=device) * 0.5
    w13_f32 = torch.randn(FC1_N, K_in, dtype=torch.float32, device=device) * 0.3
    w2_f32 = torch.randn(K_out, I, dtype=torch.float32, device=device) * 0.3

    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_f32)
    w2_p, w2_sf = quantize_to_mxfp4(w2_f32)

    ref = _expert_chain_full(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf, activation="silu")
    out = run_fused_silu_full(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)

    assert out.shape == (M, K_out)
    cos = cosine(out, ref)
    err = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    print(f"\n[multi-CTA fused vs chain  M={M} K_in={K_in} I={I} K_out={K_out}  CTAs={(M+15)//16}]")
    print(f"  cos={cos:.6f}  max_abs={err.max().item():.4f}  rmse={err.square().mean().sqrt().item():.4f}")
    assert cos > 0.998, f"multi-CTA fused drifted: cos={cos:.6f}"
