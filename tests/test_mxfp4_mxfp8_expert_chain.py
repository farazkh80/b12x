"""Phase-4 lite test: single-expert FC1+act+FC2 chain on the b12x kernel.

Validates the kernel chain `expert_chain_b12x` against the torch reference
`moe_reference_mxfp4_mxfp8` (constrained to one expert, single-batch).
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused.mxfp4_mxfp8.expert_chain import expert_chain_b12x
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine,
    moe_reference_mxfp4_mxfp8,
    quantize_to_mxfp4,
    quantize_to_mxfp8,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.mark.parametrize("activation", ["silu", "relu2"])
@pytest.mark.parametrize("T,K,I", [
    (1,  64, 32),    # tiny, single token
    (4,  64, 32),    # small batch
    (16, 64, 32),    # full M=16 batch
    (8, 128, 64),    # bigger K and I
])
def test_expert_chain_matches_reference(device, activation, T, K, I):
    torch.manual_seed(0)
    K_out = K  # MoE typically projects back to hidden dim
    FC1_N = 2 * I if activation == "silu" else I

    # Inputs in fp32, scaled to keep MXFP8/MXFP4 quant well-conditioned.
    x_f32 = torch.randn(T, K, dtype=torch.float32, device=device) * 0.5
    w13_f32 = torch.randn(FC1_N, K, dtype=torch.float32, device=device) * 0.3
    w2_f32 = torch.randn(K_out, I, dtype=torch.float32, device=device) * 0.3

    # Quantize once on the host (matches what production would store on disk).
    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_f32)
    w2_p, w2_sf = quantize_to_mxfp4(w2_f32)

    # Reference path: loop over the single expert via moe_reference_mxfp4_mxfp8
    # using top_k=1 routing with weight 1.0. We add a batch dim of 1 expert.
    w13_packed_e = w13_p.unsqueeze(0)        # (E=1, FC1_N, K/2)
    w13_sf_e = w13_sf.unsqueeze(0)
    w2_packed_e = w2_p.unsqueeze(0)
    w2_sf_e = w2_sf.unsqueeze(0)
    topk_ids = torch.zeros(T, 1, dtype=torch.int32, device=device)
    topk_weights = torch.ones(T, 1, dtype=torch.float32, device=device)
    ref = moe_reference_mxfp4_mxfp8(
        x_f32.to(torch.bfloat16), w13_packed_e, w13_sf_e, w2_packed_e, w2_sf_e,
        topk_ids, topk_weights, activation=activation,
    )

    # Kernel path
    res = expert_chain_b12x(
        x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf, activation=activation,
    )

    # Match in bf16 — the small numerical wiggle from fp32 accumulator → bf16 cast
    # is the same on both paths (the reference also casts at the end).
    cos = cosine(res.out_bf16, ref.out_bf16)
    err = (res.out_bf16.to(torch.float32) - ref.out_bf16.to(torch.float32)).abs()
    print(f"\n[T={T} K={K} I={I} act={activation}] cosine={cos:.6f}  max_abs={err.max().item():.4f}  rmse={err.square().mean().sqrt().item():.4f}")
    # 0.998 threshold accounts for the small bf16-cast wiggle and per-row UE8M0
    # rounding boundary effects. Production-grade testing should also assert
    # rmse / max_abs bounds (left for the next layer).
    assert cos > 0.998, f"cosine {cos:.6f} below threshold for T={T} K={K} I={I} act={activation}"
