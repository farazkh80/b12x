"""Phase-4 routed MoE: top-k routing on top of the single-expert kernel chain.

Compares `routed_moe_b12x` against `moe_reference_mxfp4_mxfp8` for small but
non-trivial gpt-oss-shaped configurations.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine,
    moe_reference_mxfp4_mxfp8,
    quantize_to_mxfp4,
)
from b12x.moe.fused.mxfp4_mxfp8.routed import routed_moe_b12x


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.mark.parametrize("activation", ["silu", "relu2"])
@pytest.mark.parametrize("T,E,top_k,K,I", [
    (4,  4, 2,  64, 32),     # tiny
    (8,  8, 2,  64, 32),     # more experts
    (16, 4, 2, 128, 64),     # bigger K, I
    (16, 8, 4, 128, 64),     # gpt-oss-like top_k=4
])
def test_routed_moe_matches_reference(device, activation, T, E, top_k, K, I):
    torch.manual_seed(1)
    K_out = K
    FC1_N = 2 * I if activation == "silu" else I

    x_bf16 = torch.randn(T, K, dtype=torch.bfloat16, device=device) * 0.5

    w13_f32 = torch.randn(E, FC1_N, K, dtype=torch.float32, device=device) * 0.3
    w2_f32  = torch.randn(E, K_out, I, dtype=torch.float32, device=device) * 0.3
    w13_packed = torch.empty(E, FC1_N, K // 2, dtype=torch.uint8, device=device)
    w13_sf     = torch.empty(E, FC1_N, K // 32, dtype=torch.uint8, device=device)
    w2_packed  = torch.empty(E, K_out, I // 2, dtype=torch.uint8, device=device)
    w2_sf      = torch.empty(E, K_out, I // 32, dtype=torch.uint8, device=device)
    for e in range(E):
        w13_packed[e], w13_sf[e] = quantize_to_mxfp4(w13_f32[e])
        w2_packed[e],  w2_sf[e]  = quantize_to_mxfp4(w2_f32[e])

    routing_logits = torch.randn(T, E, dtype=torch.float32, device=device)
    topk_logits, topk_ids = torch.topk(routing_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    ref = moe_reference_mxfp4_mxfp8(
        x_bf16, w13_packed, w13_sf, w2_packed, w2_sf,
        topk_ids, topk_weights, activation=activation,
    )
    actual = routed_moe_b12x(
        x_bf16, w13_packed, w13_sf, w2_packed, w2_sf,
        topk_ids, topk_weights, activation=activation,
    )

    err = (actual.to(torch.float32) - ref.out_bf16.to(torch.float32)).abs()
    cos = cosine(actual, ref.out_bf16)
    print(f"\n[T={T} E={E} top_k={top_k} K={K} I={I} act={activation}] cos={cos:.6f} max_abs={err.max().item():.4f} rmse={err.square().mean().sqrt().item():.4f}")
    assert cos > 0.998, f"cosine {cos:.6f} too low for T={T} E={E} top_k={top_k} K={K} I={I} act={activation}"
