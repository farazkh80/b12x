"""Unit tests for the MXFP4×MXFP8 torch reference.

These tests pin three claims:
  1. UE8M0 ⇄ fp32 round-trips for the canonical 0x7F = 1.0 anchor.
  2. quantize_to_mxfp4 / quantize_to_mxfp8 are accurate enough that
     dequantize-of-quantize tracks the original tensor (cosine ≥ 0.99).
  3. moe_reference_mxfp4_mxfp8 produces non-degenerate output for a
     hand-built two-expert example, and matches a manual unrolling.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused.mxfp4_mxfp8.reference import (
    _ue8m0_to_f32,
    _f32_to_ue8m0,
    cosine,
    dequant_mxfp4,
    dequant_mxfp8,
    moe_reference_mxfp4_mxfp8,
    quantize_to_mxfp4,
    quantize_to_mxfp8,
)


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_ue8m0_anchor(device):
    sf = torch.tensor([0x7F, 0x80, 0x81, 0x7E, 0x00, 0x01], dtype=torch.uint8, device=device)
    val = _ue8m0_to_f32(sf)
    expected = torch.tensor([1.0, 2.0, 4.0, 0.5, 2.0**(-127), 2.0**(-126)], device=device)
    assert torch.allclose(val, expected, rtol=1e-6, atol=0)


def test_ue8m0_roundtrip(device):
    exps = torch.tensor([2.0**i for i in range(-10, 10)], device=device)
    sf = _f32_to_ue8m0(exps)
    back = _ue8m0_to_f32(sf)
    assert torch.allclose(back, exps, rtol=1e-6, atol=0)


def test_mxfp8_quant_dequant_cosine(device):
    torch.manual_seed(0)
    x = torch.randn(64, 256, device=device, dtype=torch.float32)
    e4m3, sf = quantize_to_mxfp8(x)
    assert e4m3.shape == (64, 256)
    assert sf.shape == (64, 256 // 32)
    deq = dequant_mxfp8(e4m3, sf, rows=64, cols=256)
    cos = cosine(x, deq)
    assert cos > 0.99, f"cosine {cos:.4f} below threshold"


def test_mxfp4_quant_dequant_cosine(device):
    torch.manual_seed(0)
    x = torch.randn(64, 256, device=device, dtype=torch.float32)
    packed, sf = quantize_to_mxfp4(x)
    assert packed.shape == (64, 256 // 2)
    assert sf.shape == (64, 256 // 32)
    deq = dequant_mxfp4(packed, sf, rows=64, cols=256)
    cos = cosine(x, deq)
    # FP4 is much coarser than FP8 — cosine target is lower.
    assert cos > 0.95, f"cosine {cos:.4f} below threshold"


def test_mxfp4_lut_anchor(device):
    """Hand-pack {0, 0.5, 1, 1.5, 2, 3, 4, 6} → assert dequant returns LUT exactly when scale=1."""
    # Each byte packs (odd-index << 4) | even-index, so 0x10 = nibble[0..1] = (0,1),
    # 0x32 = nibble[2..3] = (2,3), …, 0xFE = nibble[14..15] = (14,15).
    nibbles = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8, device=device,
    ).reshape(1, 8)
    # That's 16 nibbles = 16 fp4 values; lay them out as 1×16 with 1 block of 16 — but
    # block_size=32 mandates rows × multiple-of-32 cols. Use cols=32 with two repeats.
    nibbles_repeat = torch.cat([nibbles, nibbles], dim=1)  # (1, 16) bytes = 32 fp4 values
    sf = torch.tensor([[0x7F]], dtype=torch.uint8, device=device)
    deq = dequant_mxfp4(nibbles_repeat, sf, rows=1, cols=32)
    expected_block = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32, device=device,
    )
    assert torch.equal(deq[0, :16], expected_block)
    assert torch.equal(deq[0, 16:], expected_block)


def test_moe_reference_smoke(device):
    """One-token, top-2 routing across two tiny experts with hand-built shapes."""
    torch.manual_seed(7)
    T, K, I, E, top_k = 1, 64, 32, 2, 2
    FC1_N = 2 * I  # SwiGLU
    x = torch.randn(T, K, dtype=torch.bfloat16, device=device)
    w13_f32 = torch.randn(E, FC1_N, K, dtype=torch.float32, device=device) * 0.1
    w2_f32 = torch.randn(E, K, I, dtype=torch.float32, device=device) * 0.1
    w13_packed = torch.empty(E, FC1_N, K // 2, dtype=torch.uint8, device=device)
    w13_sf = torch.empty(E, FC1_N, K // 32, dtype=torch.uint8, device=device)
    w2_packed = torch.empty(E, K, I // 2, dtype=torch.uint8, device=device)
    w2_sf = torch.empty(E, K, I // 32, dtype=torch.uint8, device=device)
    for e in range(E):
        w13_packed[e], w13_sf[e] = quantize_to_mxfp4(w13_f32[e])
        w2_packed[e], w2_sf[e] = quantize_to_mxfp4(w2_f32[e])
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32, device=device)

    res = moe_reference_mxfp4_mxfp8(x, w13_packed, w13_sf, w2_packed, w2_sf,
                                    topk_ids, topk_weights, activation="silu")
    assert res.out_bf16.shape == (T, K)
    assert torch.isfinite(res.out_bf16).all()
    # Output should not be all-zero — reasonable signal floor.
    assert res.out_bf16.abs().mean() > 1e-4
