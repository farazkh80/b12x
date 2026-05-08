"""Validate the on-device SwiGLU + MXFP8 quant kernel against the torch reference."""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused.mxfp4_mxfp8._quant_kernel import silu_mxfp8_quantize
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine,
    dequant_mxfp8,
    quantize_to_mxfp8,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.mark.parametrize("activation", ["silu", "relu2"])
@pytest.mark.parametrize("T,FC1_N", [
    (1, 64),     # smallest
    (4, 64),
    (16, 64),
    (16, 128),
    (32, 128),
])
def test_silu_mxfp8_quant_matches_reference(device, activation, T, FC1_N):
    torch.manual_seed(2)
    in_f32 = torch.randn(T, FC1_N, dtype=torch.float32, device=device)

    # CPU/torch reference path
    if activation == "silu":
        gate = in_f32[:, :FC1_N // 2]
        up = in_f32[:, FC1_N // 2:]
        intermediate = torch.nn.functional.silu(up) * gate
    else:
        intermediate = torch.nn.functional.relu(in_f32).square()
    ref_e4m3, ref_sf = quantize_to_mxfp8(intermediate)

    # Kernel path
    actual_e4m3, actual_sf = silu_mxfp8_quantize(in_f32, activation=activation)

    # Compare via dequant since direct E4M3 byte equality is too brittle.
    I = intermediate.shape[1]
    ref_dq = dequant_mxfp8(ref_e4m3, ref_sf, rows=T, cols=I)
    actual_dq = dequant_mxfp8(actual_e4m3, actual_sf, rows=T, cols=I)

    cos = cosine(actual_dq, ref_dq)
    err = (actual_dq - ref_dq).abs()
    print(f"\n[T={T} FC1_N={FC1_N} act={activation}] cos={cos:.6f} max_abs={err.max().item():.4f} rmse={err.square().mean().sqrt().item():.4f}")
    assert cos > 0.999, f"on-device quant cosine {cos:.6f} below threshold"
