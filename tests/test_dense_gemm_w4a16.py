"""Accuracy tests for b12x W4A16 dense GEMM."""

from __future__ import annotations

import pytest
import torch

from b12x.gemm.w4a16 import (
    DenseGemmW4A16MicroKernel,
    dense_gemm_w4a16,
    dense_reference_w4a16,
    quantize_dense_weight_to_fp4,
)
from b12x.moe.fused.w4a16.reference import compare_to_reference


def test_dense_reference_w4a16_signature():
    """Reference function exists, returns ``[M, N]`` bf16, both code paths."""
    device = torch.device("cpu")
    m, k, n = 2, 32, 16
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)

    out_float = dense_reference_w4a16(x, w_bf16=w_bf16)
    assert out_float.shape == (m, n)
    assert out_float.dtype == torch.bfloat16

    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w_bf16)
    out_quant = dense_reference_w4a16(
        x, w_fp4=w_fp4, w_blockscale=w_bs, w_alpha=w_alpha
    )
    assert out_quant.shape == (m, n)
    assert out_quant.dtype == torch.bfloat16


@pytest.mark.parametrize("n,k", [(16, 32), (16, 64), (32, 128), (64, 512)])
def test_dense_reference_dequant_roundtrip(n, k):
    """``dense_reference_w4a16(quantize(w))`` matches ``x @ w.T`` to FP4 precision."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    m = 4
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)

    out_quant = dense_reference_w4a16(
        x, w_fp4=w_fp4, w_blockscale=w_bs, w_alpha=w_alpha
    )
    out_float = dense_reference_w4a16(x, w_bf16=w)

    metrics = compare_to_reference(out_quant, out_float)
    # FP4 weight quant noise accumulates as ~sqrt(K) in absolute terms;
    # we only assert high cosine correlation here.  The strict
    # quant-vs-quant accuracy gate lives in the kernel tests.
    assert metrics.cos > 0.99, f"cos={metrics.cos}"


def test_dense_gemm_w4a16_signature_callable():
    """Public entry exists with the expected signature."""
    import inspect
    sig = inspect.signature(dense_gemm_w4a16)
    params = list(sig.parameters)
    assert params[:4] == ["x", "w_fp4", "w_blockscale", "w_alpha"]
    assert "out" in sig.parameters


def test_dense_gemm_w4a16_is_supported_matrix():
    """is_supported gate matches the documented decode envelope."""
    # M ladder
    for m in (1, 2, 4, 8, 10, 12, 16, 24, 32):
        assert DenseGemmW4A16MicroKernel.is_supported(m, 2688, 4096)
    for m in (0, 3, 5, 17, 33, 64):
        assert not DenseGemmW4A16MicroKernel.is_supported(m, 2688, 4096)
    # K must be a multiple of 128.  All Nano35 dense K values qualify.
    for k in (2688, 3712, 4096):
        assert DenseGemmW4A16MicroKernel.is_supported(1, k, 256)
    assert not DenseGemmW4A16MicroKernel.is_supported(1, 511, 16)
    assert not DenseGemmW4A16MicroKernel.is_supported(1, 256 + 64, 16)
    # N must be a multiple of 16.
    assert DenseGemmW4A16MicroKernel.is_supported(1, 512, 16)
    assert not DenseGemmW4A16MicroKernel.is_supported(1, 512, 15)


def test_dense_gemm_w4a16_stub_matches_reference_cpu():
    """Stub kernel call returns the reference output (CPU path)."""
    device = torch.device("cpu")
    torch.manual_seed(42)
    m, k, n = 1, 512, 16
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
    out = dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha)
    ref = dense_reference_w4a16(x, w_fp4=w_fp4, w_blockscale=w_bs, w_alpha=w_alpha)
    assert torch.equal(out, ref)


def test_dense_quantize_shapes():
    """Packer produces the expected shapes / dtypes."""
    device = torch.device("cpu")
    n, k = 64, 512
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
    assert w_fp4.shape == (n, k // 2)
    assert w_fp4.dtype == torch.uint8
    # rows_padded = ceil(64/128)*128 = 128; cols_padded = ceil(32/4)*4 = 32
    assert w_bs.shape == (128, 32)
    assert w_bs.dtype == torch.float8_e4m3fn
    assert w_alpha.dtype == torch.float32
    assert w_alpha.ndim == 0
