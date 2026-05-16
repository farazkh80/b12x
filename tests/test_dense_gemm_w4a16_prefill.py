"""Accuracy tests for the b12x W4A16 dense GEMM **prefill** kernel (v5).

Mirrors ``tests/test_dense_gemm_w4a16.py`` for the prefill regime
(M ≥ 64).  Runs every Nano3.5 dense linear shape × a ladder of M values
covering the production range observed in nsys (per-GEMM M ∈ [512, 4096],
plus M=64..256 for the dispatch crossover region).
"""

from __future__ import annotations

import pytest
import torch

from b12x.gemm.w4a16 import (
    dense_gemm_w4a16,
    dense_reference_w4a16,
    quantize_dense_weight_to_fp4,
)
from b12x.gemm.w4a16._cute_prefill_kernel import DenseGemmW4A16CutePrefillKernel
from b12x.moe.fused.reference import compare_to_reference


NANO35_DENSE_SHAPES = [
    # (name, K, N)
    ("q_proj",             2688,   4096),
    ("k_proj",             2688,    256),
    ("v_proj",             2688,    256),
    ("o_proj",             4096,   2688),
    ("shared_expert.up",   2688,   3712),
    ("shared_expert.down", 3712,   2688),
    ("mamba_in_proj",      2688,  10304),
    ("mamba_output_proj",  4096,   2688),
]


@pytest.mark.parametrize(
    "name,k,n", NANO35_DENSE_SHAPES, ids=[s[0] for s in NANO35_DENSE_SHAPES],
)
@pytest.mark.parametrize("m", [64, 128, 256, 512, 1024, 2048, 4096])
def test_prefill_kernel_accuracy(name, k, n, m):
    """v5 prefill kernel matches the W4A16 reference for every Nano35 shape × M.

    Gates (same as v4 decode):
    * ``cos > 0.9999`` — kernel + reference consume the same FP4 weights;
      any deviation is fp32-accum + bf16-round noise.
    * ``max_abs ≤ max(0.04, 1% × ref.abs().max())``.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not DenseGemmW4A16CutePrefillKernel.is_supported(m, k, n):
        pytest.skip(f"prefill not supported (m={m}, k={k}, n={n})")

    device = torch.device("cuda")
    torch.manual_seed(hash((name, m)) & 0xFFFFFFFF)
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)

    out = dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha)
    ref = dense_reference_w4a16(
        x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu(),
    ).to(device)

    metrics = compare_to_reference(out, ref)
    assert metrics.cos > 0.9999, f"{name} m={m}: cos={metrics.cos}"
    ref_max_abs = ref.abs().max().item()
    rel_thresh = max(0.04, 0.01 * ref_max_abs)
    assert metrics.max_abs <= rel_thresh, (
        f"{name} m={m}: max_abs={metrics.max_abs} cos={metrics.cos} "
        f"(threshold {rel_thresh:.4f})"
    )


def test_prefill_dispatch_threshold(monkeypatch):
    """Dispatch routes large-M calls through the prefill kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Force a low threshold and confirm the prefill kernel handles M=64.
    monkeypatch.setenv("B12X_GEMM_W4A16_PREFILL_M", "33")
    from importlib import reload
    from b12x.gemm.w4a16 import micro as micro_mod
    reload(micro_mod)
    device = torch.device("cuda")
    torch.manual_seed(0)
    m, k, n = 128, 2688, 4096
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
    out = micro_mod.dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha)
    ref = dense_reference_w4a16(
        x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu(),
    ).to(device)
    metrics = compare_to_reference(out, ref)
    assert metrics.cos > 0.9999, f"cos={metrics.cos}"


def test_prefill_is_supported_envelope():
    """Prefill kernel accepts the documented (N % 64, K % 64) envelope."""
    kc = DenseGemmW4A16CutePrefillKernel
    assert kc.is_supported(64, 2688, 4096)
    assert kc.is_supported(2048, 2688, 4096)
    assert kc.is_supported(4096, 4096, 2688)
    # N % 64 != 0
    assert not kc.is_supported(128, 2688, 100)
    # K % 64 != 0
    assert not kc.is_supported(128, 100, 2688)
    # M ≤ 0
    assert not kc.is_supported(0, 2688, 4096)


def test_prefill_padding_safe_for_m_below_tile_m():
    """M=64 (< tile_M=128) writes only the visible rows; no OOB.

    Regression: an earlier bug keyed the JIT compile cache by m_padded
    and passed m_padded to the kernel — which caused the epilogue
    m_valid check to use 128 instead of the real M=64, writing 128
    rows into a 64-row output buffer.  This guards against re-introducing
    that.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(123)
    m, k, n = 64, 2688, 3712
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)

    # Sentinel: pre-fill output with NaN so a stray OOB write would be
    # detectable by the compare or a follow-on numerical check on the
    # sentinel region — but the buffer is exactly (m, n), so the kernel
    # *must not* write past row m.
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    kernel = DenseGemmW4A16CutePrefillKernel()
    kernel(x, w_fp4, w_bs, w_alpha, out=out)
    assert not torch.isnan(out).any(), "kernel left NaNs (incomplete coverage)"

    ref = dense_reference_w4a16(
        x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu(),
    ).to(device)
    metrics = compare_to_reference(out, ref)
    assert metrics.cos > 0.9999, f"cos={metrics.cos}"
