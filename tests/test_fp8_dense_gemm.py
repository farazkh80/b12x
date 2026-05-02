from __future__ import annotations

import pytest
import torch

from b12x.gemm.fp8_dense import fp8_dense_gemm, fp8_dense_gemm_mma
from b12x.gemm.fp8_dense_cuda import fp8_dense_gemm_cuda
from .helpers import require_sm120


def _make_fp8(shape: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) / 8
    amax = src.abs().max().clamp_min(1e-6).to(torch.float32)
    q_scale = torch.finfo(torch.float8_e4m3fn).max / amax
    return (src * q_scale).to(torch.float8_e4m3fn).contiguous(), q_scale.reciprocal().reshape(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 64, 64), (8, 128, 128), (32, 256, 256)])
def test_fp8_dense_gemm_matches_scaled_mm(shape: tuple[int, int, int]) -> None:
    require_sm120()
    m, k, n = shape
    a, scale_a = _make_fp8((m, k), 1)
    b, scale_b = _make_fp8((n, k), 2)

    out = fp8_dense_gemm(a, b, scale_a, scale_b)
    ref = torch._scaled_mm(
        a,
        b.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=0.25, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_dense_gemm_mma_matches_scaled_mm_tile() -> None:
    require_sm120()
    m, k, n = 16, 32, 32
    a, scale_a = _make_fp8((m, k), 3)
    b, scale_b = _make_fp8((n, k), 4)

    ref = torch._scaled_mm(
        a,
        b.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
    )
    out = fp8_dense_gemm_mma(a, b, scale_a, scale_b)
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=0.5, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_dense_gemm_cuda_matches_scaled_mm_tile() -> None:
    require_sm120()
    m, k, n = 16, 32, 32
    a, scale_a = _make_fp8((m, k), 5)
    b, scale_b = _make_fp8((n, k), 6)

    ref = torch._scaled_mm(
        a,
        b.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
    )
    out = fp8_dense_gemm_cuda(a, b, scale_a, scale_b)
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=0.5, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_dense_gemm_cuda_restores_column_six_in_each_tile() -> None:
    require_sm120()
    scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    a = torch.zeros((16, 32), device="cuda", dtype=torch.float32)
    b = torch.zeros((32, 32), device="cuda", dtype=torch.float32)
    a[:, 0] = 1.0
    b[:, 0] = torch.arange(1, 33, device="cuda", dtype=torch.float32)

    out = fp8_dense_gemm_cuda(
        a.to(torch.float8_e4m3fn).contiguous(),
        b.to(torch.float8_e4m3fn).contiguous(),
        scale,
        scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), torch._scaled_mm(
        a.to(torch.float8_e4m3fn).contiguous(),
        b.to(torch.float8_e4m3fn).contiguous().t(),
        scale_a=scale,
        scale_b=scale,
        out_dtype=torch.bfloat16,
    ).float(), atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_dense_gemm_mma_restores_column_six_in_each_tile() -> None:
    require_sm120()
    scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    a = torch.zeros((16, 32), device="cuda", dtype=torch.float32)
    b = torch.zeros((32, 32), device="cuda", dtype=torch.float32)
    a[:, 0] = 1.0
    b[:, 0] = torch.arange(1, 33, device="cuda", dtype=torch.float32)

    out = fp8_dense_gemm_mma(
        a.to(torch.float8_e4m3fn).contiguous(),
        b.to(torch.float8_e4m3fn).contiguous(),
        scale,
        scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), torch._scaled_mm(
        a.to(torch.float8_e4m3fn).contiguous(),
        b.to(torch.float8_e4m3fn).contiguous().t(),
        scale_a=scale,
        scale_b=scale,
        out_dtype=torch.bfloat16,
    ).float(), atol=0.0, rtol=0.0)
