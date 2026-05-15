"""Triton W4A16 dense GEMM kernel for SM120 / SM121.

Decode-style (M ≤ 32) matmul with bf16 activations, packed FP4 weights,
and per-block FP8 (E4M3) scale factors.  The kernel reads the
**unswizzled** scale layout ``[N, K // 16]`` to keep the Triton indexing
straightforward; ``DenseGemmW4A16MicroKernel`` unswizzles the b12x
storage layout once per call.

This is the v1 production kernel.  A higher-perf CuTe-DSL variant (with
TMA + swizzled-scale direct reads) is tracked for v2.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# FP4 E2M1FN code -> float magnitude / sign.  Low nibble layout:
#   bits[2:0] = magnitude code in {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
#   bit[3]    = sign (1 -> negative).
_FP4_LUT_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


@triton.jit
def _w4a16_dense_decode_kernel(
    x_ptr,            # [M, K] bf16
    w_ptr,            # [N, K // 2] uint8 (FP4 packed two-per-byte)
    w_sf_ptr,         # [N, K // 16] float32 (UNswizzled, already cast)
    w_alpha_ptr,      # scalar float32
    out_ptr,          # [M, N] bf16
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wkh,    # w stride in (N, K//2) layout
    stride_sfn, stride_sfkb,  # w_sf stride in (N, K//16) layout
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    w_alpha = tl.load(w_alpha_ptr).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate K in BLOCK_K steps (BLOCK_K must be a multiple of GROUP_SIZE).
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # --- Load activation x[BLOCK_M, BLOCK_K] (bf16) ---
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )  # bf16

        # --- Load packed weight w[BLOCK_N, BLOCK_K // 2] (uint8) ---
        offs_kh = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        mask_kh = offs_kh < (K // 2)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_kh[None, :] * stride_wkh
        w_packed = tl.load(
            w_ptrs,
            mask=mask_n[:, None] & mask_kh[None, :],
            other=0,
        )  # uint8 [BLOCK_N, BLOCK_K // 2]

        # Unpack nibbles -> two FP4 codes per byte -> two interleaved
        # values per K position.  Convention matches b12x's
        # `pack_grouped_fp4_values`: low nibble is the even-K slot,
        # high nibble is the odd-K slot.
        # FP4 layout: bit[3] = sign, bits[2:0] = magnitude in
        #   {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
        w_lo = (w_packed & 0xF).to(tl.int32)
        w_hi = ((w_packed >> 4) & 0xF).to(tl.int32)

        lo_sign = tl.where((w_lo & 8) != 0, -1.0, 1.0)
        lo_mag_code = w_lo & 7
        lo_mag = tl.where(lo_mag_code == 0, 0.0,
                  tl.where(lo_mag_code == 1, 0.5,
                  tl.where(lo_mag_code == 2, 1.0,
                  tl.where(lo_mag_code == 3, 1.5,
                  tl.where(lo_mag_code == 4, 2.0,
                  tl.where(lo_mag_code == 5, 3.0,
                  tl.where(lo_mag_code == 6, 4.0, 6.0)))))))
        w_lo_f = lo_sign * lo_mag

        hi_sign = tl.where((w_hi & 8) != 0, -1.0, 1.0)
        hi_mag_code = w_hi & 7
        hi_mag = tl.where(hi_mag_code == 0, 0.0,
                  tl.where(hi_mag_code == 1, 0.5,
                  tl.where(hi_mag_code == 2, 1.0,
                  tl.where(hi_mag_code == 3, 1.5,
                  tl.where(hi_mag_code == 4, 2.0,
                  tl.where(hi_mag_code == 5, 3.0,
                  tl.where(hi_mag_code == 6, 4.0, 6.0)))))))
        w_hi_f = hi_sign * hi_mag

        # Interleave: w_decoded[:, 2*i]   = w_lo[:, i]
        #             w_decoded[:, 2*i+1] = w_hi[:, i]
        w_decoded = tl.interleave(w_lo_f, w_hi_f)  # [BLOCK_N, BLOCK_K]

        # --- Load block scales w_sf[BLOCK_N, BLOCK_K // GROUP_SIZE] ---
        offs_kb = (k0 // GROUP_SIZE) + tl.arange(0, BLOCK_K // GROUP_SIZE)
        mask_kb = offs_kb < (K // GROUP_SIZE)
        sf_ptrs = w_sf_ptr + offs_n[:, None] * stride_sfn + offs_kb[None, :] * stride_sfkb
        sf_fp32 = tl.load(
            sf_ptrs,
            mask=mask_n[:, None] & mask_kb[None, :],
            other=0.0,
        )  # [BLOCK_N, BLOCK_K // GROUP_SIZE]

        # Broadcast SF over GROUP_SIZE within each block.
        sf_expanded = tl.broadcast_to(
            sf_fp32[:, :, None],
            (BLOCK_N, BLOCK_K // GROUP_SIZE, GROUP_SIZE),
        ).reshape(BLOCK_N, BLOCK_K)

        # Final dequantized weight in fp32, then cast to bf16 for MMA.
        w_bf16 = (w_decoded * sf_expanded * w_alpha).to(tl.bfloat16)

        # MMA accumulation: out += x @ w.T -> shape [BLOCK_M, BLOCK_N].
        acc += tl.dot(x_tile, w_bf16.T)

    # Store output.
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _next_pow2_at_least_16(x: int) -> int:
    if x <= 16:
        return 16
    p = 1
    while p < x:
        p <<= 1
    return p


def _pick_block_n(n: int) -> int:
    # Honour FP4 group of 16; prefer 128 unless N is tiny.
    if n <= 16:
        return 16
    if n <= 64:
        return 64
    if n <= 128:
        return 128
    return 128


def w4a16_dense_decode_triton(
    x: torch.Tensor,             # [M, K] bf16
    w_fp4: torch.Tensor,         # [N, K // 2] uint8
    w_sf_fp32: torch.Tensor,     # [N, K // 16] float32 (UNswizzled, fp32-cast)
    w_alpha: torch.Tensor,       # scalar float32
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the Triton W4A16 dense decode kernel."""
    assert x.dtype == torch.bfloat16, f"x dtype={x.dtype}"
    assert w_fp4.dtype == torch.uint8, f"w_fp4 dtype={w_fp4.dtype}"
    assert w_sf_fp32.dtype == torch.float32, f"w_sf dtype={w_sf_fp32.dtype}"
    assert w_alpha.dtype == torch.float32

    m, k = x.shape
    n = w_fp4.shape[0]
    assert w_fp4.shape[1] == k // 2
    assert w_sf_fp32.shape == (n, k // 16)

    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    assert out.shape == (m, n)

    BLOCK_M = _next_pow2_at_least_16(m)  # Triton needs power-of-2; pad m.
    BLOCK_N = _pick_block_n(n)
    BLOCK_K = 128

    grid = (
        triton.cdiv(m, BLOCK_M),
        triton.cdiv(n, BLOCK_N),
    )

    _w4a16_dense_decode_kernel[grid](
        x, w_fp4,
        w_sf_fp32,
        w_alpha,
        out,
        m, n, k,
        x.stride(0), x.stride(1),
        w_fp4.stride(0), w_fp4.stride(1),
        w_sf_fp32.stride(0), w_sf_fp32.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE=16,
    )
    return out


__all__ = ["w4a16_dense_decode_triton"]
