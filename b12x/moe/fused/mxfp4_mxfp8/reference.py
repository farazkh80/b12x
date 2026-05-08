"""Pure-torch reference for MXFP4 weight × MXFP8 activation MoE.

This module gives correctness anchors at three levels:
  - block-32 quant + dequant for MXFP8 (E4M3 + UE8M0) and MXFP4 (E2M1 + UE8M0)
  - dense GEMM = dequant(A) @ dequant(B).T in fp32, returned as bf16
  - routed MoE forward = per-token, per-expert FP32 matmul + SwiGLU/relu2

Shapes follow flashinfer's `group_gemm_mxfp8_mxfp4_nt_groupwise` convention so
that the same tensors can feed both flashinfer (perf baseline) and our kernel
(unit under test). Specifically:
  - A (activations): row-major E4M3, shape (cum_m, K)
  - B (weights):     col-major packed E2M1, shape (E, N, K // 2) uint8 nibbles
  - SFA: col-major UE8M0, shape (cum_m_padded, K // 32) uint8
  - SFB: row-major UE8M0, shape (E, N_padded, K // 32) uint8

The padded dims come from CUTLASS's swizzled scale-factor layout (128-row
groups, 4-col groups). For the pure-torch reference we work in unswizzled
shapes and only care about the swizzle when bridging to flashinfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


_FP4_LUT_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
_UE8M0_BIAS = 127  # 0x7F = 2^0 = 1.0


def _fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(_FP4_LUT_VALUES, dtype=torch.float32, device=device)


def dequant_mxfp4(
    packed_u8: torch.Tensor,
    sf_ue8m0: torch.Tensor,
    *,
    rows: int,
    cols: int,
    block_size: int = 32,
) -> torch.Tensor:
    """Decode packed E2M1 nibbles + UE8M0 per-block-32 scales → fp32.

    Args:
        packed_u8: (rows, cols // 2) uint8, low nibble holds even-index value.
        sf_ue8m0:  (rows, cols // block_size) uint8, raw UE8M0 exponent byte.
    """
    if packed_u8.shape != (rows, cols // 2):
        raise ValueError(f"packed_u8 shape {packed_u8.shape} != ({rows}, {cols // 2})")
    if sf_ue8m0.shape != (rows, cols // block_size):
        raise ValueError(f"sf shape {sf_ue8m0.shape} != ({rows}, {cols // block_size})")
    lut = _fp4_lut(packed_u8.device)
    lo = (packed_u8 & 0x0F).to(torch.int64)
    hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
    raw = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(rows, cols)
    sf_f32 = _ue8m0_to_f32(sf_ue8m0)
    n_blocks = cols // block_size
    sf_expanded = sf_f32.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)
    return raw * sf_expanded


def dequant_mxfp8(
    e4m3_byte: torch.Tensor,
    sf_ue8m0: torch.Tensor,
    *,
    rows: int,
    cols: int,
    block_size: int = 32,
) -> torch.Tensor:
    """Decode E4M3 bytes + UE8M0 per-block-32 scales → fp32."""
    if e4m3_byte.shape != (rows, cols):
        raise ValueError(f"e4m3 shape {e4m3_byte.shape} != ({rows}, {cols})")
    if sf_ue8m0.shape != (rows, cols // block_size):
        raise ValueError(f"sf shape {sf_ue8m0.shape} != ({rows}, {cols // block_size})")
    raw = e4m3_byte.view(torch.float8_e4m3fn).to(torch.float32)
    sf_f32 = _ue8m0_to_f32(sf_ue8m0)
    n_blocks = cols // block_size
    sf_expanded = sf_f32.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)
    return raw * sf_expanded


def _ue8m0_to_f32(sf: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte → fp32 power-of-2. 0xFF (NaN encoding) → NaN."""
    sf_i = sf.to(torch.int32)
    is_nan = sf_i == 0xFF
    exp = sf_i - _UE8M0_BIAS
    out = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=sf.device), exp.to(torch.float32))
    out = torch.where(is_nan, torch.full_like(out, float("nan")), out)
    return out


def _f32_to_ue8m0(scale_f32: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-2 → UE8M0 byte. Round exponent to nearest, clamp to [0, 254]."""
    safe = torch.where(scale_f32 > 0, scale_f32, torch.full_like(scale_f32, 1e-38))
    exp = torch.log2(safe).round()
    enc = (exp + _UE8M0_BIAS).clamp(0, 254).to(torch.int32)
    return enc.to(torch.uint8)


def quantize_to_mxfp8(
    x_f32: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-block-32 amax → UE8M0 scale, quantize to E4M3 (saturate).

    Returns (e4m3_byte, sf_ue8m0). The encoding follows MX spec: scale is
    chosen so that amax / scale ≤ E4M3_max, i.e. scale = ceil_pow2(amax / max).
    """
    rows, cols = x_f32.shape
    if cols % block_size != 0:
        raise ValueError(f"cols={cols} not divisible by block_size={block_size}")
    n_blocks = cols // block_size
    blocks = x_f32.reshape(rows, n_blocks, block_size)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-38)
    # ceil(log2(amax / E4M3_MAX)): scale ≥ amax/max ensures saturation-safe.
    log_scale = torch.log2(amax / _E4M3_MAX).ceil()
    scale_byte = (log_scale + _UE8M0_BIAS).clamp(0, 254).to(torch.uint8)
    scale_f32 = _ue8m0_to_f32(scale_byte)
    quantized = blocks / scale_f32.unsqueeze(-1)
    quantized = quantized.clamp(-_E4M3_MAX, _E4M3_MAX)
    e4m3 = quantized.to(torch.float8_e4m3fn).reshape(rows, cols)
    return e4m3.view(torch.uint8), scale_byte


def quantize_to_mxfp4(
    x_f32: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-block-32 amax → UE8M0 scale, RNE quantize to E2M1, pack 2/byte.

    Returns (packed_u8, sf_ue8m0). packed_u8 shape (rows, cols // 2): low
    nibble holds even-index, high nibble holds odd-index.
    """
    rows, cols = x_f32.shape
    if cols % block_size != 0:
        raise ValueError(f"cols={cols} not divisible by block_size={block_size}")
    n_blocks = cols // block_size
    blocks = x_f32.reshape(rows, n_blocks, block_size)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-38)
    log_scale = torch.log2(amax / _E2M1_MAX).ceil()
    scale_byte = (log_scale + _UE8M0_BIAS).clamp(0, 254).to(torch.uint8)
    scale_f32 = _ue8m0_to_f32(scale_byte)
    scaled = blocks / scale_f32.unsqueeze(-1)
    nibbles = _quantize_to_e2m1_nibbles(scaled.reshape(rows, cols))
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale_byte


def _quantize_to_e2m1_nibbles(x: torch.Tensor) -> torch.Tensor:
    """fp32 (already scaled to [-6, 6] range) → E2M1 4-bit nibble.

    E2M1 representable values (per LUT): {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
    """
    sign = (x < 0).to(torch.int32) << 3
    mag = x.abs().clamp(0.0, _E2M1_MAX)
    # midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0  (RNE)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=mag.dtype, device=mag.device,
    )
    bucket = torch.bucketize(mag, boundaries)
    return (sign | bucket.to(torch.int32)) & 0x0F


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """SwiGLU: gate ⊙ silu(up) where input concat is (gate ‖ up) along last dim."""
    half = gate_up.shape[-1] // 2
    gate = gate_up[..., :half]
    up = gate_up[..., half:]
    return F.silu(up) * gate


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


@dataclass(frozen=True)
class MoEMXReferenceResult:
    out_bf16: torch.Tensor              # (num_tokens, K)
    fc1_pre_act_f32: torch.Tensor       # (num_tokens, top_k, FC1_N)  diagnostic
    intermediate_f32: torch.Tensor      # (num_tokens, top_k, I)      diagnostic


def moe_reference_mxfp4_mxfp8(
    x_bf16: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_sf: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str = "silu",
) -> MoEMXReferenceResult:
    """Routed-MoE ground truth in fp32 with MXFP4 weight × MXFP8 activation.

    Per-token, per-expert pipeline:
      1. quantize x → MXFP8, dequant → x_dq (fp32)
      2. fc1_pre = x_dq @ dequant(w13)[expert].T   (shape FC1_N)
      3. apply SwiGLU (silu) or relu² → intermediate (shape I)
      4. quantize intermediate → MXFP8, dequant → int_dq (fp32)
      5. fc2 = int_dq @ dequant(w2)[expert].T      (shape K)
      6. accumulate routed_w * fc2 into output

    Shapes:
      x_bf16:     (T, K)
      w13_packed: (E, FC1_N, K // 2)   uint8
      w13_sf:     (E, FC1_N, K // 32)  uint8 UE8M0
      w2_packed:  (E, K, I // 2)       uint8
      w2_sf:      (E, K, I // 32)      uint8
      topk_ids:   (T, top_k) int32
      topk_weights: (T, top_k) float32
    """
    T, K = x_bf16.shape
    E, FC1_N, K_half = w13_packed.shape
    assert K_half == K // 2, f"w13 K/2 {K_half} mismatch with K {K}"
    if activation == "silu":
        assert FC1_N % 2 == 0, "SwiGLU FC1 must produce even N (gate ‖ up)"
        I = FC1_N // 2
    else:
        I = FC1_N
    _, K_w2, I_half = w2_packed.shape
    assert K_w2 == K and I_half == I // 2

    device = x_bf16.device
    top_k = topk_ids.shape[1]

    out_f32 = torch.zeros(T, K, dtype=torch.float32, device=device)
    fc1_pre_dump = torch.zeros(T, top_k, FC1_N, dtype=torch.float32, device=device)
    int_dump = torch.zeros(T, top_k, I, dtype=torch.float32, device=device)

    # Per-token quantize once and reuse across its top_k experts.
    x_e4m3, x_sf = quantize_to_mxfp8(x_bf16.to(torch.float32))
    x_dq = dequant_mxfp8(x_e4m3, x_sf, rows=T, cols=K)

    # Pre-dequant all experts to fp32 once. This is heavy (~E*FC1_N*K*4 bytes) but the
    # reference is for unit tests; production uses the kernel.
    w13_dq = torch.empty(E, FC1_N, K, dtype=torch.float32, device=device)
    w2_dq = torch.empty(E, K, I, dtype=torch.float32, device=device)
    for e in range(E):
        w13_dq[e] = dequant_mxfp4(w13_packed[e], w13_sf[e], rows=FC1_N, cols=K)
        w2_dq[e] = dequant_mxfp4(w2_packed[e], w2_sf[e], rows=K, cols=I)

    activation_fn = _swiglu if activation == "silu" else _relu2

    for t in range(T):
        x_t = x_dq[t]  # (K,)
        for k_idx in range(top_k):
            e = int(topk_ids[t, k_idx])
            w_e = topk_weights[t, k_idx].to(torch.float32)
            fc1_pre = w13_dq[e] @ x_t   # (FC1_N,)
            fc1_pre_dump[t, k_idx] = fc1_pre
            intermediate = activation_fn(fc1_pre.unsqueeze(0)).squeeze(0)  # (I,)
            # MXFP8 quant on the intermediate (block-32 across I)
            int_e4m3, int_sf = quantize_to_mxfp8(intermediate.unsqueeze(0))
            int_dq = dequant_mxfp8(int_e4m3, int_sf, rows=1, cols=I).squeeze(0)
            int_dump[t, k_idx] = int_dq
            fc2 = w2_dq[e] @ int_dq      # (K,)
            out_f32[t] += w_e * fc2

    return MoEMXReferenceResult(
        out_bf16=out_f32.to(torch.bfloat16),
        fc1_pre_act_f32=fc1_pre_dump,
        intermediate_f32=int_dump,
    )


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
