"""Single-expert FC1 → SwiGLU/relu² → MXFP8 quant → FC2 chain on the b12x kernel.

This is the "lite" Phase-4 demonstrator. It composes the validated single-tile
GEMM (`run_single_tile`) twice with a host-orchestrated activation+requant in
between, and validates against the torch reference.

Limitations vs production:
  - Single CTA, single-warp m16n8 building block (M ≤ 16 token batch).
  - Two kernel launches (FC1 then FC2) instead of fusing intermediate in SMEM.
  - Activation+requant runs on the host between kernels (fp32 → MXFP8) using
    the validated reference quantizer.
  - No expert routing: caller picks one expert.

Once this matches the torch reference float-exactly, the next steps are:
  - Tile M up to ≥ 128 via multi-CTA scheduling.
  - Fuse activation+requant on-device (warp-shuffle amax + smem repack).
  - Wire expert routing + scatter/gather (~ existing static.py glue).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from b12x.moe.fused.mxfp4_mxfp8._kernel import run_single_tile
from b12x.moe.fused.mxfp4_mxfp8._quant_kernel import silu_mxfp8_quantize
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine,
    dequant_mxfp4,
    dequant_mxfp8,
    quantize_to_mxfp4,
    quantize_to_mxfp8,
)


@dataclass(frozen=True)
class ExpertChainResult:
    out_bf16: torch.Tensor       # (T, K_out)
    fc1_pre_act: torch.Tensor    # (T, FC1_N) fp32, diagnostic
    intermediate: torch.Tensor   # (T, I)     fp32, diagnostic


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    half = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    return F.silu(up) * gate


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


def expert_chain_b12x(
    x_e4m3: torch.Tensor,            # (T, K_in)  E4M3 byte
    x_sf:   torch.Tensor,            # (T, K_in/32) UE8M0
    w13_packed: torch.Tensor,        # (FC1_N, K_in/2) packed E2M1
    w13_sf:     torch.Tensor,        # (FC1_N, K_in/32)
    w2_packed:  torch.Tensor,        # (K_out, I/2) packed E2M1
    w2_sf:      torch.Tensor,        # (K_out, I/32)
    *,
    activation: str = "silu",
) -> ExpertChainResult:
    """One-expert FC1+activation+FC2 chain. T must be ≤ 16 (single m16n8 building block)."""
    T, K_in = x_e4m3.shape
    FC1_N, K_half = w13_packed.shape
    assert K_half * 2 == K_in
    K_out, I_half = w2_packed.shape
    I = I_half * 2
    if activation == "silu":
        assert FC1_N == 2 * I, f"SwiGLU expects FC1_N={2*I}, got {FC1_N}"
    else:
        assert FC1_N == I, f"relu2 expects FC1_N={I}, got {FC1_N}"

    if T > 16:
        raise ValueError(f"lite kernel limited to T<=16; got {T}")
    if T < 16:
        # pad to M=16 with zeros; output will be sliced back
        pad = 16 - T
        x_e4m3 = torch.cat([x_e4m3, torch.zeros(pad, K_in, dtype=x_e4m3.dtype, device=x_e4m3.device)], dim=0)
        x_sf = torch.cat([x_sf, torch.zeros(pad, K_in // 32, dtype=x_sf.dtype, device=x_sf.device)], dim=0)

    # FC1: (M=16, K_in) @ (FC1_N, K_in).T → (16, FC1_N)
    fc1 = _gemm_m16(x_e4m3, x_sf, w13_packed, w13_sf, n_dim=FC1_N)  # fp32

    # Fused on-device activation + MXFP8 quant — replaces the host-side step.
    int_e4m3, int_sf = silu_mxfp8_quantize(fc1, activation=activation)

    # Diagnostic intermediate (post-activation, pre-quant). Computed in fp32
    # for the diagnostic dataclass; doesn't affect the kernel data path.
    act_fn = _swiglu if activation == "silu" else _relu2
    intermediate = act_fn(fc1)

    # FC2: (16, I) @ (K_out, I).T → (16, K_out)
    fc2 = _gemm_m16(int_e4m3, int_sf, w2_packed, w2_sf, n_dim=K_out)  # fp32

    out_bf16 = fc2.to(torch.bfloat16)
    if T < 16:
        out_bf16 = out_bf16[:T].clone()
        fc1 = fc1[:T].clone()
        intermediate = intermediate[:T].clone()
    return ExpertChainResult(
        out_bf16=out_bf16,
        fc1_pre_act=fc1,
        intermediate=intermediate,
    )


def _gemm_m16(
    A_e4m3: torch.Tensor,    # (16, K)
    A_sf:   torch.Tensor,    # (16, K/32)
    B_packed: torch.Tensor,  # (N, K/2)
    B_sf:     torch.Tensor,  # (N, K/32)
    *,
    n_dim: int,
) -> torch.Tensor:
    """Loop the single-tile GEMM over N tiles of 8."""
    if n_dim % 8 != 0:
        raise ValueError(f"N must be multiple of 8; got {n_dim}")
    out = torch.empty(16, n_dim, dtype=torch.float32, device=A_e4m3.device)
    for n_off in range(0, n_dim, 8):
        Bp = B_packed[n_off:n_off + 8]
        Bsf = B_sf[n_off:n_off + 8]
        out[:, n_off:n_off + 8] = run_single_tile(A_e4m3, A_sf, Bp, Bsf)
    return out
