"""Reusable single-tile (M=16, N=8) MXFP4×MXFP8 GEMM via cvt+mxfp8 path.

Lifts the validated probe (`probe_mxfp4_mxfp8_dense_scaled.py`) into a small
import-friendly module so Phase 4's higher-level pipelines can reuse it.

Per-thread fragment layout (m16n8k32 row.col, byte-stored):
  A (16×K, row-major): 4 b32/thread.
    a0: row=g  , col=lane*8 + 0..3   (K-block first half, top row group)
    a1: row=g+8, col=lane*8 + 0..3
    a2: row=g  , col=lane*8 + 4..7
    a3: row=g+8, col=lane*8 + 4..7
  B (K×8, col-major) packed FP4: 1 b32/thread.
    bp byte i = packed nibbles for col=g, k_byte = lane*4 + i  (k = lane*8 + 0..7)
  D (16×8): 4 f32/thread.
    d0: (g  , lane*2 + 0)
    d1: (g  , lane*2 + 1)
    d2: (g+8, lane*2 + 0)
    d3: (g+8, lane*2 + 1)
  SFA (1 byte per A row, scale_vec::1X): 1 byte distributed per thread.
    lane==0 → SFA[row=g]
    lane==1 → SFA[row=g+8]
    lane>=2 → unused (set 0x7F)
  SFB (1 byte per B col): 1 byte distributed per thread.
    lane==0 → SFB[col=g]
    lane>=1 → unused (set 0x7F)

These layouts hold for one N=8 tile at a time. Larger N is handled by tiling.
"""

from __future__ import annotations

from typing import Tuple

import torch


def pack_a_fragments(A_e4m3: torch.Tensor) -> torch.Tensor:
    """(M=16, K) E4M3 byte → (32, K_blocks, 4) per-thread u32 fragments."""
    M, K = A_e4m3.shape
    if M != 16:
        raise ValueError(f"single_tile expects M=16, got {M}")
    if K % 32 != 0:
        raise ValueError(f"K must be multiple of 32, got {K}")
    K_blocks = K // 32
    out = torch.zeros(32, K_blocks, 4, dtype=torch.int32, device=A_e4m3.device)
    A_int = A_e4m3.to(torch.int32) & 0xFF
    for t in range(32):
        g = t // 4
        lane = t % 4
        for kb in range(K_blocks):
            kb_off = kb * 32
            for i in range(4):
                out[t, kb, 0] |= A_int[g,     kb_off + lane*8 + i]     << (i*8)
                out[t, kb, 1] |= A_int[g + 8, kb_off + lane*8 + i]     << (i*8)
                out[t, kb, 2] |= A_int[g,     kb_off + lane*8 + 4 + i] << (i*8)
                out[t, kb, 3] |= A_int[g + 8, kb_off + lane*8 + 4 + i] << (i*8)
    return out


def pack_b_fragments(B_packed: torch.Tensor) -> torch.Tensor:
    """(N=8, K/2) packed FP4 → (32, K_blocks, 1) per-thread u32 fragments."""
    N, K_half = B_packed.shape
    if N != 8:
        raise ValueError(f"single_tile expects N=8, got {N}")
    K = K_half * 2
    if K % 32 != 0:
        raise ValueError(f"K must be multiple of 32, got {K}")
    K_blocks = K // 32
    out = torch.zeros(32, K_blocks, 1, dtype=torch.int32, device=B_packed.device)
    B_int = B_packed.to(torch.int32) & 0xFF
    for t in range(32):
        col = t // 4
        lane = t % 4
        for kb in range(K_blocks):
            kb_byte_off = (kb * 32) // 2
            for i in range(4):
                byte_idx = kb_byte_off + lane * 4 + i
                out[t, kb, 0] |= B_int[col, byte_idx] << (i*8)
    return out


def pack_sfa(sfa_bytes: torch.Tensor) -> torch.Tensor:
    """(M=16, K_blocks) UE8M0 → (32, K_blocks) per-thread u32."""
    M, K_blocks = sfa_bytes.shape
    if M != 16:
        raise ValueError(f"expects M=16, got {M}")
    out = torch.full((32, K_blocks), 0x7F, dtype=torch.int32, device=sfa_bytes.device)
    for t in range(32):
        g = t // 4
        lane = t % 4
        if lane == 0:
            row = g
        elif lane == 1:
            row = g + 8
        else:
            continue
        for kb in range(K_blocks):
            out[t, kb] = int(sfa_bytes[row, kb].item()) & 0xFF
    return out


def pack_sfb(sfb_bytes: torch.Tensor) -> torch.Tensor:
    """(N=8, K_blocks) UE8M0 → (32, K_blocks) per-thread u32."""
    N, K_blocks = sfb_bytes.shape
    if N != 8:
        raise ValueError(f"expects N=8, got {N}")
    out = torch.full((32, K_blocks), 0x7F, dtype=torch.int32, device=sfb_bytes.device)
    for t in range(32):
        g = t // 4
        lane = t % 4
        if lane != 0:
            continue
        for kb in range(K_blocks):
            out[t, kb] = int(sfb_bytes[g, kb].item()) & 0xFF
    return out


def unpack_d(D_per_thread: torch.Tensor) -> torch.Tensor:
    """(32, 4) per-thread fp32 → (M=16, N=8) D."""
    D = torch.zeros(16, 8, dtype=torch.float32, device=D_per_thread.device)
    for t in range(32):
        g = t // 4
        lane = t % 4
        D[g,     lane*2 + 0] = D_per_thread[t, 0]
        D[g,     lane*2 + 1] = D_per_thread[t, 1]
        D[g + 8, lane*2 + 0] = D_per_thread[t, 2]
        D[g + 8, lane*2 + 1] = D_per_thread[t, 3]
    return D


def gemm_m16_n_via_tiling(
    A_e4m3: torch.Tensor,
    A_sf: torch.Tensor,
    B_packed: torch.Tensor,
    B_sf: torch.Tensor,
    *,
    n_tile_size: int = 8,
) -> torch.Tensor:
    """Host-side reference loop that calls the single-tile kernel for each N tile.

    Useful for validating shape generalization (M=16 fixed; N = multiple of 8;
    K = multiple of 32).
    """
    from b12x.moe.fused.mxfp4_mxfp8._kernel import run_single_tile  # lazy import
    M, K = A_e4m3.shape
    N, K_half = B_packed.shape
    assert M == 16
    assert N % n_tile_size == 0
    assert K_half * 2 == K
    out = torch.empty(M, N, dtype=torch.float32, device=A_e4m3.device)
    for n_off in range(0, N, n_tile_size):
        Bp_tile = B_packed[n_off:n_off + n_tile_size]
        Bsf_tile = B_sf[n_off:n_off + n_tile_size]
        out_tile = run_single_tile(A_e4m3, A_sf, Bp_tile, Bsf_tile)
        out[:, n_off:n_off + n_tile_size] = out_tile
    return out
