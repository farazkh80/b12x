"""Routed multi-expert MXFP4×MXFP8 MoE on the b12x single-expert kernel chain.

This is the second Phase-4 layer: we wrap `expert_chain_b12x` with an
expert-routing loop so the same building block handles a real MoE layer.

Strategy:
  - Group tokens by expert (per top-k slot) — for each unique expert that any
    token routes to, gather its tokens, run one expert_chain_b12x call,
    scatter-accumulate the weighted result back into the output.
  - This is the same structure that real fused-MoE kernels use; in production
    you'd fuse the gather/scatter into the kernel via a token-map indirection
    in the producer warps. We do it in Python here.

Limitations:
  - Single-CTA expert chain (M ≤ 16 per expert call).
  - Two GEMM launches per expert (no SMEM intermediate fusion).
  - Python-side gather/scatter and quantize. Production would do this on-device.
"""

from __future__ import annotations

from typing import Optional

import torch

from b12x.moe.fused.mxfp4_mxfp8.expert_chain import expert_chain_b12x
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    quantize_to_mxfp8,
)


def routed_moe_b12x(
    x_bf16: torch.Tensor,            # (T, K_in) bf16
    w13_packed: torch.Tensor,        # (E, FC1_N, K_in/2)
    w13_sf: torch.Tensor,            # (E, FC1_N, K_in/32)
    w2_packed: torch.Tensor,         # (E, K_out, I/2)
    w2_sf: torch.Tensor,             # (E, K_out, I/32)
    topk_ids: torch.Tensor,          # (T, top_k) int32
    topk_weights: torch.Tensor,      # (T, top_k) float32
    *,
    activation: str = "silu",
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-k routed MoE forward. Output shape (T, K_out) bf16."""
    T, K_in = x_bf16.shape
    E, FC1_N, _ = w13_packed.shape
    K_out = w2_packed.shape[1]
    top_k = topk_ids.shape[1]
    device = x_bf16.device

    if output is None:
        output = torch.zeros(T, K_out, dtype=torch.bfloat16, device=device)
    else:
        output.zero_()

    # Quantize tokens once.
    x_e4m3, x_sf = quantize_to_mxfp8(x_bf16.to(torch.float32))
    # x_e4m3: (T, K_in) E4M3 byte; x_sf: (T, K_in/32)

    # Pre-scan which (token, slot) pairs route to each expert.
    # For each unique expert id 'e' actually selected, gather its tokens.
    flat_ids = topk_ids.reshape(-1).cpu()        # (T*top_k,)
    flat_weights = topk_weights.reshape(-1).cpu()

    # Group by expert
    expert_to_pairs: dict[int, list[tuple[int, int]]] = {}
    for idx in range(T * top_k):
        e = int(flat_ids[idx].item())
        token_idx = idx // top_k
        expert_to_pairs.setdefault(e, []).append((token_idx, idx))

    out_f32 = torch.zeros(T, K_out, dtype=torch.float32, device=device)

    for e, pairs in expert_to_pairs.items():
        if e < 0 or e >= E:
            continue
        # Gather token rows that route to this expert.
        token_indices = [p[0] for p in pairs]
        flat_indices = [p[1] for p in pairs]
        n_tokens_for_e = len(token_indices)
        # The lite kernel takes M ≤ 16 at a time; chunk if more.
        for chunk_start in range(0, n_tokens_for_e, 16):
            chunk = pairs[chunk_start:chunk_start + 16]
            t_chunk = torch.tensor([p[0] for p in chunk], device=device, dtype=torch.long)
            f_chunk = torch.tensor([p[1] for p in chunk], device=device, dtype=torch.long)

            sub_x_e4m3 = x_e4m3[t_chunk]
            sub_x_sf = x_sf[t_chunk]

            res = expert_chain_b12x(
                sub_x_e4m3, sub_x_sf,
                w13_packed[e], w13_sf[e],
                w2_packed[e], w2_sf[e],
                activation=activation,
            )

            # Weight by topk_weight for each (token, slot) and accumulate.
            w_chunk = flat_weights[f_chunk.cpu()].to(device).to(torch.float32).unsqueeze(-1)
            sub_out_f32 = res.out_bf16.to(torch.float32) * w_chunk
            out_f32.index_add_(0, t_chunk, sub_out_f32)

    output.copy_(out_f32.to(torch.bfloat16))
    return output
