#!/usr/bin/env python3
"""Bench harness for MXFP8 act × MXFP4 weight grouped GEMM (gpt-oss shapes).

Currently runs:
  - torch.matmul on dequantized weights (correctness anchor)
  - b12x kernel (placeholder — not yet implemented; falls through with msg)
  - flashinfer.group_gemm_mxfp8_mxfp4_nt_groupwise IF the SM120 path is wired

Notes:
  * The prebuilt flashinfer wheel hard-codes `get_gemm_sm100_module()` for the
    mxfp4 entry. On SM120 hardware that crashes inside CUTLASS' init. Until the
    SM120 dispatch path is added, the flashinfer column reports SKIPPED.
  * We use flashinfer's host quantizers (`mxfp4_quantize`, `mxfp8_quantize`) so
    that the on-disk format matches whatever flashinfer's grouped GEMM expects
    when the SM120 binding lands.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine,
    dequant_mxfp4,
    dequant_mxfp8,
    quantize_to_mxfp4,
    quantize_to_mxfp8,
)


def _make_l2_flush(device: torch.device, bytes_hint: int = 0) -> callable:
    if bytes_hint <= 0:
        # 2x typical Blackwell L2 (~70 MiB).
        bytes_hint = 256 * 1024 * 1024
    buf = torch.empty(bytes_hint // 4, dtype=torch.int32, device=device)

    def flush():
        buf.zero_()
    return flush


def _bench(fn, *, warmup: int, iters: int, l2_flush) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        l2_flush()
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt) * 1000.0)  # μs
    times.sort()
    return times[len(times) // 2]


def torch_dequant_matmul(a_e4m3: torch.Tensor, a_sf: torch.Tensor,
                         b_packed: torch.Tensor, b_sf: torch.Tensor,
                         m_indptr: torch.Tensor) -> torch.Tensor:
    """Per-group dequant + matmul. Slow but exact; correctness anchor."""
    cum_m, K = a_e4m3.shape
    E, N, K_half = b_packed.shape
    assert K_half * 2 == K
    out = torch.empty(cum_m, N, dtype=torch.bfloat16, device=a_e4m3.device)
    for e in range(E):
        m_lo = int(m_indptr[e].item())
        m_hi = int(m_indptr[e + 1].item())
        if m_hi == m_lo:
            continue
        rows = m_hi - m_lo
        a_dq = dequant_mxfp8(a_e4m3[m_lo:m_hi].view(torch.uint8), a_sf[m_lo:m_hi],
                             rows=rows, cols=K)
        w_dq = dequant_mxfp4(b_packed[e], b_sf[e], rows=N, cols=K)
        out[m_lo:m_hi] = (a_dq @ w_dq.T).to(torch.bfloat16)
    return out


def flashinfer_grouped(a, b_grp, a_sf, b_sf_grp, m_indptr, *,
                       tile_m=128, tile_n=128, tile_k=128) -> Optional[torch.Tensor]:
    """Try flashinfer's grouped GEMM. Returns None if SM120 path unavailable."""
    try:
        from flashinfer.gemm import group_gemm_mxfp8_mxfp4_nt_groupwise
    except ImportError:
        return None
    try:
        return group_gemm_mxfp8_mxfp4_nt_groupwise(
            a, b_grp, a_sf, b_sf_grp, m_indptr,
            mma_sm=1, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, swap_ab=True,
        )
    except RuntimeError as e:
        if "SM100" in str(e) or "Internal" in str(e):
            return None
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--m-per-expert", type=int, default=128,
                   help="rows per group (multiple of 4)")
    p.add_argument("--n", type=int, default=2880)
    p.add_argument("--k", type=int, default=2880)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")
    torch.manual_seed(0)
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")

    E, M_per, N, K = args.num_experts, args.m_per_expert, args.n, args.k
    if K % 32 != 0:
        raise ValueError("K must be divisible by 32 (MX block size)")
    cum_m = E * M_per

    # synth activations + weights → quantize to MX
    x_bf = torch.randn(cum_m, K, dtype=torch.bfloat16, device=device)
    w_bf = torch.randn(E, N, K, dtype=torch.bfloat16, device=device)
    a_e4m3, a_sf = quantize_to_mxfp8(x_bf.to(torch.float32))
    b_packed = torch.empty(E, N, K // 2, dtype=torch.uint8, device=device)
    b_sf = torch.empty(E, N, K // 32, dtype=torch.uint8, device=device)
    for e in range(E):
        b_packed[e], b_sf[e] = quantize_to_mxfp4(w_bf[e].to(torch.float32))
    m_indptr = torch.tensor([i * M_per for i in range(E + 1)], dtype=torch.int32, device=device)

    print(f"shapes: cum_m={cum_m}, N={N}, K={K}, E={E}")

    flush = _make_l2_flush(device)

    print("\n[1/3] torch dequant matmul (correctness anchor)")
    a_e4m3_view = a_e4m3.view(torch.float8_e4m3fn)
    out_torch = torch_dequant_matmul(a_e4m3_view, a_sf, b_packed, b_sf, m_indptr)
    t_torch = _bench(
        lambda: torch_dequant_matmul(a_e4m3_view, a_sf, b_packed, b_sf, m_indptr),
        warmup=2, iters=3, l2_flush=flush,
    )
    print(f"  median: {t_torch / 1000:.2f} ms")

    print("\n[2/3] flashinfer grouped (SM120 mxfp4 path)")
    out_fi = flashinfer_grouped(a_e4m3_view, b_packed, a_sf, b_sf, m_indptr)
    if out_fi is None:
        print("  SKIPPED — flashinfer wheel does not expose SM120 mxfp4 binding.")
        t_fi = None
    else:
        cos = cosine(out_torch, out_fi)
        print(f"  cosine vs torch: {cos:.6f}")
        t_fi = _bench(
            lambda: flashinfer_grouped(a_e4m3_view, b_packed, a_sf, b_sf, m_indptr),
            warmup=args.warmup, iters=args.iters, l2_flush=flush,
        )
        print(f"  median: {t_fi:.2f} μs")

    print("\n[3/3] b12x mxfp4×mxfp8 grouped (under construction)")
    print("  PENDING — requires Phase 4 fused MoE kernel.")
    t_b12x = None

    if args.out_json:
        args.out_json.write_text(json.dumps({
            "shapes": {"cum_m": cum_m, "N": N, "K": K, "E": E, "M_per_expert": M_per},
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": f"sm_{cap[0]}{cap[1]}",
            "torch_us": t_torch,
            "flashinfer_us": t_fi,
            "b12x_us": t_b12x,
        }, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
