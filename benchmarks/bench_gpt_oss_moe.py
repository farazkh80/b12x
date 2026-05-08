#!/usr/bin/env python3
"""End-to-end fused-MoE block benchmark: b12x vs flashinfer.

Compares two implementations of the per-expert forward path:

  b12x       : run_fused_silu_global / cpasync — one cute-DSL launch that does
               FC1 + SwiGLU + MXFP8 quant + FC2.
  flashinfer : group_gemm_mxfp4_nt_groupwise(FC1) +
               host SwiGLU +
               host MXFP8 quant +
               group_gemm_mxfp4_nt_groupwise(FC2).

flashinfer 0.6.10+ is required for the SM120 mxfp4 dispatch. Earlier wheels
hard-code the SM100 entry and crash inside cutlass.gemm.initialize on SM120.

Shape note
----------
gpt-oss-20b uses K_in = I = K_out = 2880 with top-k = 4 over E = 32 experts.
b12x's current single-warp kernel can't fit that shape's per-thread register
budget (FC1_N_tiles = 720 ⇒ ~12 KB of fc1 accumulator per thread; SM120 caps
≈ 1 KB / thread). Scaling to full gpt-oss-20b needs an N-tile partitioning
loop inside the kernel (do 16 N-tiles at a time, reload the FC1 K-loop, then
move to the next 16). That's a separate next step. This bench reports
constrained shapes that the current kernel handles, and notes the gpt-oss-20b
target alongside.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from b12x.moe.fused.mxfp4_mxfp8._fused_kernel import (
    run_fused_silu_cpasync,
    run_fused_silu_global,
)
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    cosine,
    quantize_to_mxfp4,
    quantize_to_mxfp8,
)


def _l2_flush(device: torch.device, n_bytes: int = 256 * 1024 * 1024) -> callable:
    buf = torch.empty(n_bytes // 4, dtype=torch.int32, device=device)
    def flush():
        buf.zero_()
    return flush


def _bench(fn, *, warmup: int, iters: int, l2_flush: callable) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        l2_flush()
        torch.cuda.synchronize()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))  # μs
    return times[len(times) // 2]


def _flashinfer_expert_chain(
    x_e4m3: torch.Tensor,  x_sf: torch.Tensor,
    w13_p: torch.Tensor,   w13_sf: torch.Tensor,
    w2_p:  torch.Tensor,   w2_sf:  torch.Tensor,
    *, tile_m: int = 128, tile_n: int = 128, tile_k: int = 128,
) -> torch.Tensor:
    """flashinfer two-GEMM chain with host-side SwiGLU + MXFP8 quant."""
    from flashinfer import mxfp8_quantize as fi_mxfp8_quantize
    from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise

    M, K_in = x_e4m3.shape
    FC1_N, K_half = w13_p.shape
    K_out, I_half = w2_p.shape
    I = I_half * 2

    m_indptr = torch.tensor([0, M], dtype=torch.int32, device=x_e4m3.device)

    fc1 = group_gemm_mxfp4_nt_groupwise(
        x_e4m3, w13_p.unsqueeze(0), x_sf, w13_sf.unsqueeze(0), m_indptr,
        mma_sm=1, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, swap_ab=True,
        out_dtype=torch.bfloat16,
    )                                          # (M, FC1_N) bf16
    gate, up = fc1[:, :I], fc1[:, I:]
    inter = (F.silu(up) * gate).to(torch.float16)   # flashinfer mxfp8_quantize wants fp16/bf16
    int_e4m3, int_sf = fi_mxfp8_quantize(inter)

    fc2 = group_gemm_mxfp4_nt_groupwise(
        int_e4m3, w2_p.unsqueeze(0), int_sf, w2_sf.unsqueeze(0), m_indptr,
        mma_sm=1, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, swap_ab=True,
        out_dtype=torch.bfloat16,
    )
    return fc2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="constrained",
                   choices=["constrained", "single", "gpt-oss"],
                   help="constrained: kernel-fitting sweep; "
                        "single: one (M,K,I,K_out) from --M --K-in --I --K-out; "
                        "gpt-oss: report target shape, do NOT run b12x at it")
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--K-in", type=int, default=128)
    p.add_argument("--I", type=int, default=128)
    p.add_argument("--K-out", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
    print(f"PyTorch: {torch.__version__}")
    import flashinfer
    print(f"flashinfer: {flashinfer.__version__}")
    print()

    # gpt-oss-20b reference (informational only).
    print("gpt-oss-20b reference shapes (informational):")
    print("  hidden=K_in=K_out=2880, intermediate=I=2880, FC1_N=5760")
    print("  num_experts=32, top_k=4")
    print("  → b12x current kernel can't fit this in single-warp register budget;")
    print("    scaling needs in-kernel N-tile partitioning (separate work).")
    print()

    if args.shapes == "constrained":
        # All dims that flashinfer can run: K_in, I, K_out must each be a
        # multiple of 128 (group_gemm_mxfp4 kernel constraint).
        sweep = [
            ( 64, 128, 128, 128),
            (128, 128, 128, 128),
            (256, 128, 128, 128),
            (128, 256, 128, 128),
            (128, 128, 256, 128),
            (128, 128, 128, 256),
            (128, 256, 256, 256),
            (256, 256, 256, 256),
        ]
    elif args.shapes == "single":
        sweep = [(args.M, args.K_in, args.I, args.K_out)]
    else:
        # gpt-oss target — flashinfer-only (b12x kernel can't fit this shape today).
        print("gpt-oss target shape — flashinfer baseline only "
              "(b12x's single-warp kernel overflows registers at FC1_N=5760).")
        sweep = [(128, 2880, 2880, 2880)]

    flush = _l2_flush(device)
    results = []
    print(f"{'M':>4} {'K_in':>5} {'I':>5} {'K_out':>5}    "
          f"{'b12x_global':>14} {'b12x_cpasync':>14} {'fi_chain':>14}    {'fi/b12x':>10}")
    for (M, K_in, I, K_out) in sweep:
        FC1_N = 2 * I
        torch.manual_seed(M + K_in)
        x_bf = torch.randn(M, K_in, dtype=torch.bfloat16, device=device)
        w13_bf = torch.randn(FC1_N, K_in, dtype=torch.bfloat16, device=device) * 0.3
        w2_bf  = torch.randn(K_out, I, dtype=torch.bfloat16, device=device) * 0.3

        x_e4m3, x_sf = quantize_to_mxfp8(x_bf.to(torch.float32))
        w13_p, w13_sf = quantize_to_mxfp4(w13_bf.to(torch.float32))
        w2_p,  w2_sf  = quantize_to_mxfp4(w2_bf.to(torch.float32))

        # Skip b12x at over-budget shapes (would JIT-spill register stack).
        FC1_N_tiles = (2 * I) // 8
        register_budget_ok = FC1_N_tiles * 4 < 200   # rough fp32-per-thread cap
        if args.shapes == "gpt-oss" or not register_budget_ok:
            t_global = None
            t_cpasync = None
        else:
            try:
                _ = run_fused_silu_global(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)
                t_global = _bench(
                    lambda: run_fused_silu_global(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf),
                    warmup=args.warmup, iters=args.iters, l2_flush=flush,
                )
            except Exception as e:
                t_global = None
                print(f"    b12x_global error: {e}")
            try:
                _ = run_fused_silu_cpasync(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)
                t_cpasync = _bench(
                    lambda: run_fused_silu_cpasync(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf),
                    warmup=args.warmup, iters=args.iters, l2_flush=flush,
                )
            except Exception as e:
                t_cpasync = None
                print(f"    b12x_cpasync error: {e}")

        # flashinfer expects E4M3 byte (NOT uint8 view); use flashinfer's own quantizer.
        from flashinfer import mxfp4_quantize as fi_mxfp4, mxfp8_quantize as fi_mxfp8
        ax_fi, ax_fi_sf = fi_mxfp8(x_bf)
        w13_fi, w13_fi_sf = fi_mxfp4(w13_bf)
        w2_fi,  w2_fi_sf  = fi_mxfp4(w2_bf)
        try:
            _ = _flashinfer_expert_chain(
                ax_fi, ax_fi_sf, w13_fi, w13_fi_sf, w2_fi, w2_fi_sf,
            )
            t_fi = _bench(
                lambda: _flashinfer_expert_chain(
                    ax_fi, ax_fi_sf, w13_fi, w13_fi_sf, w2_fi, w2_fi_sf,
                ),
                warmup=args.warmup, iters=args.iters, l2_flush=flush,
            )
        except Exception as e:
            t_fi = None
            print(f"    flashinfer error: {e}")

        ratio = (t_fi / t_cpasync) if (t_fi is not None and t_cpasync) else None
        def _fmt(v): return f"{v:>13.2f} μs" if v is not None else f"{'NA':>14}"
        ratio_str = f"{ratio:>9.2f}x" if ratio is not None else f"{'NA':>10}"
        print(f"{M:>4} {K_in:>5} {I:>5} {K_out:>5}    "
              f"{_fmt(t_global)} {_fmt(t_cpasync)} {_fmt(t_fi)}    {ratio_str}")

        results.append({
            "M": M, "K_in": K_in, "I": I, "K_out": K_out,
            "b12x_global_us": t_global,
            "b12x_cpasync_us": t_cpasync,
            "flashinfer_chain_us": t_fi,
            "fi_over_b12x_cpasync": ratio,
        })

    if args.out_json:
        args.out_json.write_text(json.dumps({
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": f"sm_{cap[0]}{cap[1]}",
            "torch": torch.__version__,
            "flashinfer": flashinfer.__version__,
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
