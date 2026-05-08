"""Wall-clock comparison: prepacked-fragment vs global-load fused MoE block.

Both produce bit-exact output; this measures the kernel runtime difference
(host-side fragment packing for the prepacked path is excluded from the timed
window).
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import torch

from b12x.moe.fused.mxfp4_mxfp8._fused_kernel import (
    run_fused_silu_cpasync,
    run_fused_silu_cpasync_mw,
    run_fused_silu_full,
    run_fused_silu_global,
)
from b12x.moe.fused.mxfp4_mxfp8.reference import (
    quantize_to_mxfp4,
    quantize_to_mxfp8,
)


def _bench(fn, *, warmup, iters):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))  # μs
    return times[len(times) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--K-in", type=int, default=128)
    p.add_argument("--I", type=int, default=128)
    p.add_argument("--K-out", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    M, K_in, I, K_out = args.M, args.K_in, args.I, args.K_out
    FC1_N = 2 * I

    torch.manual_seed(13)
    x_f32 = torch.randn(M, K_in, dtype=torch.float32, device=device) * 0.5
    w13_f32 = torch.randn(FC1_N, K_in, dtype=torch.float32, device=device) * 0.3
    w2_f32 = torch.randn(K_out, I, dtype=torch.float32, device=device) * 0.3

    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_f32)
    w2_p, w2_sf = quantize_to_mxfp4(w2_f32)

    # Warm all compile caches.
    _ = run_fused_silu_full(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)
    _ = run_fused_silu_global(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)
    _ = run_fused_silu_cpasync(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)

    t_prepack = _bench(
        lambda: run_fused_silu_full(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf),
        warmup=args.warmup, iters=args.iters,
    )
    t_global = _bench(
        lambda: run_fused_silu_global(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf),
        warmup=args.warmup, iters=args.iters,
    )
    t_cpasync = _bench(
        lambda: run_fused_silu_cpasync(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf),
        warmup=args.warmup, iters=args.iters,
    )

    print(f"shape: M={M}, K_in={K_in}, I={I}, K_out={K_out}, CTAs={M // 16}")
    print(f"  [Stage 0] prepacked-fragment path:    {t_prepack:>9.2f} μs (median)")
    print(f"  [Stage 1] row-major global path:      {t_global:>9.2f} μs (median)")
    print(f"  [Stage 2] cp.async double-buffered:   {t_cpasync:>9.2f} μs (median)")

    for W in (2, 4):
        if M % (16 * W) != 0:
            continue
        _ = run_fused_silu_cpasync_mw(
            x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf, warps_per_cta=W,
        )
        t_mw = _bench(
            lambda W=W: run_fused_silu_cpasync_mw(
                x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf, warps_per_cta=W,
            ),
            warmup=args.warmup, iters=args.iters,
        )
        print(f"  [Stage 3] cp.async multi-warp W={W}:    {t_mw:>9.2f} μs (median)  [{M // (16*W)} CTAs]")


if __name__ == "__main__":
    main()
