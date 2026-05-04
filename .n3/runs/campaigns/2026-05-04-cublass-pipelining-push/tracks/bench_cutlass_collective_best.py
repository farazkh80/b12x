"""Clean head-to-head benchmark for the best cutlass-collective config.

L2 is flushed before every measured iteration so neither path benefits from
hot weights. Run on Spark with B12X_CUTLASS_TILE_{M,N,K} + B12X_CUTLASS_SCHED_STAGES
set to the candidate config and a unique B12X_FP8_DENSE_CUTLASS_BUILD_DIR.
"""

from __future__ import annotations

import os
import statistics

import torch

from b12x.gemm.fp8_dense_cutlass_collective import fp8_dense_gemm_cutlass_collective


def make_l2_flush(num_bytes: int = 32 << 20):
    buf = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")

    def flush():
        buf.bitwise_not_()

    return flush


def bench(fn, *, warmup: int, iters: int, repeats: int, flush) -> dict:
    """Per-iteration timing. Each iter individually wrapped in a CUDA event,
    L2 flushed before the start.record(). Median of repeat-medians is reported.
    """
    repeat_medians = []
    samples_all = []
    for _ in range(repeats):
        for _ in range(warmup):
            flush()
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            flush()
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends, strict=True)]
        us.sort()
        repeat_medians.append(us[iters // 2])
        samples_all.extend(us)
    repeat_medians.sort()
    samples_all.sort()
    return {
        "median_us": repeat_medians[len(repeat_medians) // 2],
        "min_us": samples_all[0],
        "p10_us": samples_all[len(samples_all) // 10],
        "p90_us": samples_all[len(samples_all) * 9 // 10],
        "spread_pct": 100.0 * (repeat_medians[-1] - repeat_medians[0]) / repeat_medians[len(repeat_medians) // 2],
        "repeat_medians_us": repeat_medians,
    }


def main() -> None:
    torch.manual_seed(0)
    M, K, N = 32, 5376, 5376
    warmup = int(os.getenv("B12X_TUNE_WARMUP", "20"))
    iters = int(os.getenv("B12X_TUNE_ITERS", "100"))
    repeats = int(os.getenv("B12X_TUNE_REPEATS", "9"))

    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
    sa = torch.ones((), device="cuda", dtype=torch.float32)
    sb = torch.ones((), device="cuda", dtype=torch.float32)

    out = fp8_dense_gemm_cutlass_collective(a, b, sa, sb)
    ref = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    diff = (out.float() - ref.float()).abs()
    cos = float(torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0))
    print(f"correctness: max_abs={diff.max().item():.6g}  cos={cos:.10f}")

    flush = make_l2_flush()

    torch_stats = bench(
        lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16),
        warmup=warmup, iters=iters, repeats=repeats, flush=flush,
    )
    cute_stats = bench(
        lambda: fp8_dense_gemm_cutlass_collective(a, b, sa, sb),
        warmup=warmup, iters=iters, repeats=repeats, flush=flush,
    )

    cfg = (
        os.getenv("B12X_CUTLASS_TILE_M", "?"),
        os.getenv("B12X_CUTLASS_TILE_N", "?"),
        os.getenv("B12X_CUTLASS_TILE_K", "?"),
        os.getenv("B12X_CUTLASS_SCHED_STAGES", "?"),
    )

    print(f"\nshape (M,K,N) = ({M},{K},{N})")
    print(f"cutlass collective config: TILE_M={cfg[0]} TILE_N={cfg[1]} TILE_K={cfg[2]} STAGES={cfg[3]}")
    print(f"warmup={warmup}  iters={iters}  repeats={repeats}  L2-flush=on")
    print()
    print(f"{'kernel':<24} {'median µs':>10} {'min µs':>10} {'p10':>8} {'p90':>8} {'spread%':>8}")
    print(f"{'torch._scaled_mm':<24} {torch_stats['median_us']:>10.2f} {torch_stats['min_us']:>10.2f} "
          f"{torch_stats['p10_us']:>8.2f} {torch_stats['p90_us']:>8.2f} {torch_stats['spread_pct']:>8.2f}")
    print(f"{'cutlass collective':<24} {cute_stats['median_us']:>10.2f} {cute_stats['min_us']:>10.2f} "
          f"{cute_stats['p10_us']:>8.2f} {cute_stats['p90_us']:>8.2f} {cute_stats['spread_pct']:>8.2f}")
    print()
    print(f"ratio cutlass / torch median: {cute_stats['median_us'] / torch_stats['median_us']:.4f}")


if __name__ == "__main__":
    main()
