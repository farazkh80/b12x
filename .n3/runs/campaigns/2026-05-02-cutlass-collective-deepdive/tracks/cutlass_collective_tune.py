from __future__ import annotations

import os
import statistics

import torch

from b12x.gemm.fp8_dense_cutlass_collective import fp8_dense_gemm_cutlass_collective


def measure(fn, warmup: int, iters: int, repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / iters)
    return times


def main() -> None:
    torch.manual_seed(0)
    m, k, n = 32, 5376, 5376
    warmup = int(os.getenv("B12X_TUNE_WARMUP", "10"))
    iters = int(os.getenv("B12X_TUNE_ITERS", "50"))
    repeats = int(os.getenv("B12X_TUNE_REPEATS", "5"))

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
    scale_a = torch.ones((), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((), device="cuda", dtype=torch.float32)

    out = fp8_dense_gemm_cutlass_collective(a, b, scale_a, scale_b)
    ref = torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs()
    print("correct", {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cos": float(torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()),
    })

    baseline_times = measure(
        lambda: torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16),
        warmup,
        iters,
        repeats,
    )
    cutlass_times = measure(
        lambda: fp8_dense_gemm_cutlass_collective(a, b, scale_a, scale_b),
        warmup,
        iters,
        repeats,
    )
    print("torch", {"median": statistics.median(baseline_times), "min": min(baseline_times), "all": baseline_times})
    print("cutlass", {"median": statistics.median(cutlass_times), "min": min(cutlass_times), "all": cutlass_times})
    print("ratio", statistics.median(cutlass_times) / statistics.median(baseline_times))


if __name__ == "__main__":
    main()
