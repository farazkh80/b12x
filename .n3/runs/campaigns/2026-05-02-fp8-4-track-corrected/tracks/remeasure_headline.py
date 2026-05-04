from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from b12x.gemm.fp8_dense_cuda import fp8_dense_gemm_cuda

M, K, N = 32, 5376, 5376
WARMUP = 20
ITERS = 100
REPEATS = 7


def make_fp8(shape: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) / 8
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    return src.to(torch.float8_e4m3fn).contiguous(), scale


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def bench(fn, warmup: int = WARMUP, iters: int = ITERS, repeats: int = REPEATS) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return summarize(samples)


def graph_callable(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def main() -> None:
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    a, scale_a = make_fp8((M, K), 1000)
    b, scale_b = make_fp8((N, K), 2000)
    baseline_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    inline_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    def baseline():
        return torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16, out=baseline_out)

    def inline():
        return fp8_dense_gemm_cuda(a, b, scale_a, scale_b, out=inline_out)

    baseline()
    inline()
    torch.cuda.synchronize()
    ref = baseline_out.float()
    diff = (inline_out.float() - ref).abs()
    correctness = {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cos": float(torch.nn.functional.cosine_similarity(inline_out.float().flatten(), ref.flatten(), dim=0).item()),
    }

    results = {
        "shape": {"M": M, "K": K, "N": N},
        "kernel_variant": "set_by_caller",
        "warmup": WARMUP,
        "iters": ITERS,
        "repeats": REPEATS,
        "baseline_eager": bench(baseline),
        "inline_eager": bench(inline),
        "baseline_graph": bench(graph_callable(baseline)),
        "inline_graph": bench(graph_callable(inline)),
        "inline_vs_baseline_correctness": correctness,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
