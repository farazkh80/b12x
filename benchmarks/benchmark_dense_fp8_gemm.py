#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable

import torch

from b12x.gemm.fp8_dense import fp8_dense_gemm, fp8_dense_gemm_mma
from b12x.gemm.fp8_dense_cuda import fp8_dense_gemm_cuda


def make_fp8(shape: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) / 8
    amax = src.abs().max().clamp_min(1e-6).to(torch.float32)
    q_scale = torch.finfo(torch.float8_e4m3fn).max / amax
    return (src * q_scale).to(torch.float8_e4m3fn).contiguous(), q_scale.reciprocal().reshape(1)


def bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return {
        "median_us": statistics.median(times),
        "mean_us": statistics.mean(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--k", type=int, default=5376)
    parser.add_argument("--n", type=int, default=5376)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--mode", choices=("scalar", "mma", "cuda"), default="scalar")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    a, scale_a = make_fp8((args.m, args.k), 1)
    b, scale_b = make_fp8((args.n, args.k), 2)
    out = torch.empty((args.m, args.n), device="cuda", dtype=torch.bfloat16)

    def ours() -> torch.Tensor:
        if args.mode == "cuda":
            return fp8_dense_gemm_cuda(a, b, scale_a, scale_b, out=out)
        if args.mode == "mma":
            return fp8_dense_gemm_mma(a, b, scale_a, scale_b, out=out)
        return fp8_dense_gemm(a, b, scale_a, scale_b, out=out, threads_per_cta=args.threads)

    def baseline() -> torch.Tensor:
        return torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)

    ours()
    base = baseline()
    torch.cuda.synchronize()
    result = {
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "mode": args.mode,
        "threads_per_cta": args.threads,
        "ours": bench(ours, args.warmup, args.iters),
        "torch_scaled_mm": bench(baseline, args.warmup, args.iters),
    }

    if args.check:
        diff = (out.float() - base.float()).abs()
        result["correctness"] = {
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
        }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
