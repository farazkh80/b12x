#!/usr/bin/env python3
"""Benchmark b12x W4A16 dense GEMM at prefill shapes (large M).

Compares three implementations across the Nano3.5 dense linear shapes:

* **torch.nn.functional.linear** (bf16 ref): fastest possible bf16 path,
  upper-bound reference.  Weight stays in bf16 — this is *not* the
  W4A16 production path; it's a perf floor for "what if dequant were
  free."
* **v4 decode kernel** (CuTe-DSL, ``_cute_dense_kernel.py``):
  warp-level MMA, tile (32, 64, 64), 4 MMA warps.  Supported at any M
  but optimized for M ≤ 32.
* **v5 prefill kernel** (CuTe-DSL, ``_cute_prefill_kernel.py``):
  scaled-up sibling, tile (128, 64, 64), 8 MMA warps.  Designed for
  M ≥ 64.

Reports median per-call time in microseconds.  Use ``--m-list`` to
override the default sweep.

Open question (captured in the design doc): per-M Marlin baselines for
the prefill regime are not yet available — only M=1 Marlin numbers
exist.  Once those are correlated from the Spark nsys trace, the
``_MARLIN_PER_M`` table below can be populated.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

os.environ.setdefault("B12X_GEMM_W4A16_USE_CUTE", "1")

from b12x.gemm.w4a16 import quantize_dense_weight_to_fp4
from b12x.gemm.w4a16._cute_dense_kernel import DenseGemmW4A16CuteDenseKernel
from b12x.gemm.w4a16._cute_prefill_kernel import DenseGemmW4A16CutePrefillKernel


_NANO35_DENSE_SHAPES = [
    # (name, K, N)
    ("q_proj",             2688,   4096),
    ("k_proj",             2688,    256),
    ("o_proj",             4096,   2688),
    ("shared.up",          2688,   3712),
    ("shared.dn",          3712,   2688),
    ("mamba_in_proj",      2688,  10304),
    ("mamba_output_proj",  4096,   2688),
]


def _bench(fn: Callable, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    S = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    E = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        S[i].record()
        fn()
        E[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(S, E))
    return times[iters // 2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--m-list", default="64,128,256,512,1024,2048,4096",
        help="comma-separated M values",
    )
    p.add_argument("--shapes", default="all", help="comma-separated shape names or 'all'")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-decode", action="store_true",
                   help="skip v4 decode kernel (only bench prefill + bf16 ref)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(
        f"SM={props.major}{props.minor}  GPU={props.name}  torch={torch.__version__}",
        flush=True,
    )

    ms = [int(x) for x in args.m_list.split(",") if x]
    if args.shapes == "all":
        shapes = _NANO35_DENSE_SHAPES
    else:
        names = set(args.shapes.split(","))
        shapes = [s for s in _NANO35_DENSE_SHAPES if s[0] in names]

    hdr = (
        f"{'shape':18s} {'M':>5s} {'K':>5s} {'N':>6s}  "
        f"{'bf16_us':>9s}  {'v4_us':>9s}  {'v5_us':>9s}  "
        f"{'v4/bf16':>8s}  {'v5/bf16':>8s}  {'v5/v4':>7s}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    decode_kernel = None if args.skip_decode else DenseGemmW4A16CuteDenseKernel()
    prefill_kernel = DenseGemmW4A16CutePrefillKernel()

    for name, k, n in shapes:
        torch.manual_seed(args.seed)
        w = (torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
        w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)

        for m in ms:
            torch.manual_seed(args.seed + m)
            x = (torch.randn(m, k, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
            out_v4 = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
            out_v5 = torch.empty(m, n, dtype=torch.bfloat16, device=dev)

            # bf16 reference (torch.nn.functional.linear)
            def call_bf16():
                return torch.nn.functional.linear(x, w)

            try:
                bf16_us = _bench(call_bf16, warmup=args.warmup, iters=args.iters)
            except Exception as e:
                print(f"# {name} M={m}: bf16 failed: {e}", flush=True)
                bf16_us = None

            # v4 decode kernel — call directly even at large M (it pads
            # up internally; num_m_tiles scales).  Lets us measure v5's
            # crossover empirically.
            v4_us = None
            if decode_kernel is not None and n % 64 == 0 and k % 64 == 0:
                def call_v4():
                    return decode_kernel(x, w_fp4, w_bs, w_alpha, out=out_v4)
                try:
                    v4_us = _bench(call_v4, warmup=args.warmup, iters=args.iters)
                except Exception as e:
                    print(f"# {name} M={m}: v4 failed: {e}", flush=True)

            # v5 prefill kernel
            v5_us = None
            if prefill_kernel.is_supported_instance(m, k, n):
                def call_v5():
                    return prefill_kernel(x, w_fp4, w_bs, w_alpha, out=out_v5)
                try:
                    v5_us = _bench(call_v5, warmup=args.warmup, iters=args.iters)
                except Exception as e:
                    print(f"# {name} M={m}: v5 failed: {e}", flush=True)

            def fmt(x): return f"{x:6.1f}us" if x is not None else "      -"
            def ratio(num, den):
                if num is None or den is None or den <= 0: return "      -"
                return f"{num/den:6.2f}x"

            print(
                f"{name:18s} {m:5d} {k:5d} {n:6d}  "
                f"{fmt(bf16_us):>9s}  {fmt(v4_us):>9s}  {fmt(v5_us):>9s}  "
                f"{ratio(v4_us, bf16_us):>8s}  {ratio(v5_us, bf16_us):>8s}  "
                f"{ratio(v5_us, v4_us):>7s}",
                flush=True,
            )
        print("-" * len(hdr), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
