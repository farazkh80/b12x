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

Marlin baseline (column ``marlin_us``): per-(shape, M) p50 µs from a
correlation pass on the Nano3.5 nsys trace (Spark GB10, SM121, mem BW
~273 GB/s) at the closest single-chunk prefill M.  Captured by
``scripts/correlate_marlin_prefill.py``; raw data in
``.claude_docs/marlin-baseline/`` (CSV + markdown).  Caveat: the
trace is from Spark hardware while this benchmark runs on the local
SM120 dev GPU (RTX PRO 6000 Blackwell), so the absolute µs aren't
directly comparable — but the v5/Marlin ratio is informative because
both kernels are W4A16 and both are memory-traffic-bound at the
weight loads.
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
from b12x.gemm.w4a16.micro import DenseGemmW4A16MicroKernel


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


# Marlin baseline p50 µs at each prefill M, captured by
# scripts/correlate_marlin_prefill.py from the Spark nsys trace
# (single-chunk iterations; M=1521 omitted as anomalous).  Keyed by
# the b12x shape name; missing entries fall back to '—' in the table.
# q_proj/k_proj/v_proj separately don't exist in the Nano3.5 trace
# (qkv is fused into a single 'self_attn_qkv_linear' call with
# N=4608 = 4096 + 256 + 256).
_MARLIN_BASELINE_US = {
    # (b12x shape, marlin_M) -> p50_us
    ("o_proj",             579):  182.0,
    ("o_proj",            1823):  526.1,
    ("o_proj",            2001):  584.3,
    ("o_proj",            2015):  578.0,
    ("shared.up",          579):  174.5,   # K=2688, N=3712 = shared_fc1
    ("shared.up",         1823):  486.7,
    ("shared.up",         2001):  536.2,
    ("shared.up",         2015):  534.3,
    ("shared.dn",          579):  163.9,   # K=3712, N=2688 = shared_fc2
    ("shared.dn",         1823):  486.3,
    ("shared.dn",         2001):  532.7,
    ("shared.dn",         2015):  531.0,
    ("mamba_in_proj",      579):  563.6,   # K=2688, N=10304
    ("mamba_in_proj",     1823):  654.1,
    ("mamba_in_proj",     2001):  814.2,
    ("mamba_in_proj",     2015):  810.0,
    ("mamba_output_proj",  579):  183.2,   # K=4096, N=2688
    ("mamba_output_proj", 1823):  526.1,
    ("mamba_output_proj", 2001):  581.8,
    ("mamba_output_proj", 2015):  578.7,
}

# Map b12x bench M values to the nearest Marlin-traced M.
_MARLIN_M_NEAREST = {
    64: None, 128: None, 256: None,
    512: 579,
    1024: None,   # no clean Marlin sample at this M
    2048: 2015,
    4096: None,
}


def _marlin_baseline_us(shape_name: str, m: int):
    marlin_M = _MARLIN_M_NEAREST.get(m)
    if marlin_M is None:
        return None
    return _MARLIN_BASELINE_US.get((shape_name, marlin_M))


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
        f"{'bf16_us':>9s}  {'v4_us':>9s}  {'v5_us':>9s}  {'v5_cfg':>9s}  "
        f"{'marlin_us':>10s}  {'v5/v4':>7s}  {'v5/marlin':>9s}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    decode_kernel = None if args.skip_decode else DenseGemmW4A16CuteDenseKernel()
    # v5 column goes through the production ``DenseGemmW4A16MicroKernel``
    # so it reflects the full autotuned dispatch: (tile_k, n_per_cta)
    # picked per-(K, N, M) per the rules in micro.py.
    prefill_dispatch = DenseGemmW4A16MicroKernel()

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

            # v5 prefill — full autotuned dispatch via micro.py.
            v5_us = None
            cfg_tile_k, cfg_npc = prefill_dispatch._pick_prefill_cfg(m, k, n)
            v5_label = f"tK{cfg_tile_k}_n{cfg_npc}"
            try:
                def call_v5():
                    return prefill_dispatch._pick_prefill(m, k, n)(
                        x, w_fp4, w_bs, w_alpha, out=out_v5,
                    )
                v5_us = _bench(call_v5, warmup=args.warmup, iters=args.iters)
            except Exception as e:
                print(f"# {name} M={m}: v5 failed: {e}", flush=True)

            def fmt(x): return f"{x:6.1f}us" if x is not None else "      -"
            def ratio(num, den):
                if num is None or den is None or den <= 0: return "      -"
                return f"{num/den:6.2f}x"

            marlin_us = _marlin_baseline_us(name, m)
            print(
                f"{name:18s} {m:5d} {k:5d} {n:6d}  "
                f"{fmt(bf16_us):>9s}  {fmt(v4_us):>9s}  {fmt(v5_us):>9s}  "
                f"{v5_label:>9s}  "
                f"{fmt(marlin_us):>10s}  "
                f"{ratio(v5_us, v4_us):>7s}  {ratio(v5_us, marlin_us):>9s}",
                flush=True,
            )
        print("-" * len(hdr), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
