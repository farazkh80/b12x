#!/usr/bin/env python3
"""Autotune the v5 prefill kernel over ab_stage and tile_k.

Currently varies only the smem-cheap knobs (ab_stage, tile_k); the
atom_layout / permutation_mnk wiring is hard-coded for tile_M=128,
tile_N=64, so changing those requires re-deriving the layout.  Scope
this sweep to what's safe.

Usage:
    python scripts/tune_prefill_v5.py --shapes all --m-list 2048
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Callable, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

os.environ.setdefault("B12X_GEMM_W4A16_USE_CUTE", "1")

from b12x.gemm.w4a16 import quantize_dense_weight_to_fp4, dense_reference_w4a16
from b12x.gemm.w4a16._cute_prefill_kernel import DenseGemmW4A16CutePrefillKernel
from b12x.moe.fused.reference import compare_to_reference


_SHAPES = [
    ("q_proj",            2688,  4096),   # N=4096, num_n_tiles=64 (even)
    ("k_proj",            2688,   256),   # N=256,  num_n_tiles=4  (even, small)
    ("o_proj",            4096,  2688),   # N=2688, num_n_tiles=42 (even)
    ("shared.up",         2688,  3712),   # N=3712, num_n_tiles=58 (even)
    ("shared.dn",         3712,  2688),   # N=2688, num_n_tiles=42 (even)
    ("mamba_in_proj",     2688, 10304),   # N=10304, num_n_tiles=161 (odd)
    ("mamba_output_proj", 4096,  2688),   # N=2688, num_n_tiles=42 (even)
]

# Configs to try.  tile_M=128, tile_N=64 fixed (atom_layout dependency).
# n_per_cta=2 skipped when num_n_tiles is odd.
_CONFIGS: List[Tuple[int, int, int]] = [
    # (tile_k, ab_stage, n_per_cta)
    (64, 2, 1),   # baseline n1
    (64, 3, 1),
    (32, 2, 1),
    (32, 3, 1),
    (32, 4, 1),
    (64, 2, 2),   # baseline n2 (when applicable)
    (64, 3, 2),
    (32, 2, 2),
    (32, 3, 2),
    (32, 4, 2),
]


def _bench(fn: Callable, warmup: int = 5, iters: int = 30) -> float:
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


def _check_accuracy(kernel, x, w_fp4, w_bs, w_alpha) -> bool:
    out = kernel(x, w_fp4, w_bs, w_alpha)
    ref = dense_reference_w4a16(
        x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu(),
    ).to(x.device)
    m = compare_to_reference(out, ref)
    ref_max = ref.abs().max().item()
    return m.cos > 0.9999 and m.max_abs <= max(0.04, 0.01 * ref_max)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="all")
    p.add_argument("--m-list", default="2048")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"SM={props.major}{props.minor}  GPU={props.name}\n")

    if args.shapes == "all":
        shapes = _SHAPES
    else:
        names = set(args.shapes.split(","))
        shapes = [s for s in _SHAPES if s[0] in names]
    ms = [int(x) for x in args.m_list.split(",") if x]

    for name, k, n in shapes:
        torch.manual_seed(args.seed)
        w = (torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
        w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)

        num_n_tiles = n // 64
        print(f"\n=== {name}  K={k} N={n} (num_n_tiles={num_n_tiles}) ===")
        hdr = f"{'M':>5} {'tile_K':>6} {'ab':>3} {'n/cta':>5} {'us':>8} {'vs base':>8} {'status':>8}"
        print(hdr)
        print("-" * len(hdr))

        for m in ms:
            torch.manual_seed(args.seed + m)
            x = (torch.randn(m, k, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
            out_buf = torch.empty(m, n, dtype=torch.bfloat16, device=dev)

            baseline_us = None
            results = []
            for tile_k, ab, npc in _CONFIGS:
                # Skip n_per_cta=2 when num_n_tiles isn't even.
                if npc > 1 and num_n_tiles % npc != 0:
                    continue
                # Skip impossible smem configs.
                sA = 128 * tile_k * 2 * ab
                sB = 64 * tile_k * 2 * ab
                sC = 128 * 64 * 2
                if sA + sB + sC > 100 * 1024:
                    continue
                # K must be a multiple of tile_k.
                if k % tile_k != 0:
                    continue

                try:
                    kern = DenseGemmW4A16CutePrefillKernel(
                        tile_k=tile_k, ab_stage=ab, n_per_cta=npc,
                    )
                    ok = _check_accuracy(kern, x, w_fp4, w_bs, w_alpha)
                    if not ok:
                        results.append((tile_k, ab, npc, None, "FAIL_ACC"))
                        continue
                    us = _bench(lambda: kern(x, w_fp4, w_bs, w_alpha, out=out_buf),
                                warmup=args.warmup, iters=args.iters)
                    if baseline_us is None and tile_k == 64 and ab == 2 and npc == 1:
                        baseline_us = us
                    results.append((tile_k, ab, npc, us, "OK"))
                except Exception as e:
                    results.append((tile_k, ab, npc, None, f"FAIL: {str(e)[:30]}"))

            # Print sorted by time
            ok_results = [r for r in results if r[3] is not None]
            ok_results.sort(key=lambda r: r[3])
            best = ok_results[0] if ok_results else None
            for tile_k, ab, npc, us, status in results:
                if us is None:
                    print(f"{m:5d} {tile_k:>6d} {ab:>3d} {npc:>5d} {'—':>8s} {'—':>8s} {status:>8s}")
                else:
                    ratio = f"{us/baseline_us:.2f}x" if baseline_us else "—"
                    star = "  *" if (best and (tile_k, ab, npc) == (best[0], best[1], best[2])) else ""
                    print(f"{m:5d} {tile_k:>6d} {ab:>3d} {npc:>5d} {us:7.1f}us {ratio:>8s} {status:>8s}{star}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
