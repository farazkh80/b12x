#!/usr/bin/env python3
"""Run v5 prefill kernel under nsys with NVTX markers for hotspot analysis.

Usage:
    # On Spark (via command_on_spark.sh wrapper):
    nsys profile -c cudaProfilerApi -t cuda,nvtx \\
        -o v5_prefill_profile \\
        python3 scripts/profile_v5_nsys.py --shape o_proj --m 2048

Captured profile windows (cudaProfilerStart/Stop):
* 10 warmup iters BEFORE start (not captured)
* 30 iters CAPTURED, each wrapped in an NVTX range:
    - "v5_prefill_<shape>_M<M>"  — the full kernel call (host + GPU)

The intent is to drop the captured iterations into ``nsys stats`` and
look at:
* ``cuda_gpu_kern_sum`` — top kernel breakdown (v5's @cute.kernel + any
  helpers).  Helps spot whether FP4 staging is a separate launch or
  fused with the main kernel.
* ``cuda_api_sum``      — host-side API overhead (cudaLaunchKernel etc.)
* ``cuda_gpu_mem_time_sum`` — memcpy/transfer time (should be ~0 for
  steady-state inner loop).
* ``nvtx_kern_sum``     — kernels attributed to the prefill NVTX range.

For kernel-internal hotspots (FP4 staging vs MMA *within* the one
kernel call) nsys can't help — switch to ncu after this run.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("B12X_GEMM_W4A16_USE_CUTE", "1")

import torch
import torch.cuda.nvtx as nvtx

from b12x.gemm.w4a16 import dense_gemm_w4a16, quantize_dense_weight_to_fp4
from b12x.gemm.w4a16.micro import DenseGemmW4A16MicroKernel


_SHAPES = {
    # name → (K, N)
    "q_proj":             (2688,   4096),
    "k_proj":             (2688,    256),
    "o_proj":             (4096,   2688),
    "shared.up":          (2688,   3712),
    "shared.dn":          (3712,   2688),
    "mamba_in_proj":      (2688,  10304),
    "mamba_output_proj":  (4096,   2688),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="o_proj", choices=list(_SHAPES.keys()))
    p.add_argument("--m", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2

    k, n = _SHAPES[args.shape]
    m = args.m
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"SM={props.major}{props.minor}  GPU={props.name}")
    print(f"shape={args.shape}  M={m}  K={k}  N={n}")

    torch.manual_seed(0)
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)

    # Resolve the dispatch config so the NVTX label includes it.
    dispatch = DenseGemmW4A16MicroKernel()
    tile_k, n_per_cta = dispatch._pick_prefill_cfg(m, k, n)
    label = f"v5_prefill_{args.shape}_M{m}_K{k}_N{n}_tK{tile_k}_n{n_per_cta}"
    print(f"dispatch cfg: tile_K={tile_k}, n_per_cta={n_per_cta}")

    # Warmup (outside capture range).  Touches the compile cache so the
    # captured iters are pure runtime.
    for _ in range(args.warmup):
        dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha, out=out)
    torch.cuda.synchronize()

    print(f"capture range: {args.iters} iters")
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.iters):
        nvtx.range_push(f"{label}_iter{i}")
        dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha, out=out)
        nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
