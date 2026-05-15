#!/usr/bin/env python3
"""Benchmark b12x W4A16 dense GEMM vs TRT-LLM (and optionally FlashInfer).

Targets the dense linears of the Nano3.5-BF16-NVFP4-W4A16-LMHEAD-CT
checkpoint (q/k/v/o + shared expert up/down + lm_head).  The TRT-LLM
baseline times the *bf16->FP4 activation quantization + matmul* so the
comparison is apples-to-apples vs b12x's bf16-in path.

Run inside the dev container with both /workspace/agentic-dev/b12x and
/workspace/TensorRT-LLM mounted.  The TRT-LLM op library is loaded via
ctypes to avoid pulling in the full tensorrt_llm Python import (which
needs transformers etc.).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import pathlib
import statistics
import sys
from typing import Callable

# Make `b12x` importable when invoked as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.gemm.w4a16 import (
    dense_reference_w4a16,
    quantize_dense_weight_to_fp4,
)

# ---------------------------------------------------------------------------
# TRT-LLM op loading
# ---------------------------------------------------------------------------

_TRTLLM_LIB_CANDIDATES = [
    "/workspace/TensorRT-LLM/tensorrt_llm/libs/libtensorrt_llm.so",
    "/workspace/TensorRT-LLM/tensorrt_llm/libs/libth_common.so",
]


def _ensure_trtllm_ops_loaded() -> bool:
    """Best-effort dlopen of TRT-LLM op libraries.  Idempotent."""
    if getattr(_ensure_trtllm_ops_loaded, "_done", False):
        return _ensure_trtllm_ops_loaded._ok
    ok = True
    for path in _TRTLLM_LIB_CANDIDATES:
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            ok = False
    has_op = ok and hasattr(torch.ops.trtllm, "cuda_core_nvfp4_gemm")
    _ensure_trtllm_ops_loaded._done = True
    _ensure_trtllm_ops_loaded._ok = has_op
    return has_op


# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------

# (name, K, N) for Nano3.5 dense linears.
NANO35_SHAPES = [
    ("q_proj",             2688,   4096),
    ("k_proj",             2688,    256),
    ("v_proj",             2688,    256),
    ("o_proj",             4096,   2688),
    ("shared_expert.up",   2688,   3712),
    ("shared_expert.down", 3712,   2688),
]
NANO35_LM_HEAD = ("lm_head", 2688, 131072)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def bench_events(fn: Callable[[], None], *, warmup: int, iters: int) -> list[float] | None:
    """Return per-iter latencies in microseconds, or None if `fn` errors."""
    torch.cuda.synchronize()
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
    except RuntimeError:
        return None
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    try:
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
    except RuntimeError:
        return None
    return [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]  # ms -> us


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def make_b12x_runner(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale: torch.Tensor,
    w_alpha: torch.Tensor,
) -> Callable[[], torch.Tensor] | None:
    """b12x W4A16 dense GEMM runner.

    Returns None during Tasks 4-5 (stub stage) because the reference is
    CPU-only.  After Task 8 lands, swap to ``dense_gemm_w4a16``.
    """
    try:
        from b12x.gemm.w4a16 import dense_gemm_w4a16  # noqa: F401
    except ImportError:
        return None
    out = torch.empty(x.shape[0], w_fp4.shape[0], dtype=torch.bfloat16, device=x.device)

    def run() -> torch.Tensor:
        return dense_gemm_w4a16(x, w_fp4, w_blockscale, w_alpha, out=out)

    return run


def make_trtllm_runner(
    x_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
) -> Callable[[], torch.Tensor] | None:
    """TRT-LLM cuda_core_nvfp4_gemm baseline (bf16->FP4 quant + matmul, both timed)."""
    if not _ensure_trtllm_ops_loaded():
        return None
    sf_vec_size = 16
    global_scale = torch.tensor(1.0, dtype=torch.float32, device=w_bf16.device)
    # Pre-quantize the weight (offline cost, mirrors checkpoint storage).
    w_fp4_packed, w_sf = torch.ops.trtllm.fp4_quantize(
        w_bf16, global_scale, sf_vec_size, False, True
    )
    alpha = torch.tensor(1.0, dtype=torch.float32, device=w_bf16.device)

    def run() -> torch.Tensor:
        x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(
            x_bf16, global_scale, sf_vec_size, False, True
        )
        return torch.ops.trtllm.cuda_core_nvfp4_gemm(
            x_fp4, w_fp4_packed, x_sf, w_sf, alpha,
            None, torch.bfloat16, 0, None,
        )

    return run


def make_flashinfer_runner(
    x_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
) -> Callable[[], torch.Tensor] | None:
    """FlashInfer mm_fp4 baseline.

    Disabled in v1: ``mm_fp4`` expects B as ``(K_packed, N)`` storage,
    while ``trtllm.fp4_quantize`` (and our reference packer) produce
    ``(N, K_packed)``.  Transposing a packed-FP4 tensor (nibble-level
    transpose) is non-trivial and not needed for the head-to-head this
    bench targets — the TRT-LLM ``cuda_core_nvfp4_gemm`` path is the
    kernel we are replacing.  Left as a stub for a v2 follow-up.
    """
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def make_inputs(m: int, k: int, n: int, *, device: torch.device, seed: int = 0):
    torch.manual_seed(seed)
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_fp4, w_blockscale, w_alpha = quantize_dense_weight_to_fp4(w)
    return x, w, w_fp4, w_blockscale, w_alpha


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-list", type=str, default="1,8,16,32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lm-head", action="store_true",
                    help="Include the lm_head (2688x131072) shape.")
    ap.add_argument(
        "--baselines", type=str, default="trtllm,flashinfer",
        help="Comma-separated subset of {trtllm,flashinfer}.",
    )
    ap.add_argument("--output", type=pathlib.Path, default=None)
    ap.add_argument("--skip-b12x", action="store_true",
                    help="Skip timing the b12x kernel (pre-Task-8 harness validation).")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    m_list = [int(s) for s in args.m_list.split(",")]
    baselines = {s.strip() for s in args.baselines.split(",") if s.strip()}

    shapes = list(NANO35_SHAPES)
    if args.lm_head:
        shapes.append(NANO35_LM_HEAD)

    rows: list[dict] = []
    for name, k, n in shapes:
        for m in m_list:
            x_bf16, w_bf16, w_fp4, w_blockscale, w_alpha = make_inputs(
                m, k, n, device=device, seed=hash((name, m)) & 0xFFFFFFFF,
            )
            row = {"shape": name, "M": m, "K": k, "N": n,
                   "b12x_us": None, "trtllm_us": None, "fi_us": None}

            if not args.skip_b12x:
                r = make_b12x_runner(x_bf16, w_fp4, w_blockscale, w_alpha)
                if r is not None:
                    t = bench_events(r, warmup=args.warmup, iters=args.iters)
                    if t is not None:
                        row["b12x_us"] = round(statistics.median(t), 3)

            if "trtllm" in baselines:
                r = make_trtllm_runner(x_bf16, w_bf16)
                if r is not None:
                    t = bench_events(r, warmup=args.warmup, iters=args.iters)
                    if t is not None:
                        row["trtllm_us"] = round(statistics.median(t), 3)

            if "flashinfer" in baselines:
                r = make_flashinfer_runner(x_bf16, w_bf16)
                if r is not None:
                    t = bench_events(r, warmup=args.warmup, iters=args.iters)
                    if t is not None:
                        row["fi_us"] = round(statistics.median(t), 3)

            # Pretty-print one row
            rows.append(row)
            cells = [
                f"{row['shape']:>22}",
                f"M={row['M']:>3}",
                f"K={row['K']:>5}",
                f"N={row['N']:>6}",
                f"b12x={row['b12x_us'] if row['b12x_us'] is not None else '-':>8}",
                f"trtllm={row['trtllm_us'] if row['trtllm_us'] is not None else '-':>8}",
                f"fi={row['fi_us'] if row['fi_us'] is not None else '-':>8}",
            ]
            print(" | ".join(cells))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["shape", "M", "K", "N", "b12x_us", "trtllm_us", "fi_us"]
        with args.output.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
