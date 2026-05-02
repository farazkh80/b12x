#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.gemm.fp8_dense_cuda import fp8_dense_gemm_cuda


SHAPES_DECODE = [
    (1, 4096, 4096),
    (1, 4096, 5376),
    (1, 5376, 4096),
    (4, 4096, 5376),
    (32, 4096, 5376),
    (32, 5376, 5376),  # fixed headline shape from production nsys profile
    (80, 4096, 5376),
]
SHAPES_PREFILL = [
    (8192, 4096, 5376),
    (8192, 5376, 4096),
    (16384, 4096, 5376),
    (32768, 4096, 5376),
]
SHAPES_DECODE_TP4 = [(m, k, n // 4) for (m, k, n) in SHAPES_DECODE]
SHAPES_PREFILL_TP4 = [(m, k, n // 4) for (m, k, n) in SHAPES_PREFILL]

COS_THRESHOLD = 0.9995
MAX_ABS_REF_FRACTION = 0.01
_L2_FLUSH_CACHE: dict[tuple[int, int], torch.Tensor] = {}


@dataclass(frozen=True)
class ShapeCase:
    group: str
    m: int
    k: int
    n: int

    @property
    def key(self) -> str:
        return f"{self.group}:M={self.m},K={self.k},N={self.n}"


def canonical_shapes() -> list[ShapeCase]:
    groups = [
        ("decode", SHAPES_DECODE),
        ("prefill", SHAPES_PREFILL),
        ("decode_tp4", SHAPES_DECODE_TP4),
        ("prefill_tp4", SHAPES_PREFILL_TP4),
    ]
    cases: list[ShapeCase] = []
    seen: set[tuple[str, int, int, int]] = set()
    for group, shapes in groups:
        for m, k, n in shapes:
            item = (group, m, k, n)
            if item not in seen:
                cases.append(ShapeCase(group, m, k, n))
                seen.add(item)
    return cases


def parse_shape(text: str) -> ShapeCase:
    values: dict[str, int] = {}
    for part in text.replace(",", " ").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip().lower()] = int(value)
    if not {"m", "k", "n"}.issubset(values):
        raise argparse.ArgumentTypeError("shape must contain M=<int>,K=<int>,N=<int>")
    return ShapeCase("custom", values["m"], values["k"], values["n"])


def require_sm120() -> None:
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise RuntimeError(f"Requires sm_120/sm_121, got sm_{major}{minor}")


def make_fp8(shape: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    source = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) / 8
    amax = source.abs().max().clamp_min(1e-6).to(torch.float32)
    q_scale = torch.finfo(torch.float8_e4m3fn).max / amax
    quantized = (source * q_scale).to(torch.float8_e4m3fn).contiguous()
    dequant_scale = q_scale.reciprocal().reshape(1).contiguous()
    return quantized, dequant_scale, source


def resolve_l2_flush_bytes(bytes_hint: int) -> int:
    if bytes_hint < 0:
        raise ValueError(f"l2 flush bytes must be non-negative, got {bytes_hint}")
    if bytes_hint > 0:
        return int(bytes_hint)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = int(getattr(props, "l2_cache_size", 0) or 0)
    return 2 * l2_bytes if l2_bytes > 0 else 32 << 20


def make_l2_flush_fn(enabled: bool, bytes_hint: int) -> Callable[[], None] | None:
    if not enabled:
        return None
    flush_bytes = resolve_l2_flush_bytes(bytes_hint)
    device_idx = torch.cuda.current_device()
    key = (device_idx, flush_bytes)
    buffer = _L2_FLUSH_CACHE.get(key)
    if buffer is None:
        buffer = torch.empty(flush_bytes, dtype=torch.uint8, device=f"cuda:{device_idx}")
        _L2_FLUSH_CACHE[key] = buffer

    def flush(cache_buffer: torch.Tensor = buffer) -> None:
        cache_buffer.bitwise_not_()

    return flush


def summarize_samples(samples_ms: list[float]) -> dict[str, float]:
    samples_us = [x * 1000.0 for x in samples_ms]
    return {
        "median_us": float(statistics.median(samples_us)),
        "mean_us": float(statistics.mean(samples_us)),
        "min_us": float(min(samples_us)),
        "max_us": float(max(samples_us)),
    }


def summarize_repeats(runs_ms: list[list[float]]) -> dict[str, Any]:
    repeat_medians_us = [statistics.median(run) * 1000.0 for run in runs_ms]
    repeat_mins_us = [min(run) * 1000.0 for run in runs_ms]
    samples_us = [sample * 1000.0 for run in runs_ms for sample in run]
    median_us = float(statistics.median(repeat_medians_us))
    spread_pct = 0.0 if median_us == 0.0 else float((max(repeat_medians_us) - min(repeat_medians_us)) / median_us * 100.0)
    return {
        "median_us": median_us,
        "mean_us": float(statistics.mean(samples_us)),
        "min_us": float(min(samples_us)),
        "max_us": float(max(samples_us)),
        "sample_min_us": float(min(repeat_mins_us)),
        "repeat_medians_us": [float(x) for x in repeat_medians_us],
        "repeat_median_min_us": float(min(repeat_medians_us)),
        "repeat_median_max_us": float(max(repeat_medians_us)),
        "repeat_median_spread_pct": spread_pct,
    }


def bench_events(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    l2_flush: Callable[[], None] | None,
) -> dict[str, Any]:
    runs: list[list[float]] = []
    for _ in range(repeats):
        for _ in range(warmup):
            if l2_flush is not None:
                l2_flush()
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            if l2_flush is not None:
                l2_flush()
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        runs.append([start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)])
    summary = summarize_repeats(runs)
    summary["warmup"] = warmup
    summary["iters"] = iters
    summary["repeats"] = repeats
    return summary


def capture_graph(fn: Callable[[], Any]) -> Callable[[], None]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    return replay


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    denom = torch.linalg.vector_norm(a_f) * torch.linalg.vector_norm(b_f)
    if float(denom.item()) == 0.0:
        return 1.0 if bool(torch.equal(a_f, b_f)) else 0.0
    return float(torch.dot(a_f, b_f).div(denom).item())


def oracle_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    cand_f = candidate.float()
    ref_f = reference.float()
    diff = cand_f - ref_f
    ref_norm_inf = float(ref_f.abs().max().item())
    max_abs = float(diff.abs().max().item())
    rmse = float(diff.square().mean().sqrt().item())
    cos = cosine_similarity(candidate, reference)
    finite = bool(torch.isfinite(cand_f).all().item() and torch.isfinite(ref_f).all().item())
    max_abs_limit = MAX_ABS_REF_FRACTION * ref_norm_inf
    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "cos": cos,
        "ref_norm_inf": ref_norm_inf,
        "max_abs_limit": max_abs_limit,
        "finite": finite,
        "passed": bool(finite and cos >= COS_THRESHOLD and max_abs <= max_abs_limit),
    }


def make_reference(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    return (a.float() * scale_a.float()) @ (b.float() * scale_b.float()).t()


def run_100_replay_check(replay: Callable[[], None], out: torch.Tensor) -> bool:
    for _ in range(100):
        replay()
    torch.cuda.synchronize()
    return bool(torch.isfinite(out).all().item())


def empty_timing(reason: str) -> dict[str, Any]:
    return {"median_us": None, "error": reason}


def ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def pad_fp8_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[0] == rows:
        return x
    padded = torch.zeros((rows, x.shape[1]), device=x.device, dtype=x.dtype)
    padded[: x.shape[0], :].copy_(x)
    return padded.contiguous()


def run_shape(case: ShapeCase, args: argparse.Namespace, l2_flush: Callable[[], None] | None) -> dict[str, Any]:
    a, scale_a, _ = make_fp8((case.m, case.k), 1000 + case.m + case.k + case.n)
    b, scale_b, _ = make_fp8((case.n, case.k), 2000 + case.m + case.k + case.n)
    baseline_out = torch.empty((case.m, case.n), device="cuda", dtype=torch.bfloat16)
    inline_m = ceil_multiple(case.m, 16)
    inline_n = ceil_multiple(case.n, 32)
    inline_a = pad_fp8_rows(a, inline_m)
    inline_b = pad_fp8_rows(b, inline_n)
    inline_out_full = torch.empty((inline_m, inline_n), device="cuda", dtype=torch.bfloat16)

    def baseline_launch() -> torch.Tensor:
        return torch._scaled_mm(
            a,
            b.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
            out=baseline_out,
        )

    def inline_launch() -> torch.Tensor:
        return fp8_dense_gemm_cuda(inline_a, inline_b, scale_a, scale_b, out=inline_out_full)

    baseline_launch()
    torch.cuda.synchronize()
    reference = make_reference(a, b, scale_a, scale_b)
    baseline_metrics = oracle_metrics(baseline_out, reference)

    row: dict[str, Any] = {
        "shape_group": case.group,
        "M": case.m,
        "K": case.k,
        "N": case.n,
        "ab_dtype": "float8_e4m3fn",
        "c_dtype": "bfloat16",
        "scaling": "per_tensor",
        "baseline_kernel": "torch._scaled_mm",
        "baseline_oracle": baseline_metrics,
    }

    baseline_graph = capture_graph(baseline_launch)
    row["baseline_eager"] = bench_events(
        baseline_launch,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        l2_flush=l2_flush,
    )
    row["baseline_graph"] = bench_events(
        baseline_graph,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        l2_flush=l2_flush,
    )
    row["baseline_eager_median_us"] = row["baseline_eager"]["median_us"]
    row["baseline_graph_median_us"] = row["baseline_graph"]["median_us"]
    row["baseline_graph_100_replay_finite"] = run_100_replay_check(baseline_graph, baseline_out)

    try:
        inline_launch()
        torch.cuda.synchronize()
        inline_out = inline_out_full[: case.m, : case.n]
        inline_metrics = oracle_metrics(inline_out, reference)
        inline_graph = capture_graph(inline_launch)
        row["b12x_inline_ptx_oracle"] = inline_metrics
        row["b12x_inline_ptx_eager"] = bench_events(
            inline_launch,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            l2_flush=l2_flush,
        )
        row["b12x_inline_ptx_graph"] = bench_events(
            inline_graph,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            l2_flush=l2_flush,
        )
        row["b12x_inline_ptx_eager_us"] = row["b12x_inline_ptx_eager"]["median_us"]
        row["b12x_inline_ptx_graph_us"] = row["b12x_inline_ptx_graph"]["median_us"]
        row["b12x_inline_ptx_graph_100_replay_finite"] = run_100_replay_check(inline_graph, inline_out)
        row["b12x_inline_ptx_error"] = None
    except Exception as exc:
        torch.cuda.synchronize()
        reason = f"{type(exc).__name__}: {exc}"
        row["b12x_inline_ptx_oracle"] = {"passed": False, "error": reason}
        row["b12x_inline_ptx_eager"] = empty_timing(reason)
        row["b12x_inline_ptx_graph"] = empty_timing(reason)
        row["b12x_inline_ptx_eager_us"] = None
        row["b12x_inline_ptx_graph_us"] = None
        row["b12x_inline_ptx_graph_100_replay_finite"] = False
        row["b12x_inline_ptx_error"] = reason

    return row


def select_shapes(args: argparse.Namespace) -> list[ShapeCase]:
    if args.shapes:
        return args.shapes
    cases = canonical_shapes()
    if args.limit_shapes is not None:
        cases = cases[: args.limit_shapes]
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and benchmark FP8 dense GEMM baselines on SM120/SM121.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shape", dest="shapes", action="append", type=parse_shape, help="Custom shape, e.g. M=80,K=4096,N=5376. May be repeated.")
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    require_sm120()
    torch.set_grad_enabled(False)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    cases = select_shapes(args)

    entries = []
    for idx, case in enumerate(cases):
        print(f"[{idx + 1}/{len(cases)}] {case.key}", file=sys.stderr, flush=True)
        entries.append(run_shape(case, args, l2_flush))

    payload = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "measurement": {
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "flush_l2": args.flush_l2,
            "l2_flush_bytes": resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0,
            "cos_threshold": COS_THRESHOLD,
            "max_abs_ref_fraction": MAX_ABS_REF_FRACTION,
        },
        "entries": entries,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
