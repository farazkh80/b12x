#!/usr/bin/env python3
"""Drive tune_gemm against every shape in a cublaslt_shapes.json file.

Bridges trtllm-bench's cublaslt_shapes.json output to tune_gemm. For each
shape (filtered by dtype), invokes tune_gemm with the shape's M/K/N/dtype/
trans/compute fields, captures the output JSON next to the input, and
emits a markdown leaderboard.

Usage:
    tune_shapes_batch.py \\
        --shapes-json .../cublaslt_shapes.json \\
        --out-dir     .../tune_output/        \\
        [--dtype-filter fp4_e2m1|e4m3|bf16|all]   default: fp4_e2m1
        [--top-n N]                                default: all
        [--tune-gemm /path/to/tune_gemm]           default: same dir as this script
        [--cmd-prefix "command_on_spark.sh "]      default: empty (run directly)
        [--request-count 16] [--warmup 5] [--iters 20] [--repeats 3]
        [--keep-going]    keep tuning even if one shape errors (default: stop on first non-zero exit)

Outputs:
    <out-dir>/tune_M<M>_K<K>_N<N>_<a>_<trans>.json     per-shape tune_gemm output
    <out-dir>/tune_M<M>_K<K>_N<N>_<a>_<trans>.stderr   per-shape stderr
    <out-dir>/batch_summary.json                       aggregated machine-readable
    <out-dir>/batch_summary.md                         human-readable leaderboard
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# tune_gemm only knows these dtype names; if the shape JSON has something
# else, skip the shape with a clear note.
SUPPORTED_INPUT_DTYPES = {"bf16", "fp16", "fp32", "e4m3", "e5m2", "int8", "fp4_e2m1"}
SUPPORTED_COMPUTE = {"fp32", "fp16", "fp32_tf32", "int32"}


def shape_label(s: dict) -> str:
    return (f"M{s['M']}_K{s['K']}_N{s['N']}_"
            f"{s['a_dtype']}_{s['trans_a']}{s['trans_b']}")


def run_one(tune_gemm: str, cmd_prefix: list[str], shape: dict, out_dir: Path,
            request_count: int, warmup: int, iters: int, repeats: int) -> dict:
    label = shape_label(shape)
    out_json = out_dir / f"tune_{label}.json"
    out_err = out_dir / f"tune_{label}.stderr"

    args = [
        tune_gemm,
        "--M", str(shape["M"]),
        "--K", str(shape["K"]),
        "--N", str(shape["N"]),
        "--a-dtype", shape["a_dtype"],
        "--b-dtype", shape["b_dtype"],
        "--c-dtype", shape["c_dtype"],
        "--compute", shape["compute_type"],
        "--trans-a", shape["trans_a"],
        "--trans-b", shape["trans_b"],
        "--request-count", str(request_count),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--repeats", str(repeats),
    ]
    full_cmd = cmd_prefix + args

    t0 = time.time()
    rc = subprocess.run(
        full_cmd,
        stdout=out_json.open("wb"),
        stderr=out_err.open("wb"),
    ).returncode
    elapsed = time.time() - t0

    record: dict[str, Any] = {
        "shape": {k: shape[k] for k in (
            "M", "K", "N", "trans_a", "trans_b",
            "a_dtype", "b_dtype", "c_dtype", "compute_type",
        )},
        "call_count": shape.get("call_count"),
        "rank_by_calls": shape.get("rank_by_calls"),
        "exit_code": rc,
        "elapsed_s": round(elapsed, 2),
        "tune_json_path": str(out_json.relative_to(out_dir.parent)) if out_dir.parent in out_json.parents else str(out_json),
        "tune_stderr_path": str(out_err.relative_to(out_dir.parent)) if out_dir.parent in out_err.parents else str(out_err),
    }

    if rc == 0 and out_json.is_file() and out_json.stat().st_size > 0:
        try:
            tune = json.loads(out_json.read_text())
            record["heuristic_returned"] = tune.get("heuristic_returned", 0)
            hb = tune.get("heuristic_baseline")
            best = tune.get("best")
            record["baseline_us"] = hb["median_us"] if hb else None
            record["best_us"] = best["median_us"] if best else None
            record["best_algo"] = (
                f"algo={best['algo_id']} tile={best['tile_id']} stages={best['stages_id']} "
                f"splitK={best['splitk_num']} swizzle={best['swizzling']}"
                if best else None
            )
            record["speedup"] = best["speedup_vs_heuristic_baseline"] if best else None
        except json.JSONDecodeError as e:
            record["parse_error"] = str(e)
    return record


def render_md(records: list[dict], shapes_blob: dict, args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append(f"# tune_shapes_batch — {args.shapes_json}")
    lines.append("")
    lines.append(f"**Source bench**: model=`{shapes_blob.get('model','')}`, "
                 f"trtllm_sha=`{shapes_blob.get('trtllm_git_sha','')[:7] or '?'}`, "
                 f"device=`{shapes_blob.get('device','')}`")
    lines.append(f"**Filter**: dtype={args.dtype_filter}, top_n={args.top_n or 'all'}")
    lines.append(f"**Tuned**: {len(records)} shape(s); "
                 f"{sum(1 for r in records if r.get('best_us') is not None)} succeeded.")
    lines.append("")

    def fmt(x: Any, w: int = 8, p: int = 2) -> str:
        if x is None: return "—"
        if isinstance(x, float): return f"{x:.{p}f}"
        return str(x)

    lines.append("| rank | calls | M | K | N | trans | dt | heur | baseline (µs) | best (µs) | speedup | best algo |")
    lines.append("|---:|---:|---:|---:|---:|:--:|:--:|---:|---:|---:|---:|:--|")
    sorted_records = sorted(records, key=lambda r: r.get("rank_by_calls") or 1e9)
    for r in sorted_records:
        s = r["shape"]
        lines.append(
            f"| {fmt(r.get('rank_by_calls'))} | {fmt(r.get('call_count'))} "
            f"| {s['M']} | {s['K']} | {s['N']} | {s['trans_a']}{s['trans_b']} "
            f"| {s['a_dtype']} "
            f"| {fmt(r.get('heuristic_returned'))} "
            f"| {fmt(r.get('baseline_us'))} "
            f"| {fmt(r.get('best_us'))} "
            f"| {fmt(r.get('speedup'), p=3)}× "
            f"| {r.get('best_algo') or '—'} |"
        )
    lines.append("")
    lines.append(f"_Total wall-time: {sum(r.get('elapsed_s', 0) for r in records):.1f}s_")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype-filter", default="fp4_e2m1",
                    help="filter shapes by a_dtype (or 'all'); default fp4_e2m1")
    ap.add_argument("--top-n", type=int, default=0,
                    help="only tune the top N shapes by call_count; 0 = all")
    ap.add_argument("--tune-gemm", default="",
                    help="path to tune_gemm binary; default = sibling 'tune_gemm' next to this script")
    ap.add_argument("--cmd-prefix", default="",
                    help="optional prefix to prepend (e.g. 'command_on_spark.sh ')")
    ap.add_argument("--request-count", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    tune_gemm = args.tune_gemm or str(here / "tune_gemm")
    cmd_prefix = shlex.split(args.cmd_prefix) if args.cmd_prefix else []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes_blob = json.loads(Path(args.shapes_json).read_text())
    shapes = shapes_blob.get("shapes", [])

    if args.dtype_filter != "all":
        shapes = [s for s in shapes if s.get("a_dtype") == args.dtype_filter]

    # skip shapes tune_gemm doesn't know
    skipped = []
    runnable = []
    for s in shapes:
        for k in ("a_dtype", "b_dtype", "c_dtype"):
            if s.get(k) not in SUPPORTED_INPUT_DTYPES:
                skipped.append({"shape": s, "reason": f"unsupported {k}={s.get(k)}"})
                break
        else:
            if s.get("compute_type") not in SUPPORTED_COMPUTE:
                skipped.append({"shape": s, "reason": f"unsupported compute={s.get('compute_type')}"})
                continue
            runnable.append(s)

    if args.top_n > 0:
        runnable = runnable[:args.top_n]

    print(f"[batch] tuning {len(runnable)} shape(s); skipping {len(skipped)} as unsupported",
          file=sys.stderr)
    if skipped:
        for sk in skipped[:5]:
            print(f"[batch]   skip: {shape_label(sk['shape'])} ({sk['reason']})",
                  file=sys.stderr)

    records: list[dict] = []
    for i, s in enumerate(runnable):
        label = shape_label(s)
        print(f"[batch] {i+1}/{len(runnable)} {label} ...", file=sys.stderr, flush=True)
        rec = run_one(tune_gemm, cmd_prefix, s, out_dir,
                      args.request_count, args.warmup, args.iters, args.repeats)
        records.append(rec)
        if rec["exit_code"] != 0 and not args.keep_going:
            print(f"[batch] ABORT — {label} exit={rec['exit_code']}; "
                  f"see {rec['tune_stderr_path']}. Pass --keep-going to continue.",
                  file=sys.stderr)
            break
        if rec.get("best_us") is not None:
            print(f"[batch]   ok: {rec['heuristic_returned']} cands, "
                  f"baseline={rec['baseline_us']:.1f}us best={rec['best_us']:.1f}us "
                  f"speedup={rec['speedup']:.3f}x", file=sys.stderr)
        else:
            print(f"[batch]   no candidates", file=sys.stderr)

    summary = {
        "shapes_json": str(Path(args.shapes_json).resolve()),
        "source_bench": {
            "model": shapes_blob.get("model"),
            "trtllm_git_sha": shapes_blob.get("trtllm_git_sha"),
            "device": shapes_blob.get("device"),
            "run_id": shapes_blob.get("run_id"),
        },
        "filter": {"dtype": args.dtype_filter, "top_n": args.top_n},
        "tune_args": {
            "request_count": args.request_count,
            "warmup": args.warmup, "iters": args.iters, "repeats": args.repeats,
        },
        "skipped_count": len(skipped),
        "tuned_count": len(records),
        "results": records,
    }
    (out_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "batch_summary.md").write_text(render_md(records, shapes_blob, args))
    print(f"[batch] wrote {out_dir/'batch_summary.json'} and {out_dir/'batch_summary.md'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
