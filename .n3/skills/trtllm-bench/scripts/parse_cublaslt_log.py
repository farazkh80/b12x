#!/usr/bin/env python3
"""Parse a cublasLt.log file (CUBLASLT_LOG_LEVEL=2) into structured JSON.

Real log format observed on TRT-LLM 1.3.0rc12 / cuBLAS 13.2.1 / GB10:

  [<timestamp>][cublasLt][<pid>][Trace][<fn>] A=0X... Adesc=[type=R_... rows=N cols=N ld=N] \
    B=0X... Bdesc=[...] C=0X... Cdesc=[...] D=0X... Ddesc=[...] \
    computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T ... \
                 aScalePointer=0x... bScalePointer=0x...] \
    algo=[algoId=67 tile=MATMUL_TILE_64x32 stages=MATMUL_STAGES_64xAUTO customOption=29] \
    workSpace=0X... workSpaceSizeInBytes=N beta=N outOfPlace=N stream=0X...

One matmul = one line. <fn> is `cublasLtMatmul` (FP8/scaled path) or
`cublasLtTSTMatmul` (typed-storage BF16/FP16/FP32 path); both are captured.

Outputs:
  - cublaslt_calls.jsonl    one call per line, full details
  - cublaslt_shapes.json    deduped (M,K,N,trans,dtype...) → input format for
                            the cublas-gemm-tuning skill's `tune_gemm` binary
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LINE_HEADER = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\[cublasLt\]\[(?P<pid>\d+)\]\[(?P<level>[^\]]+)\]"
    r"\[(?P<fn>cublasLt[A-Za-z0-9_]*)\]\s*(?P<rest>.*)$"
)

DTYPE_MAP = {
    "R_16BF": "bf16",
    "R_16F": "fp16",
    "R_32F": "fp32",
    "R_64F": "fp64",
    "R_8F_E4M3": "e4m3",
    "R_8F_E5M2": "e5m2",
    "R_8I": "int8",
    "R_8U": "uint8",
    "R_32I": "int32",
    "R_4F_E2M1": "fp4_e2m1",
}

COMPUTE_MAP = {
    "COMPUTE_16F": "fp16",
    "COMPUTE_32F": "fp32",
    "COMPUTE_64F": "fp64",
    "COMPUTE_32I": "int32",
    "COMPUTE_32F_FAST_16F": "fp32_tf32",
    "COMPUTE_32F_FAST_16BF": "fp32_tf32",
    "COMPUTE_32F_FAST_TF32": "fp32_tf32",
    "COMPUTE_32F_FAST_TF32_BF": "fp32_tf32",
    "COMPUTE_32F_PEDANTIC": "fp32_pedantic",
}

# Future-proofing: any field on a Matmul line that is NOT in these sets is
# stashed under `extras` in the per-call record so we never silently drop new
# cuBLAS-Lt attributes.
KNOWN_TOP_FIELDS = {
    "A", "Adesc", "B", "Bdesc", "C", "Cdesc", "D", "Ddesc",
    "computeDesc", "algo",
    "alpha", "beta",
    "workSpace", "workSpaceSizeInBytes",
    "outOfPlace", "stream",
}
KNOWN_DESC_FIELDS = {
    "type", "rows", "cols", "ld",
    "batchCount", "strideA", "strideB", "strideC", "strideD",
    "order",
}
KNOWN_COMPUTE_FIELDS = {
    "computeType", "scaleType",
    "transa", "transb",
    "smCountTarget",
    "aScalePointer", "bScalePointer", "cScalePointer", "dScalePointer",
    "epilogue", "biasPointer", "biasDataType",
    "amaxDPointer", "amaxAuxPointer",
    "epilogueAuxDataType", "epilogueAuxLd",
    "fillMode", "pointerMode",
    "fastAccumMode",
}
KNOWN_ALGO_FIELDS = {
    "algoId",
    "tile", "stages",
    "customOption",
    "splitKNum", "splitKMode", "swizzling", "swizzle",
    "reductionScheme",
    "innerShapeId", "clusterShapeId",
}


def parse_brackets(s: str, start: int) -> tuple[str, int]:
    """s[start] must be '['. Return (inner, index_past_closing_bracket)."""
    if s[start] != "[":
        raise ValueError(f"expected '[' at {start}, saw {s[start]!r}")
    depth = 0
    i = start
    n = len(s)
    while i < n:
        c = s[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
        i += 1
    raise ValueError(f"unclosed bracket starting at {start}")


def parse_kv(s: str) -> dict[str, Any]:
    """Parse space-separated `key=value` pairs. Values may be:
      - bracketed substructure (recursively parsed)
      - bare token (no spaces)
    """
    out: dict[str, Any] = {}
    i = 0
    n = len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        # key
        j = i
        while j < n and s[j] != "=" and not s[j].isspace():
            j += 1
        if j >= n or s[j] != "=":
            # malformed token; skip to next whitespace
            while j < n and not s[j].isspace():
                j += 1
            i = j
            continue
        key = s[i:j]
        i = j + 1  # past '='
        if i >= n:
            out[key] = ""
            break
        if s[i] == "[":
            inner, i = parse_brackets(s, i)
            out[key] = parse_kv(inner)
        else:
            j = i
            while j < n and not s[j].isspace():
                j += 1
            out[key] = s[i:j]
            i = j
    return out


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, str):
            x = x.strip()
            if x.lower().startswith("0x"):
                return int(x, 16)
        return int(x)
    except (TypeError, ValueError):
        return default


def derive_mnk(adesc: dict, bdesc: dict, cdesc: dict,
               trans_a: str, trans_b: str) -> tuple[int, int, int]:
    """cuBLAS-Lt is column-major. With transa=N, A in memory is (M, K).
    With transa=T, A in memory is (K, M). Likewise B."""
    rA = safe_int(adesc.get("rows"))
    cA = safe_int(adesc.get("cols"))
    rB = safe_int(bdesc.get("rows"))
    cB = safe_int(bdesc.get("cols"))
    rC = safe_int(cdesc.get("rows"))
    # cdesc is always (M, N) in column-major regardless of trans
    M_from_C, N_from_C = rC, safe_int(cdesc.get("cols"))
    if trans_a == "T":
        K = rA
        M = cA
    else:
        M = rA
        K = cA
    if trans_b == "T":
        N = rB
    else:
        N = cB
    # Prefer cdesc for M/N if it disagrees (it's the authoritative output shape)
    if M_from_C and M_from_C != M:
        M = M_from_C
    if N_from_C and N_from_C != N:
        N = N_from_C
    return M, N, K


def parse_line(line: str, call_idx: int, byte_offset: int,
               line_no: int, log_basename: str) -> dict | None:
    """Returns a record dict on success, None only if the line is clearly not
    a Matmul event. Format drift (missing/renamed fields, bracket parse error)
    still produces a record — degraded fields become None and the raw bracket
    payload is stashed under `_raw` so nothing is silently lost. Each degraded
    record carries a `_log_pointer` so AVO can `sed -n '<line_no>p' <log>` to
    inspect the original log entry directly."""
    m = LINE_HEADER.match(line)
    if not m:
        return None
    fn = m.group("fn")
    if "Matmul" not in fn:
        return None
    rest = m.group("rest")
    parse_warnings: list[str] = []

    def log_ptr() -> dict:
        # Truncate the excerpt so a single huge line doesn't blow up the JSON.
        excerpt = line if len(line) <= 400 else (line[:400] + "...[truncated]")
        return {
            "log": log_basename,
            "line_no": line_no,
            "byte_offset": byte_offset,
            "excerpt": excerpt,
        }

    try:
        fields = parse_kv(rest)
    except ValueError as e:
        # Bracket-parser failure: keep the line and tag it for review.
        return {
            "call_idx": call_idx,
            "ts": m.group("ts"),
            "pid": safe_int(m.group("pid")),
            "fn": fn,
            "M": None, "K": None, "N": None,
            "_parse_warnings": [f"bracket_parse_error: {e}"],
            "_raw": rest[:4000],
            "_log_pointer": log_ptr(),
            "raw_log_offset": byte_offset,
        }

    missing = [k for k in ("Adesc", "Bdesc", "Cdesc") if k not in fields]
    if missing:
        parse_warnings.append(f"missing_fields: {','.join(missing)}")

    adesc = fields.get("Adesc") if isinstance(fields.get("Adesc"), dict) else {}
    bdesc = fields.get("Bdesc") if isinstance(fields.get("Bdesc"), dict) else {}
    cdesc = fields.get("Cdesc") if isinstance(fields.get("Cdesc"), dict) else {}
    ddesc = fields.get("Ddesc") if isinstance(fields.get("Ddesc"), dict) else {}
    cd = fields.get("computeDesc") if isinstance(fields.get("computeDesc"), dict) else {}
    adesc, bdesc, cdesc, ddesc, cd = adesc or {}, bdesc or {}, cdesc or {}, ddesc or {}, cd or {}

    trans_a = "T" if cd.get("transa") == "OP_T" else "N"
    trans_b = "T" if cd.get("transb") == "OP_T" else "N"

    try:
        M, N, K = derive_mnk(adesc, bdesc, cdesc, trans_a, trans_b)
    except Exception as e:  # noqa: BLE001
        M, N, K = None, None, None
        parse_warnings.append(f"mnk_derive_failed: {e}")

    a_type = adesc.get("type", "")
    b_type = bdesc.get("type", "")
    c_type = (ddesc.get("type") or cdesc.get("type") or "")
    compute_type = cd.get("computeType", "")
    scale_type = cd.get("scaleType", "")
    for label, raw_type in (("a_dtype", a_type), ("b_dtype", b_type),
                             ("c_dtype", c_type), ("compute_type", compute_type),
                             ("scale_type", scale_type)):
        # Don't warn on empty (descriptor missing) — already covered by missing_fields.
        # Do warn on a non-empty value we don't recognize.
        if raw_type and raw_type not in DTYPE_MAP and raw_type not in COMPUTE_MAP \
                and not raw_type.startswith("R_") and not raw_type.startswith("COMPUTE_"):
            parse_warnings.append(f"unknown_{label}: {raw_type}")

    algo = fields.get("algo", {}) if isinstance(fields.get("algo"), dict) else {}
    algo_id = safe_int(algo.get("algoId"), -1)
    tile = (algo.get("tile") or "").replace("MATMUL_TILE_", "")
    stages = (algo.get("stages") or "").replace("MATMUL_STAGES_", "")
    custom_option = safe_int(algo.get("customOption"), -1)

    workspace = safe_int(fields.get("workSpaceSizeInBytes"), 0)

    extras = collect_extras(fields, adesc, bdesc, cdesc, ddesc, cd, algo)

    rec = {
        "call_idx": call_idx,
        "ts": m.group("ts"),
        "pid": safe_int(m.group("pid")),
        "fn": fn,
        "M": M, "K": K, "N": N,
        "trans_a": trans_a, "trans_b": trans_b,
        "a_dtype": DTYPE_MAP.get(a_type, a_type or "?"),
        "b_dtype": DTYPE_MAP.get(b_type, b_type or "?"),
        "c_dtype": DTYPE_MAP.get(c_type, c_type or "?"),
        "compute_type": COMPUTE_MAP.get(compute_type, compute_type or "?"),
        "scale_type": DTYPE_MAP.get(scale_type, scale_type or "?"),
        "lda": safe_int(adesc.get("ld"), 0),
        "ldb": safe_int(bdesc.get("ld"), 0),
        "ldc": safe_int(cdesc.get("ld"), 0),
        "algo_id": algo_id if algo_id >= 0 else None,
        "tile": tile or None,
        "stages": stages or None,
        "custom_option": custom_option if custom_option >= 0 else None,
        "workspace_bytes": workspace,
        "has_a_scale": bool(cd.get("aScalePointer")),
        "has_b_scale": bool(cd.get("bScalePointer")),
        "raw_log_offset": byte_offset,
    }
    if extras:
        rec["extras"] = extras
    if parse_warnings:
        rec["_parse_warnings"] = parse_warnings
        rec["_log_pointer"] = log_ptr()
    return rec


def collect_extras(top: dict, adesc: dict, bdesc: dict, cdesc: dict,
                   ddesc: dict, computeDesc: dict, algo: dict) -> dict:
    """Stash every parsed field that is NOT in our KNOWN_* sets, so a future
    cuBLAS-Lt log version that adds new attributes still round-trips through
    the JSON instead of being silently discarded."""
    out: dict = {}
    extra_top = {k: v for k, v in top.items() if k not in KNOWN_TOP_FIELDS}
    if extra_top:
        out["top"] = extra_top
    for label, src in (("Adesc", adesc), ("Bdesc", bdesc),
                       ("Cdesc", cdesc), ("Ddesc", ddesc)):
        rest = {k: v for k, v in src.items() if k not in KNOWN_DESC_FIELDS}
        if rest:
            out[label] = rest
    rest = {k: v for k, v in computeDesc.items() if k not in KNOWN_COMPUTE_FIELDS}
    if rest:
        out["computeDesc"] = rest
    rest = {k: v for k, v in algo.items() if k not in KNOWN_ALGO_FIELDS}
    if rest:
        out["algo"] = rest
    return out


SHAPE_KEY_FIELDS = (
    "M", "K", "N",
    "trans_a", "trans_b",
    "a_dtype", "b_dtype", "c_dtype",
    "compute_type", "scale_type",
)


def aggregate_shapes(calls: list[dict]) -> list[dict]:
    """Dedupe on SHAPE_KEY_FIELDS; aggregate call_count + algo histogram.

    Calls with any None / missing shape-key field (e.g. parse-degraded records)
    are bucketed into a single 'unknown' shape so they're still counted but
    don't generate a million degenerate rows."""
    buckets: dict[tuple, dict] = defaultdict(lambda: {
        "call_count": 0,
        "picked_algos": defaultdict(int),
        "first_seen_call_idx": -1,
    })
    for c in calls:
        try:
            key = tuple(c[k] for k in SHAPE_KEY_FIELDS)
            if any(v is None for v in key):
                key = ("__degraded__",)
        except KeyError:
            key = ("__degraded__",)
        b = buckets[key]
        b["call_count"] += 1
        if c.get("algo_id") is not None:
            b["picked_algos"][str(c["algo_id"])] += 1
        if b["first_seen_call_idx"] == -1:
            b["first_seen_call_idx"] = c.get("call_idx", -1)

    shapes = []
    for key, agg in buckets.items():
        if key == ("__degraded__",):
            d = {k: None for k in SHAPE_KEY_FIELDS}
            d["__degraded__"] = True
        else:
            d = dict(zip(SHAPE_KEY_FIELDS, key))
        d["call_count"] = agg["call_count"]
        d["picked_algos"] = dict(agg["picked_algos"])
        d["first_seen_call_idx"] = agg["first_seen_call_idx"]
        shapes.append(d)
    shapes.sort(key=lambda s: -s["call_count"])
    for i, s in enumerate(shapes):
        s["rank_by_calls"] = i + 1
    return shapes


def parse_log(log_path: Path) -> tuple[list[dict], list[dict], int]:
    """Stream the log; return (calls, unparsed_matmul_lines, total_lines).

    unparsed_matmul_lines captures any non-empty line that mentions 'Matmul'
    but didn't match the expected header regex — useful for catching log
    format drift in future cuBLAS-Lt versions. Each unparsed entry includes
    a log pointer so AVO can `sed -n '<line_no>p' <log>` to investigate.
    """
    calls: list[dict] = []
    unparsed: list[dict] = []
    call_idx = 0
    total = 0
    byte_offset = 0
    log_basename = log_path.name
    with log_path.open("rb") as f:
        for line_no, raw in enumerate(f, start=1):
            total += 1
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                rec = parse_line(line, call_idx, byte_offset, line_no, log_basename)
                if rec is not None:
                    calls.append(rec)
                    call_idx += 1
                elif "Matmul" in line:
                    unparsed.append({
                        "log": log_basename,
                        "line_no": line_no,
                        "byte_offset": byte_offset,
                        "excerpt": line[:400] + ("...[truncated]" if len(line) > 400 else ""),
                    })
            byte_offset += len(raw)
    return calls, unparsed, total


PARSER_VERSION = "trtllm-bench-cublaslt-parser/1"


def write_outputs(run_dir: Path, calls: list[dict], unparsed: list[dict],
                  header: dict) -> None:
    calls_path = run_dir / "cublaslt_calls.jsonl"
    with calls_path.open("w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")
    shapes = aggregate_shapes(calls)
    degraded = sum(1 for c in calls if "_parse_warnings" in c)
    shapes_blob = dict(header)
    shapes_blob["parser_version"] = PARSER_VERSION
    shapes_blob["shapes"] = shapes
    shapes_blob["total_calls"] = len(calls)
    shapes_blob["unparsed_matmul_lines"] = len(unparsed)
    shapes_blob["degraded_calls"] = degraded
    if degraded or unparsed:
        # Tell the agent exactly where to look. cublaslt_calls.jsonl carries
        # `_log_pointer` on each degraded record; cublaslt_unparsed.jsonl
        # carries one entry per unparsed line. The raw log itself is right
        # next to the parser outputs.
        shapes_blob["inspect_hint"] = (
            f"Format drift detected. {degraded} call(s) parsed in degraded "
            f"mode (see _log_pointer fields in cublaslt_calls.jsonl); "
            f"{len(unparsed)} matmul-mentioning line(s) failed to match the "
            f"header regex (see cublaslt_unparsed.jsonl). Each entry carries "
            f"log/line_no/byte_offset/excerpt — open cublasLt.log alongside "
            f"the run dir and inspect the cited line(s) directly."
        )
    (run_dir / "cublaslt_shapes.json").write_text(
        json.dumps(shapes_blob, indent=2) + "\n"
    )
    if unparsed:
        # Surface format drift loudly — these are lines that mention Matmul
        # but didn't match the parser. Worth eyeballing before trusting the
        # rest of the run.
        (run_dir / "cublaslt_unparsed.jsonl").write_text(
            "\n".join(json.dumps(u) for u in unparsed) + "\n"
        )


def build_header(run_dir: Path, model: str | None) -> dict:
    h: dict = {"model": model or ""}
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        try:
            mj = json.loads(manifest.read_text())
            if not h["model"]:
                h["model"] = (mj.get("model") or {}).get("name", "")
            h["trtllm_git_sha"] = (mj.get("git") or {}).get("trtllm_sha", "")
            h["b12x_git_sha"] = (mj.get("git") or {}).get("b12x_sha", "")
            h["device"] = (mj.get("env") or {}).get("gpu", "")
            h["run_id"] = mj.get("run_id", run_dir.name)
        except json.JSONDecodeError:
            pass
    h.setdefault("device", "")
    h.setdefault("compute_capability", "")
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--log", help="explicit cublasLt.log path (overrides run-dir/cublasLt.log)")
    ap.add_argument("--model", help="model name (forwarded into shapes header)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.run_dir:
        print("--run-dir required (or --self-test)", file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"run-dir not found: {run_dir}", file=sys.stderr)
        return 2

    log_path = Path(args.log) if args.log else (run_dir / "cublasLt.log")
    if not log_path.is_file():
        print(f"cublasLt.log not found: {log_path}", file=sys.stderr)
        return 2

    calls, unparsed, total_lines = parse_log(log_path)
    print(f"[parse_cublaslt_log] {total_lines} lines, {len(calls)} matmul calls, "
          f"{len(unparsed)} unparsed matmul lines",
          file=sys.stderr)
    header = build_header(run_dir, args.model)
    write_outputs(run_dir, calls, unparsed, header)
    print(f"[parse_cublaslt_log] wrote {run_dir/'cublaslt_calls.jsonl'} "
          f"and {run_dir/'cublaslt_shapes.json'}"
          + (f" + {run_dir/'cublaslt_unparsed.jsonl'}" if unparsed else ""),
          file=sys.stderr)
    return 0


SELF_TEST_FIXTURE = """\
[2026-05-04 15:08:47][cublasLt][53055][Trace][cublasLtTSTMatmul] A=0X32F600000 \
Adesc=[type=R_16BF rows=2048 cols=1024 ld=2048] B=0X32EE00000 \
Bdesc=[type=R_16BF rows=1024 cols=32 ld=1024] C=0X32EE10000 \
Cdesc=[type=R_16BF rows=2048 cols=32 ld=2048] D=0X32EE10000 \
Ddesc=[type=R_16BF rows=2048 cols=32 ld=2048] \
computeDesc=[computeType=COMPUTE_32F scaleType=R_32F smCountTarget=48] \
algo=[algoId=67 tile=MATMUL_TILE_64x32 stages=MATMUL_STAGES_64xAUTO customOption=29] \
workSpace=0X0 workSpaceSizeInBytes=0 beta=0 outOfPlace=0 stream=0X0
[2026-05-04 15:08:47][cublasLt][53055][Trace][cublasLtMatmul] A=0X330220000 \
Adesc=[type=R_8F_E4M3 rows=1024 cols=2048 ld=1024] B=0X32EE50000 \
Bdesc=[type=R_8F_E4M3 rows=1024 cols=32 ld=1024] C=0X330220000 \
Cdesc=[type=R_16BF rows=2048 cols=32 ld=2048] D=0X32EE58000 \
Ddesc=[type=R_16BF rows=2048 cols=32 ld=2048] \
computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T \
aScalePointer=0x32ee30200 bScalePointer=0x32ee30000] \
algo=[algoId=67 tile=MATMUL_TILE_32x32 stages=MATMUL_STAGES_128xAUTO customOption=6] \
workSpace=0X32EE78000 workSpaceSizeInBytes=1048576 beta=0 outOfPlace=1 stream=0X0
[2027-01-01 00:00:00][cublasLt][99999][Trace][cublasLtMatmul] A=0X1 \
Adesc=[type=R_4F_E2M1 rows=512 cols=128 ld=512 sfBlockSize=32] \
Bdesc=[type=R_4F_E2M1 rows=128 cols=64 ld=128 sfBlockSize=32] \
Cdesc=[type=R_16BF rows=512 cols=64 ld=512] \
Ddesc=[type=R_16BF rows=512 cols=64 ld=512] \
computeDesc=[computeType=COMPUTE_32F scaleType=R_32F newFutureField=42 \
microscalingMode=NVFP4 aScalePointer=0xabc bScalePointer=0xdef] \
algo=[algoId=200 tile=MATMUL_TILE_128x128 stages=MATMUL_STAGES_2xAUTO \
customOption=0 unicornFlag=true] \
workSpace=0X100 workSpaceSizeInBytes=2048 beta=0 outOfPlace=1 stream=0X0 myFutureTopAttr=hi
"""


def self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        log = d / "cublasLt.log"
        log.write_text(SELF_TEST_FIXTURE)
        calls, unparsed, _ = parse_log(log)
        assert len(calls) == 3, calls
        assert len(unparsed) == 0, unparsed
        # Line 1: BF16 TST, transa default N → M=2048 K=1024 N=32
        c0 = calls[0]
        assert c0["fn"] == "cublasLtTSTMatmul"
        assert (c0["M"], c0["K"], c0["N"]) == (2048, 1024, 32), c0
        assert c0["a_dtype"] == "bf16" and c0["b_dtype"] == "bf16" and c0["c_dtype"] == "bf16"
        assert c0["compute_type"] == "fp32"
        assert c0["trans_a"] == "N" and c0["trans_b"] == "N"
        assert c0["algo_id"] == 67 and c0["tile"] == "64x32"
        assert "extras" not in c0  # all fields known
        # Line 2: FP8 transa=OP_T → A in memory (K=1024, M=2048), so K=1024, M=2048
        c1 = calls[1]
        assert c1["fn"] == "cublasLtMatmul"
        assert (c1["M"], c1["K"], c1["N"]) == (2048, 1024, 32), c1
        assert c1["a_dtype"] == "e4m3" and c1["c_dtype"] == "bf16"
        assert c1["trans_a"] == "T" and c1["trans_b"] == "N"
        assert c1["has_a_scale"] is True and c1["has_b_scale"] is True
        assert c1["workspace_bytes"] == 1048576
        # Line 3: hypothetical future NVFP4 record with unknown fields
        c2 = calls[2]
        assert c2["a_dtype"] == "fp4_e2m1"
        assert c2["algo_id"] == 200
        assert "extras" in c2
        ex = c2["extras"]
        # Each unknown field is bucketed by its container
        assert ex.get("Adesc", {}).get("sfBlockSize") == "32"
        assert ex.get("Bdesc", {}).get("sfBlockSize") == "32"
        assert ex.get("computeDesc", {}).get("newFutureField") == "42"
        assert ex.get("computeDesc", {}).get("microscalingMode") == "NVFP4"
        assert ex.get("algo", {}).get("unicornFlag") == "true"
        assert ex.get("top", {}).get("myFutureTopAttr") == "hi"

        shapes = aggregate_shapes(calls)
        assert len(shapes) == 3, shapes
        for s in shapes:
            assert s["call_count"] == 1
        print("[parse_cublaslt_log] self-test OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
