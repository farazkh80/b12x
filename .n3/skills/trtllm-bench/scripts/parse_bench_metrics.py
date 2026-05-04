#!/usr/bin/env python3
"""Parse trtllm-bench artifacts in a run-dir into metrics.json + summary.md.

Lenient by design — schema drift in trtllm-bench's --report_json never makes
this parser fail. Instead it:
  - probes multiple known paths for each metric (first hit wins)
  - falls back to stdout regex when JSON is missing
  - records `_parser_warnings` + `inspect_hint` in metrics.json whenever it
    couldn't find a known field, surfacing exactly which file to read next
  - lists any unrecognized top-level keys so a new release that adds a
    section like `kv_cache_metrics` is visible without reading the raw JSON


Reads (in order of preference):
  - bench_report.json   primary structured source emitted by trtllm-bench
  - bench_iter.log      per-iteration log (optional, for spot-checks)
  - bench.stdout        fallback for fields not in the JSON
  - manifest.json       run metadata
  - cublaslt_shapes.json   to surface top GEMM shapes in summary

Outputs:
  - metrics.json   normalized schema (see trtllm-bench SKILL.md)
  - summary.md     human-readable digest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# trtllm-bench --report_json schema (verified on 1.3.0rc12):
#   {
#     "engine":       {model, model_path, version, backend, dtype, kv_cache_dtype, quantization, ...},
#     "machine":      {name, "memory.total", ...},
#     "world_info":   {tp_size, max_batch_size, max_num_tokens, kv_cache_percentage, ...},
#     "request_info": {num_requests, avg_input_length, avg_output_length, ...},
#     "performance":  {total_latency_ms, request_throughput_req_s,
#                      system_total_throughput_tok_s, system_output_throughput_tok_s,
#                      output_throughput_per_user_tok_s, output_throughput_per_gpu_tok_s,
#                      request_latency_percentiles_ms: {p50,p90,p95,p99,minimum,maximum,average}},
#     "energy":       {total_energy_j, output_tps_per_w, average_gpu_power},
#     "streaming_metrics": {token_output_speed_tok_s, avg_ttft_ms, avg_tpot_ms,
#                           ttft_percentiles, tpot_percentiles, gen_tps_percentiles},
#     "dataset":      {avg_isl, avg_osl, num_requests, dataset_path, max_isl, max_osl, ...},
#   }
# Older legacy paths kept as backups (first hit wins).

# (deep-key-path, output-key) — first hit wins.
THROUGHPUT_PROBES = [
    # Current 1.3.x schema
    (("performance", "system_total_throughput_tok_s"), "tokens_per_second"),
    (("performance", "system_output_throughput_tok_s"), "output_tokens_per_second"),
    (("performance", "request_throughput_req_s"), "requests_per_second"),
    (("performance", "output_throughput_per_user_tok_s"), "output_tokens_per_second_per_user"),
    (("performance", "output_throughput_per_gpu_tok_s"), "output_tokens_per_second_per_gpu"),
    # Legacy / earlier rc paths
    (("performance", "tokens_per_second"), "tokens_per_second"),
    (("throughput", "total_token_throughput"), "tokens_per_second"),
    (("performance", "output_tokens_per_second"), "output_tokens_per_second"),
    (("throughput", "output_token_throughput"), "output_tokens_per_second"),
    (("performance", "input_tokens_per_second"), "input_tokens_per_second"),
    (("throughput", "input_token_throughput"), "input_tokens_per_second"),
    (("performance", "requests_per_second"), "requests_per_second"),
    (("throughput", "request_throughput"), "requests_per_second"),
]

LATENCY_PROBES = [
    # TTFT / TPOT live under streaming_metrics in 1.3.x
    (("streaming_metrics", "ttft_percentiles", "p50"), "ttft_p50"),
    (("streaming_metrics", "ttft_percentiles", "p90"), "ttft_p90"),
    (("streaming_metrics", "ttft_percentiles", "p95"), "ttft_p95"),
    (("streaming_metrics", "ttft_percentiles", "p99"), "ttft_p99"),
    (("streaming_metrics", "tpot_percentiles", "p50"), "tpot_p50"),
    (("streaming_metrics", "tpot_percentiles", "p90"), "tpot_p90"),
    (("streaming_metrics", "tpot_percentiles", "p95"), "tpot_p95"),
    (("streaming_metrics", "tpot_percentiles", "p99"), "tpot_p99"),
    (("streaming_metrics", "avg_ttft_ms"), "ttft_avg"),
    (("streaming_metrics", "avg_tpot_ms"), "tpot_avg"),
    # E2E / request-latency lives under performance
    (("performance", "request_latency_percentiles_ms", "p50"), "e2e_p50"),
    (("performance", "request_latency_percentiles_ms", "p90"), "e2e_p90"),
    (("performance", "request_latency_percentiles_ms", "p95"), "e2e_p95"),
    (("performance", "request_latency_percentiles_ms", "p99"), "e2e_p99"),
    (("performance", "avg_request_latency_ms"), "e2e_avg"),
    (("performance", "total_latency_ms"), "total_latency_ms"),
    # Legacy
    (("latency_ms", "ttft_p50"), "ttft_p50"),
    (("latency_ms", "ttft_p95"), "ttft_p95"),
    (("latency_ms", "tpot_p50"), "tpot_p50"),
    (("latency_ms", "tpot_p95"), "tpot_p95"),
    (("latency_ms", "e2e_p50"), "e2e_p50"),
    (("latency_ms", "e2e_p95"), "e2e_p95"),
]

TOKEN_PROBES = [
    (("request_info", "num_requests"), "num_requests"),
    (("request_info", "avg_input_length"), "avg_input_length"),
    (("request_info", "avg_output_length"), "avg_output_length"),
    (("request_info", "avg_num_concurrent_requests"), "avg_concurrent_requests"),
    (("dataset", "num_requests"), "num_requests"),
    (("dataset", "avg_isl"), "avg_input_length"),
    (("dataset", "avg_osl"), "avg_output_length"),
    (("dataset", "max_isl"), "max_input_length"),
    (("dataset", "max_osl"), "max_output_length"),
]

ENERGY_PROBES = [
    (("energy", "total_energy_j"), "total_energy_j"),
    (("energy", "output_tps_per_w"), "output_tps_per_w"),
    (("energy", "average_gpu_power"), "average_gpu_power_w"),
]

PARSER_VERSION = "trtllm-bench-metrics-parser/1"


def deep_get(obj: Any, path: tuple) -> Any:
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def collect(report: dict, probes: list) -> dict:
    out: dict = {}
    for path, out_key in probes:
        if out_key in out:
            continue  # first hit wins
        v = deep_get(report, path)
        if isinstance(v, (int, float)):
            out[out_key] = v
    return out


# stdout regex fallbacks — only used when the JSON is missing or sparse.
STDOUT_PATTERNS = {
    "tokens_per_second": [
        re.compile(r"Total\s+Token\s+Throughput[^0-9]*([0-9.]+)", re.I),
        re.compile(r"Tokens?/sec[^0-9]*([0-9.]+)", re.I),
    ],
    "output_tokens_per_second": [
        re.compile(r"Output\s+Token\s+Throughput[^0-9]*([0-9.]+)", re.I),
    ],
    "requests_per_second": [
        re.compile(r"Request\s+Throughput[^0-9]*([0-9.]+)", re.I),
    ],
    "ttft_p50": [re.compile(r"TTFT.*p50[^0-9]*([0-9.]+)", re.I)],
    "ttft_p95": [re.compile(r"TTFT.*p95[^0-9]*([0-9.]+)", re.I)],
    "tpot_p50": [re.compile(r"TPOT.*p50[^0-9]*([0-9.]+)", re.I)],
    "tpot_p95": [re.compile(r"TPOT.*p95[^0-9]*([0-9.]+)", re.I)],
    "e2e_p50": [re.compile(r"E2E.*p50[^0-9]*([0-9.]+)", re.I)],
    "e2e_p95": [re.compile(r"E2E.*p95[^0-9]*([0-9.]+)", re.I)],
}


def fill_from_stdout(stdout: str, target: dict, keys: list[str]) -> None:
    for k in keys:
        if k in target and target[k] is not None:
            continue
        for pat in STDOUT_PATTERNS.get(k, []):
            m = pat.search(stdout)
            if m:
                try:
                    target[k] = float(m.group(1))
                    break
                except ValueError:
                    pass


def parse(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "bench_report.json"
    stdout_path = run_dir / "bench.stdout"

    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            # Don't fail the whole parse if manifest got corrupted; just log
            # and proceed without it (bench_report.json is the real source).
            print(f"[parse_bench_metrics] WARN: manifest.json malformed ({e}); "
                  f"continuing without it", file=sys.stderr)

    report: dict = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            # trtllm-bench occasionally emits NaN; tolerate it
            txt = report_path.read_text().replace("NaN", "null")
            report = json.loads(txt)

    stdout_txt = stdout_path.read_text() if stdout_path.is_file() else ""

    throughput = collect(report, THROUGHPUT_PROBES)
    latency = collect(report, LATENCY_PROBES)
    tokens = collect(report, TOKEN_PROBES)
    energy = collect(report, ENERGY_PROBES)

    # stdout fallbacks (latency in ms; throughput in tok/s)
    fill_from_stdout(stdout_txt, throughput,
                     ["tokens_per_second", "output_tokens_per_second",
                      "requests_per_second"])
    fill_from_stdout(stdout_txt, latency,
                     ["ttft_p50", "ttft_p95", "tpot_p50",
                      "tpot_p95", "e2e_p50", "e2e_p95"])

    # Surface the engine block (model_path, version, dtype, quantization, ...)
    # since it's useful in summary.md.
    engine = {}
    if isinstance(report.get("engine"), dict):
        for k in ("model", "model_path", "version", "backend",
                  "dtype", "kv_cache_dtype", "quantization"):
            v = report["engine"].get(k)
            if v is not None:
                engine[k] = v

    # Surface schema drift loudly: any top-level keys we didn't probe end up
    # here so AVO knows there's new info worth inspecting in the raw report.
    KNOWN_REPORT_TOP = {
        "engine", "machine", "world_info", "request_info",
        "performance", "energy", "streaming_metrics", "dataset",
    }
    unrecognized_top = sorted(set(report.keys()) - KNOWN_REPORT_TOP) if report else []

    parser_warnings: list[str] = []
    inspect_hints: list[str] = []
    if not throughput:
        parser_warnings.append("no_throughput_metrics_found")
        inspect_hints.append("throughput probes missed all known paths in performance.* "
                              "and throughput.* — open bench_report.json and look for new "
                              "tok/s field names")
    if not latency:
        parser_warnings.append("no_latency_metrics_found")
        inspect_hints.append("latency probes missed all known paths under streaming_metrics "
                              "and performance.request_latency_percentiles_ms — open "
                              "bench_report.json")
    if unrecognized_top:
        parser_warnings.append(f"unrecognized_top_level_keys: {','.join(unrecognized_top)}")
        inspect_hints.append(
            f"bench_report.json contains top-level keys not handled by this parser: "
            f"{', '.join(unrecognized_top)} — read those sections of bench_report.json "
            f"and consider extending {Path(__file__).name}"
        )

    metrics = {
        "parser_version": PARSER_VERSION,
        "run_id": manifest.get("run_id", run_dir.name),
        "model": (manifest.get("model") or {}).get("name", "") or engine.get("model", ""),
        "config": manifest.get("config", {}),
        "engine": engine,
        "throughput": throughput,
        "latency_ms": latency,
        "tokens": tokens,
        "energy": energy,
        "raw_report_path": "bench_report.json" if report_path.is_file() else None,
        "raw_stdout_path": "bench.stdout" if stdout_path.is_file() else None,
        "raw_iter_log_path": "bench_iter.log" if (run_dir / "bench_iter.log").is_file() else None,
    }
    if parser_warnings:
        metrics["_parser_warnings"] = parser_warnings
        metrics["inspect_hint"] = " | ".join(inspect_hints) if inspect_hints else None
    return metrics


def render_summary(metrics: dict, run_dir: Path) -> str:
    lines: list[str] = []
    lines.append(f"# trtllm-bench summary — {metrics['run_id']}")
    lines.append("")
    lines.append(f"**Model:** `{metrics.get('model', '')}`")
    cfg = metrics.get("config", {}) or {}
    if cfg:
        cfg_bits = [f"{k}={v}" for k, v in cfg.items() if v is not None]
        lines.append(f"**Config:** {', '.join(cfg_bits)}")
    lines.append("")

    eng = metrics.get("engine", {}) or {}
    if eng:
        bits = [f"{k}={v}" for k, v in eng.items() if v]
        lines.append(f"**Engine:** {', '.join(bits)}")
        lines.append("")

    th = metrics.get("throughput", {}) or {}
    if th:
        lines.append("## Throughput")
        for k in ("tokens_per_second", "output_tokens_per_second",
                 "input_tokens_per_second", "requests_per_second",
                 "output_tokens_per_second_per_user",
                 "output_tokens_per_second_per_gpu"):
            if k in th:
                lines.append(f"- `{k}`: **{th[k]:.2f}**")
        lines.append("")

    lat = metrics.get("latency_ms", {}) or {}
    if lat:
        lines.append("## Latency (ms)")
        for k in ("ttft_avg", "ttft_p50", "ttft_p90", "ttft_p95", "ttft_p99",
                 "tpot_avg", "tpot_p50", "tpot_p90", "tpot_p95", "tpot_p99",
                 "e2e_avg", "e2e_p50", "e2e_p90", "e2e_p95", "e2e_p99",
                 "total_latency_ms"):
            if k in lat:
                lines.append(f"- `{k}`: **{lat[k]:.2f}**")
        lines.append("")

    tok = metrics.get("tokens", {}) or {}
    if tok:
        lines.append("## Tokens / dataset")
        for k, v in tok.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    en = metrics.get("energy", {}) or {}
    if en:
        lines.append("## Energy")
        for k, v in en.items():
            try:
                lines.append(f"- `{k}`: **{float(v):.3f}**")
            except (TypeError, ValueError):
                lines.append(f"- `{k}`: {v}")
        lines.append("")

    # Top GEMM shapes if cublaslt_shapes.json exists
    shapes_path = run_dir / "cublaslt_shapes.json"
    if shapes_path.is_file():
        try:
            shapes_blob = json.loads(shapes_path.read_text())
            top = (shapes_blob.get("shapes") or [])[:10]
            if top:
                lines.append("## Top GEMM shapes (cuBLAS-Lt)")
                lines.append("")
                lines.append("| rank | M | K | N | trans | a/b/c | calls | algo (top) |")
                lines.append("|---:|---:|---:|---:|:--:|:--:|---:|:--|")
                for s in top:
                    algos = s.get("picked_algos") or {}
                    top_algo = max(algos.items(), key=lambda kv: kv[1])[0] if algos else "-"
                    lines.append(
                        f"| {s.get('rank_by_calls','?')} "
                        f"| {s.get('M','?')} "
                        f"| {s.get('K','?')} "
                        f"| {s.get('N','?')} "
                        f"| {s.get('trans_a','?')}{s.get('trans_b','?')} "
                        f"| {s.get('a_dtype','?')}/{s.get('b_dtype','?')}/{s.get('c_dtype','?')} "
                        f"| {s.get('call_count','?')} "
                        f"| {top_algo} |"
                    )
                lines.append("")
        except Exception as e:  # noqa: BLE001
            lines.append(f"_(failed to parse cublaslt_shapes.json: {e})_")
            lines.append("")

    lines.append(f"**Run dir:** `{run_dir}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=False)
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

    metrics = parse(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (run_dir / "summary.md").write_text(render_summary(metrics, run_dir))
    print(f"[parse_bench_metrics] wrote {run_dir/'metrics.json'} and {run_dir/'summary.md'}",
          file=sys.stderr)
    return 0


def self_test() -> int:
    import tempfile

    # Fixture mirrors the real 1.3.0rc12 bench_report.json schema observed
    # on a Nemotron-Nano-3 NVFP4 run.
    fixture_report = {
        "engine": {
            "model": "selftest/model",
            "model_path": "/path/to/model",
            "version": "1.2",
            "backend": "Pytorch",
            "dtype": "bfloat16",
            "kv_cache_dtype": "FP8",
            "quantization": "NVFP4",
        },
        "machine": {"name": "NVIDIA GB10", "memory.total": 119.7},
        "world_info": {"tp_size": 1, "max_batch_size": 1, "max_num_tokens": 512},
        "request_info": {"num_requests": 1, "avg_input_length": 64.0,
                         "avg_output_length": 16.0,
                         "avg_num_concurrent_requests": 1.0},
        "performance": {
            "total_latency_ms": 4743.5,
            "avg_request_latency_ms": 4743.5,
            "request_throughput_req_s": 0.21,
            "system_output_throughput_tok_s": 3.37,
            "system_total_throughput_tok_s": 16.86,
            "output_throughput_per_user_tok_s": 3.37,
            "output_throughput_per_gpu_tok_s": 3.37,
            "request_latency_percentiles_ms": {
                "p50": 4743.5, "p90": 4743.5, "p95": 4743.5, "p99": 4743.5,
            },
        },
        "energy": {"total_energy_j": 56.1, "output_tps_per_w": 0.285,
                   "average_gpu_power": 11.83},
        "streaming_metrics": {
            "token_output_speed_tok_s": 71.29,
            "avg_ttft_ms": 4533.1,
            "avg_tpot_ms": 14.03,
            "tpot_percentiles": {"p50": 14.03, "p90": 14.03, "p95": 14.03, "p99": 14.03},
            "ttft_percentiles": {"p50": 4533.1, "p90": 4533.1, "p95": 4533.1, "p99": 4533.1},
            "gen_tps_percentiles": {"p50": 71.29},
        },
        "dataset": {"avg_isl": 64.0, "avg_osl": 16.0, "num_requests": 1,
                    "max_isl": 64, "max_osl": 16},
    }
    fixture_manifest = {
        "run_id": "selftest",
        "model": {"name": "selftest/model"},
        "config": {"concurrency": 1, "max_batch_size": 1, "tp": 1, "backend": "pytorch"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "bench_report.json").write_text(json.dumps(fixture_report))
        (d / "manifest.json").write_text(json.dumps(fixture_manifest))
        (d / "bench.stdout").write_text("")
        m = parse(d)
        assert m["throughput"]["tokens_per_second"] == 16.86, m
        assert m["throughput"]["output_tokens_per_second"] == 3.37, m
        assert m["throughput"]["requests_per_second"] == 0.21, m
        assert m["latency_ms"]["ttft_p50"] == 4533.1, m
        assert m["latency_ms"]["tpot_p50"] == 14.03, m
        assert m["latency_ms"]["e2e_p50"] == 4743.5, m
        assert m["latency_ms"]["ttft_avg"] == 4533.1, m
        assert m["tokens"]["num_requests"] == 1, m
        assert m["tokens"]["avg_input_length"] == 64.0, m
        assert m["energy"]["total_energy_j"] == 56.1, m
        assert m["engine"]["quantization"] == "NVFP4", m
        summary = render_summary(m, d)
        assert "tokens_per_second" in summary
        assert "Engine" in summary
        print("[parse_bench_metrics] self-test OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
