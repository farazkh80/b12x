---
name: trtllm-bench
description: >
  Run trtllm-bench (TRT-LLM throughput benchmark) on the Spark host with
  cuBLAS / cuBLAS-Lt logging enabled, then parse the artifacts into clean
  JSON: per-call cuBLAS-Lt GEMM history (consumable by cublas-gemm-tuning)
  and normalized bench metrics. Use whenever AVO needs to (a) produce a
  reproducible TRT-LLM throughput measurement on a model, (b) capture which
  GEMM shapes the model actually issues so the next iteration can tune
  them, or (c) compare two builds/configs by diffing the parsed JSON.
user-invocable: false
license: LicenseRef-NvidiaProprietary
metadata:
  author: AVO project
---

# trtllm-bench with cuBLAS-Lt collection

A single shell entrypoint (`scripts/run_bench.sh`) drives the full flow:

1. Resolve a unique run-id under `$WORKSPACE/.n3/runs/trtllm/<run_id>/`.
2. Optionally regenerate a dataset via `prepare_dataset.py`.
3. Run `trtllm-bench throughput` with `CUBLAS_LOGINFO_DBG=1` and
   `CUBLASLT_LOG_LEVEL=2` so every cuBLAS / cuBLAS-Lt API call lands in a
   per-run log file (not a global tmp file).
4. Hand the artifacts to two parsers:
   - `parse_bench_metrics.py` → `metrics.json` + `summary.md`
   - `parse_cublaslt_log.py`  → `cublaslt_calls.jsonl` + `cublaslt_shapes.json`

All GPU work is wrapped via `.claude_docs/scripts/command_on_spark.sh` so this
skill never assumes a local GPU.

## When to use

Trigger on these intents:
- "benchmark `<model>` with trtllm-bench"
- "what cuBLAS-Lt shapes does `<model>` actually call"
- "compare bench numbers for `<model>` between b12x on/off"
- "produce a tuning input for cublas-gemm-tuning from a real bench run"

Do **not** trigger for:
- `aiperf` client benchmarks against a running `trtllm-serve` — use
  `aiperf_*.sh` scripts in `TensorRT-LLM/.claude_docs/nemo-claw-super-fp4/`.
- Multi-config sweeps — drive this skill in a loop instead.

## Quick recipe

```bash
# Required env
export B12X_TRTLLM_REPO=/home/farazkh_scratch/agentic-dev/avo-b12x-trt/TensorRT-LLM
# Optional env (defaults shown)
export B12X_RUN_ROOT=/home/farazkh_scratch/agentic-dev/avo-b12x-trt/.n3/runs/trtllm

# Smoke run on Nemotron Nano 3 NVFP4 (model staged on NFS, no HF download).
.claude_docs/scripts/command_on_spark.sh \
  b12x/.n3/skills/trtllm-bench/scripts/run_bench.sh \
    --label nemotron_nano3_smoke \
    --model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --model-path "$B12X_TRTLLM_REPO/models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" \
    --num-requests 2 --input-mean 512 --output-mean 128 \
    --concurrency 1 --max-batch-size 1 --max-num-tokens 2048
```

Run from the **AVO host** (the script forwards to Spark itself if invoked
as `command_on_spark.sh ...`). Or run directly inside the spark container if
you started a shell there with `command_on_spark.sh --shell`.

## Inputs (`run_bench.sh` flags)

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--label <s>` | yes | — | Short identifier baked into the run-id, e.g. `nemotron_nano3_2k1k_b12x_off`. |
| `--model-name <hf-id>` | yes | — | Used as `--model` arg to `trtllm-bench` and as the tokenizer for `prepare_dataset.py` if no `--dataset` is given. |
| `--model-path <p>` | no | (skip flag) | If set, passes `--model_path` to `trtllm-bench` (offline weights). |
| `--dataset <p>` | no | regenerate | Path to a token-norm-dist `.jsonl`. If omitted, the skill calls `prepare_dataset.py` and writes `<run_dir>/dataset.jsonl`. |
| `--num-requests N` | no | `2` | For dataset prep AND for `trtllm-bench --num_requests`. |
| `--input-mean N` | no | `2048` | `prepare_dataset.py --input-mean`. |
| `--output-mean N` | no | `1024` | `prepare_dataset.py --output-mean`. |
| `--concurrency N` | no | `1` | `trtllm-bench --concurrency`. |
| `--max-batch-size N` | no | `1` | |
| `--max-num-tokens N` | no | `auto` (= input-mean+output-mean+128) | |
| `--tp N` | no | `1` | Tensor parallel. |
| `--backend <s>` | no | `pytorch` | |
| `--extra-llm-api-options <p>` | no | template default | Path to a yaml. If not given, the skill copies `templates/extra-llm-api-config.yml` into the run dir. |
| `--no-streaming` | no | — | Disable `--streaming` (default ON). |
| `--no-cublas-log` | no | — | Skip cuBLAS env vars (occasionally useful for clean throughput numbers without log overhead). |
| `--warmup N` | no | `0` | |
| `--extra-bench-args "<...>"` | no | — | Free-form extra args appended to `trtllm-bench`. |

The script echoes the resolved `run_dir` on stdout (last line) so callers can
capture it.

## Run-dir layout

```
$WORKSPACE/.n3/runs/trtllm/<UTC>-<label>/
  manifest.json              # env + args + git SHAs + GPU info
  extra-llm-api-config.yml   # whatever yaml was used (defaulted or supplied)
  dataset.jsonl              # input dataset (regenerated or copied)
  bench.stdout               # full stdout of trtllm-bench
  bench.stderr               # full stderr (PyTorch warnings, NVML, etc.)
  bench_report.json          # native trtllm-bench --report_json output
  bench_iter.log             # native trtllm-bench --iteration_log output
  cublas.log                 # raw classic-cuBLAS API log (kept verbatim)
  cublasLt.log               # raw cuBLAS-Lt API log (kept verbatim)
  cublaslt_calls.jsonl       # one JSON per matmul call (parser output)
  cublaslt_shapes.json       # deduplicated shapes — input format for cublas-gemm-tuning
  metrics.json               # normalized bench metrics (parser output)
  summary.md                 # human-readable digest of metrics + top GEMM shapes
```

Run-id is `<UTC>-<label>`, e.g. `2026-05-04T18-30-12Z-nemotron_nano3_smoke`.

## Parsers

### `scripts/parse_bench_metrics.py`

Inputs: `bench_report.json` (primary), `bench_iter.log` (per-iter, optional),
`bench.stdout` (fallback for fields trtllm-bench might not put in JSON).

Output (`metrics.json`):

```json
{
  "run_id": "...",
  "model": "...",
  "config": {"concurrency": 1, "max_batch_size": 1, "max_num_tokens": 2048,
             "tp": 1, "backend": "pytorch", "streaming": true},
  "throughput": {
    "tokens_per_second": 123.4,
    "requests_per_second": 0.5,
    "input_tokens_per_second": 1024.0,
    "output_tokens_per_second": 256.0
  },
  "latency_ms": {
    "ttft_p50": 1234.5, "ttft_p95": 1500.2,
    "tpot_p50": 12.3,   "tpot_p95": 18.7,
    "e2e_p50": 5678.9,  "e2e_p95": 6100.1
  },
  "tokens": {"input_total": 1024, "output_total": 256, "num_requests": 2},
  "raw_report_path": "bench_report.json"
}
```

Run standalone:

```bash
python3 scripts/parse_bench_metrics.py --run-dir <run_dir>
```

### `scripts/parse_cublaslt_log.py`

Inputs: `cublasLt.log` (primary), optional `cublas.log`.

Output 1 — `cublaslt_calls.jsonl` (one JSON per matmul call, in order):

```json
{"call_idx": 0, "ts_ns": 1714824601234567890,
 "M": 32, "K": 5376, "N": 5376,
 "trans_a": "N", "trans_b": "T",
 "a_dtype": "e4m3", "b_dtype": "e4m3", "c_dtype": "bf16",
 "compute_type": "fp32", "scale_type": "fp32",
 "lda": 32, "ldb": 5376, "ldc": 32,
 "algo_id": 25, "tile_id": 27, "stages_id": 12,
 "split_k": 1, "swizzle": 0, "reduction_scheme": 0,
 "workspace_bytes": 33554432,
 "raw_log_offset": 12873}
```

Output 2 — `cublaslt_shapes.json` (deduplicated, sorted by call_count desc):

```json
{
  "device": "NVIDIA GB10",
  "compute_capability": "12.1",
  "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
  "trtllm_git_sha": "...",
  "shapes": [
    {"M": 32, "K": 5376, "N": 5376,
     "trans_a": "N", "trans_b": "T",
     "a_dtype": "e4m3", "b_dtype": "e4m3", "c_dtype": "bf16",
     "compute_type": "fp32", "scale_type": "fp32",
     "call_count": 1280,
     "picked_algos": {"25": 1280},
     "first_seen_call_idx": 17,
     "rank_by_calls": 1}
  ]
}
```

The `shapes[]` element schema is **the input format for the
`cublas-gemm-tuning` skill's `tune_gemm` binary** (same field names, same
dtype encoding). A future `cublas-gemm-tune-from-bench` skill can iterate
`shapes[]` and call `tune_gemm` per entry.

Run standalone:

```bash
python3 scripts/parse_cublaslt_log.py --run-dir <run_dir>
```

Both parsers also accept `--self-test` to validate against an embedded
fixture without touching a real run.

## Hardware-routing (CRITICAL)

GB10 / Spark is the only host that can run trtllm-bench in this workspace.
Always invoke this skill via `command_on_spark.sh`:

```bash
.claude_docs/scripts/command_on_spark.sh \
  b12x/.n3/skills/trtllm-bench/scripts/run_bench.sh ...
```

The parsers are pure-Python (no GPU) and can run either on the Spark host or
on the AVO host — both see the same NFS-mounted run-dir.

## cuBLAS-Lt logging notes

- `CUBLAS_LOGINFO_DBG=1` enables the classic cuBLAS API log; we keep it for
  completeness even though the modern path is Lt.
- `CUBLASLT_LOG_LEVEL=2` is the lowest verbosity that emits **per-call
  matmul descriptors** (level 1 only logs library lifecycle).
- `CUBLAS_LOGDEST_DBG` and `CUBLASLT_LOG_FILE` are absolute paths so the
  driver writes directly to the run-dir. If they were relative, the driver
  resolves them against `$PWD` at the time of process start, which is
  fragile across `docker exec` environments.
- Logging adds 1-3 % overhead at concurrency 1 (driver-side string
  formatting, no syscall per call). Pass `--no-cublas-log` if you need
  uncontaminated throughput numbers; the skill still produces the bench
  artifacts.

## Common pitfalls

- **Model not found**: if `--model-path` is passed but the path doesn't exist
  inside the spark container, trtllm-bench falls back to HF download. Set
  `HF_HOME=$WORKSPACE/.n3/hf-cache` (already done by `command_on_spark.sh`)
  to keep downloads on NFS.
- **`max_num_tokens` too small**: the bench fails opaquely if the model's
  `max_seq_len` × concurrency exceeds `--max_num_tokens`. The script's
  default `auto` (= `input-mean + output-mean + 128`) handles concurrency 1
  fine but increase it for higher concurrency.
- **Dataset re-prep is silent**: if `--dataset` is omitted, the skill always
  regenerates from `prepare_dataset.py`. Two runs with the same `--label`
  share a run-id only if started in the same UTC second; otherwise their
  datasets differ. Pass an explicit `--dataset` for reproducible diffs.
- **cuBLAS-Lt log can be huge** (100s of MB on long bench runs). Parser
  streams the file line-by-line — don't try to slurp it.

## Files

| File | Purpose |
|---|---|
| `SKILL.md` | This file |
| `scripts/run_bench.sh` | Main entrypoint — args → dataset prep → trtllm-bench → parsers |
| `scripts/parse_bench_metrics.py` | bench artifacts → `metrics.json` + `summary.md` |
| `scripts/parse_cublaslt_log.py` | `cublasLt.log` → `cublaslt_calls.jsonl` + `cublaslt_shapes.json` |
| `templates/extra-llm-api-config.yml` | Default `extra_llm_api_options` yaml (block-reuse off, kv-frac 0.2) |
| `tests/cublaslt_fixture.log` | Tiny synthetic log for parser self-test |
