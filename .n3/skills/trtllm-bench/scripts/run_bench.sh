#!/usr/bin/env bash
# run_bench.sh — single entrypoint for the trtllm-bench AVO skill.
#
# Resolves a run dir under $B12X_RUN_ROOT, optionally regenerates a dataset,
# runs trtllm-bench with cuBLAS-Lt logging on, then invokes both parsers.
# Designed to be called via .claude_docs/scripts/command_on_spark.sh so all
# CUDA work lands on the Spark host.
set -euo pipefail

# ----- defaults --------------------------------------------------------------
LABEL=""
MODEL_NAME=""
MODEL_PATH=""
DATASET=""
NUM_REQUESTS=2
INPUT_MEAN=2048
OUTPUT_MEAN=1024
INPUT_STDEV=0
OUTPUT_STDEV=0
CONCURRENCY=1
MAX_BATCH_SIZE=1
MAX_NUM_TOKENS=""        # auto if empty
TP=1
BACKEND=pytorch
EXTRA_LLM_API_OPTIONS=""
STREAMING=1
CUBLAS_LOG=1
WARMUP=0
EXTRA_BENCH_ARGS=""

# Workspace conventions. Caller may override.
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DEFAULT="$(cd "${SKILL_DIR}/../../../.." && pwd)"
WORKSPACE="${B12X_WORKSPACE:-$WORKSPACE_DEFAULT}"
RUN_ROOT="${B12X_RUN_ROOT:-${WORKSPACE}/.n3/runs/trtllm}"
TRTLLM_REPO="${B12X_TRTLLM_REPO:-${WORKSPACE}/TensorRT-LLM}"

# ----- argparse --------------------------------------------------------------
usage() {
    sed -n '1,80p' "${BASH_SOURCE[0]}" | grep -E '^# ' | sed 's/^# //'
    cat <<EOF

Usage:
  run_bench.sh --label <s> --model-name <hf-id> [options]

Required:
  --label <s>           Short identifier baked into the run-id.
  --model-name <s>      HF id (used for --model and dataset tokenizer).

Common options:
  --model-path <p>      Local weights path (passed as --model_path).
  --dataset <p>         Existing token-norm-dist .jsonl. Skips prepare_dataset.
  --num-requests N      Default: 2.
  --input-mean N        Default: 2048.
  --output-mean N       Default: 1024.
  --concurrency N       Default: 1.
  --max-batch-size N    Default: 1.
  --max-num-tokens N    Default: auto (= input + output + 128).
  --tp N                Default: 1.
  --backend <s>         Default: pytorch.
  --extra-llm-api-options <p>   Default: templates/extra-llm-api-config.yml.
  --no-streaming        Disable --streaming (default ON).
  --no-cublas-log       Skip CUBLAS*_LOG* env vars.
  --warmup N            Default: 0.
  --extra-bench-args "<...>"    Free-form extra args appended to trtllm-bench.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
        --input-mean) INPUT_MEAN="$2"; shift 2 ;;
        --output-mean) OUTPUT_MEAN="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max-num-tokens) MAX_NUM_TOKENS="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --extra-llm-api-options) EXTRA_LLM_API_OPTIONS="$2"; shift 2 ;;
        --no-streaming) STREAMING=0; shift ;;
        --no-cublas-log) CUBLAS_LOG=0; shift ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --extra-bench-args) EXTRA_BENCH_ARGS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[run_bench] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n "$LABEL" ]]      || { echo "[run_bench] --label required" >&2; exit 2; }
[[ -n "$MODEL_NAME" ]] || { echo "[run_bench] --model-name required" >&2; exit 2; }

if [[ -z "$MAX_NUM_TOKENS" ]]; then
    MAX_NUM_TOKENS=$(( INPUT_MEAN + OUTPUT_MEAN + 128 ))
fi

# ----- resolve run dir -------------------------------------------------------
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${TS}-${LABEL}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "[run_bench] run_dir=$RUN_DIR" >&2

# ----- copy or generate extra-llm-api yaml -----------------------------------
TARGET_YAML="${RUN_DIR}/extra-llm-api-config.yml"
if [[ -n "$EXTRA_LLM_API_OPTIONS" ]]; then
    cp "$EXTRA_LLM_API_OPTIONS" "$TARGET_YAML"
else
    cp "${SKILL_DIR}/templates/extra-llm-api-config.yml" "$TARGET_YAML"
fi

# ----- manifest --------------------------------------------------------------
# Some collected fields (trtllm version banner, GPU name) can contain stray
# emoji / control chars that break heredoc-built JSON. Build the manifest in
# Python so json.dumps escapes everything correctly.
git_sha() { ( cd "$1" 2>/dev/null && git rev-parse HEAD 2>/dev/null ) || true; }
git_dirty() { ( cd "$1" 2>/dev/null && [[ -n "$(git status --porcelain 2>/dev/null)" ]] ) && echo "1" || echo "0"; }

MANIFEST_GPU="$(nvidia-smi -L 2>/dev/null | head -1 | sed 's/^GPU [0-9]*: //' | sed 's/ (UUID: .*//' || true)"
# tensorrt_llm import emits a startup banner to stdout. Take the last line which
# is the actual __version__ value.
MANIFEST_TRTLLM_VER="$(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null | tail -1 || true)"
MANIFEST_TRTLLM_SHA="$(git_sha "$TRTLLM_REPO")"
MANIFEST_TRTLLM_DIRTY="$(git_dirty "$TRTLLM_REPO")"
MANIFEST_B12X_SHA="$(git_sha "${WORKSPACE}/b12x")"
MANIFEST_B12X_DIRTY="$(git_dirty "${WORKSPACE}/b12x")"

env \
    M_RUN_ID="$RUN_ID" M_LABEL="$LABEL" M_TS="$TS" \
    M_MODEL_NAME="$MODEL_NAME" M_MODEL_PATH="$MODEL_PATH" \
    M_CONCURRENCY="$CONCURRENCY" M_MAX_BATCH_SIZE="$MAX_BATCH_SIZE" \
    M_MAX_NUM_TOKENS="$MAX_NUM_TOKENS" M_TP="$TP" M_BACKEND="$BACKEND" \
    M_STREAMING="$STREAMING" M_CUBLAS_LOG="$CUBLAS_LOG" \
    M_NUM_REQUESTS="$NUM_REQUESTS" M_INPUT_MEAN="$INPUT_MEAN" \
    M_OUTPUT_MEAN="$OUTPUT_MEAN" \
    M_TRTLLM_SHA="$MANIFEST_TRTLLM_SHA" M_TRTLLM_DIRTY="$MANIFEST_TRTLLM_DIRTY" \
    M_B12X_SHA="$MANIFEST_B12X_SHA" M_B12X_DIRTY="$MANIFEST_B12X_DIRTY" \
    M_GPU="$MANIFEST_GPU" M_TRTLLM_VER="$MANIFEST_TRTLLM_VER" \
    M_WORKSPACE="$WORKSPACE" \
python3 <<'PY' > "${RUN_DIR}/manifest.json"
import json, os
def b(s): return s == "1"
m = {
    "run_id": os.environ.get("M_RUN_ID", ""),
    "label":  os.environ.get("M_LABEL", ""),
    "started_at_utc": os.environ.get("M_TS", ""),
    "model": {
        "name": os.environ.get("M_MODEL_NAME", ""),
        "path": os.environ.get("M_MODEL_PATH", ""),
    },
    "config": {
        "concurrency":    int(os.environ.get("M_CONCURRENCY") or 0),
        "max_batch_size": int(os.environ.get("M_MAX_BATCH_SIZE") or 0),
        "max_num_tokens": int(os.environ.get("M_MAX_NUM_TOKENS") or 0),
        "tp":             int(os.environ.get("M_TP") or 0),
        "backend":        os.environ.get("M_BACKEND", ""),
        "streaming":      b(os.environ.get("M_STREAMING", "")),
        "cublas_log":     b(os.environ.get("M_CUBLAS_LOG", "")),
        "num_requests":   int(os.environ.get("M_NUM_REQUESTS") or 0),
        "input_mean":     int(os.environ.get("M_INPUT_MEAN") or 0),
        "output_mean":    int(os.environ.get("M_OUTPUT_MEAN") or 0),
    },
    "git": {
        "trtllm_sha":   os.environ.get("M_TRTLLM_SHA", ""),
        "trtllm_dirty": b(os.environ.get("M_TRTLLM_DIRTY", "")),
        "b12x_sha":     os.environ.get("M_B12X_SHA", ""),
        "b12x_dirty":   b(os.environ.get("M_B12X_DIRTY", "")),
    },
    "env": {
        "gpu":            os.environ.get("M_GPU", ""),
        "trtllm_version": os.environ.get("M_TRTLLM_VER", ""),
        "workspace":      os.environ.get("M_WORKSPACE", ""),
    },
}
print(json.dumps(m, indent=2))
PY

# ----- dataset prep ----------------------------------------------------------
if [[ -z "$DATASET" ]]; then
    DATASET="${RUN_DIR}/dataset.jsonl"
    PREP="${TRTLLM_REPO}/benchmarks/cpp/prepare_dataset.py"
    [[ -f "$PREP" ]] || { echo "[run_bench] prepare_dataset.py not found at $PREP" >&2; exit 1; }
    TOKENIZER="$MODEL_NAME"
    [[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]] && TOKENIZER="$MODEL_PATH"
    echo "[run_bench] preparing dataset (n=${NUM_REQUESTS}, in=${INPUT_MEAN}, out=${OUTPUT_MEAN}) -> $DATASET" >&2
    python3 "$PREP" \
        --tokenizer "$TOKENIZER" \
        --stdout token-norm-dist \
        --num-requests="$NUM_REQUESTS" \
        --input-mean="$INPUT_MEAN" --output-mean="$OUTPUT_MEAN" \
        --input-stdev="$INPUT_STDEV" --output-stdev="$OUTPUT_STDEV" \
        > "$DATASET"
else
    cp "$DATASET" "${RUN_DIR}/dataset.jsonl" 2>/dev/null || true
    DATASET="${RUN_DIR}/dataset.jsonl"
fi

# ----- compose trtllm-bench command ------------------------------------------
BENCH_ARGS=( --model "$MODEL_NAME" )
[[ -n "$MODEL_PATH" ]] && BENCH_ARGS+=( --model_path "$MODEL_PATH" )
BENCH_ARGS+=(
    throughput
    --dataset "$DATASET"
    --extra_llm_api_options "$TARGET_YAML"
    --backend "$BACKEND"
    --tp "$TP"
    --concurrency "$CONCURRENCY"
    --max_batch_size "$MAX_BATCH_SIZE"
    --max_num_tokens "$MAX_NUM_TOKENS"
    --num_requests "$NUM_REQUESTS"
    --warmup "$WARMUP"
    --report_json "${RUN_DIR}/bench_report.json"
    --iteration_log "${RUN_DIR}/bench_iter.log"
)
[[ $STREAMING -eq 1 ]] && BENCH_ARGS+=( --streaming )
if [[ -n "$EXTRA_BENCH_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=( $EXTRA_BENCH_ARGS )
    BENCH_ARGS+=( "${EXTRA_ARR[@]}" )
fi

# ----- cuBLAS env ------------------------------------------------------------
CUBLAS_ENV=()
if [[ $CUBLAS_LOG -eq 1 ]]; then
    CUBLAS_ENV+=(
        "CUBLAS_LOGINFO_DBG=1"
        "CUBLAS_LOGDEST_DBG=${RUN_DIR}/cublas.log"
        "CUBLASLT_LOG_LEVEL=2"
        "CUBLASLT_LOG_FILE=${RUN_DIR}/cublasLt.log"
    )
fi

echo "[run_bench] launching trtllm-bench" >&2
echo "[run_bench] cmd: env ${CUBLAS_ENV[*]:-} trtllm-bench ${BENCH_ARGS[*]}" >&2

set +e
env "${CUBLAS_ENV[@]}" trtllm-bench "${BENCH_ARGS[@]}" \
    > "${RUN_DIR}/bench.stdout" 2> "${RUN_DIR}/bench.stderr"
RC=$?
set -e

echo "[run_bench] trtllm-bench exit=$RC" >&2
if [[ $RC -ne 0 ]]; then
    echo "[run_bench] bench failed; tail of stderr:" >&2
    tail -40 "${RUN_DIR}/bench.stderr" >&2 || true
    # Still try to parse partial artifacts so the failure is inspectable.
fi

# ----- parsers ---------------------------------------------------------------
PY=python3
if [[ -f "${RUN_DIR}/cublasLt.log" ]]; then
    echo "[run_bench] parsing cublasLt.log" >&2
    "$PY" "${SKILL_DIR}/scripts/parse_cublaslt_log.py" \
        --run-dir "$RUN_DIR" \
        --model "$MODEL_NAME" \
        || echo "[run_bench] cublasLt parser failed (non-fatal)" >&2
fi

if [[ -f "${RUN_DIR}/bench_report.json" ]]; then
    echo "[run_bench] parsing bench metrics" >&2
    "$PY" "${SKILL_DIR}/scripts/parse_bench_metrics.py" \
        --run-dir "$RUN_DIR" \
        || echo "[run_bench] bench parser failed (non-fatal)" >&2
fi

# Final line is the run-dir so callers can capture it.
echo "$RUN_DIR"
exit $RC
