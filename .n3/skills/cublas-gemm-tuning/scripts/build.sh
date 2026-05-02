#!/usr/bin/env bash
# build.sh — compile tune_gemm.cu against cuBLASLt. Idempotent.
#
# AVO calls:
#   .n3/skills/cublas-gemm-tuning/scripts/build.sh
# from anywhere; the script resolves its own dir and the resulting binary lives
# next to the source as scripts/tune_gemm.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/tune_gemm.cu"
OUT="$HERE/tune_gemm"

# Skip rebuild if binary is newer than source.
if [[ -x "$OUT" && "$OUT" -nt "$SRC" ]]; then
  echo "[build] $OUT is up-to-date" >&2
  exit 0
fi

NVCC="${NVCC:-nvcc}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Architectures: explicitly include sm_120 (GB10/Spark) and sm_121 because some
# code paths gate on the feature macro. -gencode lines target SM75+ for tensor
# cores and add sm_89/sm_90/sm_120 explicitly.
ARCH_FLAGS=(
  -gencode arch=compute_89,code=sm_89
  -gencode arch=compute_90,code=sm_90
  -gencode arch=compute_120,code=sm_120
)

# CUTLASS feature macro that gates the SM120 FP8/FP6/FP4 mma path inside cute.
DEFS=( -DCUTE_ARCH_F8F6F4_MMA_ENABLED -DCUTE_ARCH_MMA_F32_SM89_ENABLED )

# Compile.
"$NVCC" -O3 -std=c++17 \
  "${DEFS[@]}" \
  "${ARCH_FLAGS[@]}" \
  -I"$CUDA_HOME/include" \
  -L"$CUDA_HOME/lib64" \
  -lcublas -lcublasLt -lcudart \
  -o "$OUT" \
  "$SRC"

echo "[build] built $OUT" >&2
