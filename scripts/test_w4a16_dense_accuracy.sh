#!/usr/bin/env bash
# test_w4a16_dense_accuracy.sh
# ---------------------------
# Run the bit-exact accuracy gate for the W4A16 dense GEMM kernels across all
# Nano3.5 dense shapes.  Two passes:
#   1. Triton (default backend) at M ∈ {1, 8, 32}
#   2. CuTe-DSL (env-gated v4 → v3 dispatch) at M=16
#
# Both passes assert cos > 0.9999 and max_abs ≤ max(0.04, 1% × ref.abs().max()).
# The v4 path consumes the swizzled FP8 SF tensor directly; the v3 path pays a
# host-side unswizzle.  Both share the test harness in tests/test_dense_gemm_w4a16.py.
#
# Usage:
#   scripts/test_w4a16_dense_accuracy.sh                     # both passes
#   scripts/test_w4a16_dense_accuracy.sh triton              # Triton only
#   scripts/test_w4a16_dense_accuracy.sh cute                # CuTe only
#
# Env overrides:
#   B12X_GEMM_W4A16_MAX_ACTIVE   — persistent scheduler cap (default: SM*3, max 256)
#   TRITON_CACHE_DIR             — override triton cache dir (default: /tmp/triton-$USER)
#
# On Spark (cutlass-dsl 4.3.4 default), prepend the 4.4.2 sidecar to PYTHONPATH:
#   pip3 install --target=/tmp/cutlass_4_4_2 'nvidia-cutlass-dsl==4.4.2'
#   export PYTHONPATH=/tmp/cutlass_4_4_2/nvidia_cutlass_dsl/python_packages:${PYTHONPATH}
#
# The cute pass cold-JIT-compiles one kernel per unique (m, n, k, tile_*).
# Expect ~5-15s per shape at K=2688/4096 on first run, near-instant on cached re-runs.

set -euo pipefail
cd "$(dirname "$0")/.."  # repo root (b12x)

PASS="${1:-both}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-${USER:-claude}}"
mkdir -p "$TRITON_CACHE_DIR"

run_triton_pass() {
  echo "=========================================================="
  echo "  Pass 1: Triton (default), all Nano35 shapes × M ∈ {1,8,32}"
  echo "=========================================================="
  unset B12X_GEMM_W4A16_USE_CUTE
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    python -m pytest tests/test_dense_gemm_w4a16.py::test_dense_gemm_w4a16_nano35_accuracy \
      -v --tb=short
}

run_cute_pass() {
  echo "=========================================================="
  echo "  Pass 2: CuTe-DSL (v4 → v3 dispatch), all Nano35 shapes × M=16"
  echo "=========================================================="
  export B12X_GEMM_W4A16_USE_CUTE=1
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    python -m pytest tests/test_dense_gemm_w4a16.py::test_dense_gemm_w4a16_cute_m16_accuracy \
      -v --tb=short
  unset B12X_GEMM_W4A16_USE_CUTE
}

case "$PASS" in
  triton) run_triton_pass ;;
  cute)   run_cute_pass ;;
  both)   run_triton_pass; echo; run_cute_pass ;;
  *) echo "Usage: $0 [triton|cute|both]" >&2; exit 2 ;;
esac

echo
echo "=========================================================="
echo "  Accuracy sweep PASSED"
echo "=========================================================="
