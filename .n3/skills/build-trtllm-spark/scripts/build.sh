#!/usr/bin/env bash
# build.sh — build TensorRT-LLM inside the persistent Spark container.
#
# Wraps scripts/build_wheel.py --install with the host-side pre-flight
# (chmod, stale-binding cleanup) + Spark-side prologue (safe.directory,
# venv nuke, CCACHE_DIR redirect to NFS) so a single host call kicks off
# the full build and we don't rediscover the quirks on every iteration.
#
# Usage:
#   build.sh                # background-launch + monitor cold/warm build
#   build.sh --install-only # re-run just the install step (after a previous
#                            # build_wheel_targets succeeded but install_file
#                            # failed on a stale bindings.so chmod)
#   build.sh --foreground   # run synchronously and tail the log
#
# Env overrides:
#   B12X_TRTLLM_REPO   default: $WORKSPACE/TensorRT-LLM
#   B12X_TRTLLM_ARCH   default: 121 (sm_121, GB10/Spark)
#   B12X_TRTLLM_JOBS   default: 12
#   B12X_WORKSPACE     default: parent of this script's parents
#
# Output:
#   $WORKSPACE/.n3/runs/build/sm<arch>-build.log     full build log (NFS)
#   $WORKSPACE/.n3/ccache/                            populated ccache for next run

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DEFAULT="$(cd "${SKILL_DIR}/../../../.." && pwd)"
WORKSPACE="${B12X_WORKSPACE:-$WORKSPACE_DEFAULT}"
TRTLLM_REPO="${B12X_TRTLLM_REPO:-${WORKSPACE}/TensorRT-LLM}"
ARCH="${B12X_TRTLLM_ARCH:-121}"
JOBS="${B12X_TRTLLM_JOBS:-12}"

INSTALL_ONLY=0
FOREGROUND=0
for arg in "$@"; do
    case "$arg" in
        --install-only) INSTALL_ONLY=1 ;;
        --foreground|--fg) FOREGROUND=1 ;;
        --help|-h)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "[build-trtllm-spark] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

LOG_DIR="${WORKSPACE}/.n3/runs/build"
LOG="${LOG_DIR}/sm${ARCH}-build.log"
CCACHE_DIR="${WORKSPACE}/.n3/ccache"

# ----- pre-flight (host side) -----------------------------------------------
# These dirs must exist + be world-writable BEFORE the container (running as
# root, NFS-squashed to nobody:nogroup) tries to write to them. If the dirs
# already exist with restrictive perms the build script's redirects will fail
# silently with "Permission denied".
mkdir -p "$LOG_DIR" "$CCACHE_DIR"
chmod 777 "$LOG_DIR" "$CCACHE_DIR" 2>/dev/null || true
chmod 777 "${WORKSPACE}/.n3" 2>/dev/null || true
chmod 777 "${WORKSPACE}/.n3/runs" 2>/dev/null || true

# Stale aarch64 binding from a previous build trips install_file because root
# in the container (NFS root-squash → nobody:nogroup) cannot chmod a file
# owned by the host user. Remove it so install_file recreates it fresh.
# The .x86_64 sibling is harmless — leave it alone.
if [[ -f "${TRTLLM_REPO}/tensorrt_llm/bindings.cpython-312-aarch64-linux-gnu.so" ]]; then
    rm -f "${TRTLLM_REPO}/tensorrt_llm/bindings.cpython-312-aarch64-linux-gnu.so"
    echo "[build-trtllm-spark] cleared stale aarch64 binding" >&2
fi

# Verify the repo on disk matches expectations.
if [[ ! -f "${TRTLLM_REPO}/scripts/build_wheel.py" ]]; then
    echo "[build-trtllm-spark] ERROR: ${TRTLLM_REPO}/scripts/build_wheel.py not found" >&2
    echo "[build-trtllm-spark] set B12X_TRTLLM_REPO to a valid TRT-LLM checkout" >&2
    exit 1
fi

# ----- assemble the inner command (runs in spark container as root) ---------
INNER_CMD=$(cat <<EOF
set -euo pipefail
cd ${TRTLLM_REPO}
git config --global --add safe.directory '*' 2>/dev/null || true
$( [[ $INSTALL_ONLY -eq 0 ]] && echo "rm -rf .venv-3.12" )
export CCACHE_DIR=${CCACHE_DIR}
( echo "[start] \$(date)"
  $( [[ $INSTALL_ONLY -eq 1 ]] \
       && echo "python3 ./scripts/build_wheel.py --cuda_architectures '${ARCH}-real' -j ${JOBS} --use_ccache --skip_building_wheel --install" \
       || echo "python3 ./scripts/build_wheel.py --cuda_architectures '${ARCH}-real' -j ${JOBS} --configure_cmake --use_ccache --install" )
  echo "[end] \$(date) exit=\$?"
) > ${LOG} 2>&1 &
echo "[build-trtllm-spark] build PID=\$!"
EOF
)

# ----- launch via command_on_spark.sh ---------------------------------------
SPARK="${WORKSPACE}/.claude_docs/scripts/command_on_spark.sh"
[[ -x "$SPARK" ]] || { echo "[build-trtllm-spark] ERROR: $SPARK not executable" >&2; exit 1; }

echo "[build-trtllm-spark] repo:    ${TRTLLM_REPO}"
echo "[build-trtllm-spark] arch:    ${ARCH}-real (sm_${ARCH})"
echo "[build-trtllm-spark] jobs:    ${JOBS}"
echo "[build-trtllm-spark] log:     ${LOG}"
echo "[build-trtllm-spark] ccache:  ${CCACHE_DIR}"
echo "[build-trtllm-spark] mode:    $( [[ $INSTALL_ONLY -eq 1 ]] && echo install-only || echo full )"
echo "[build-trtllm-spark] kicking off in spark container..."
echo

if [[ $FOREGROUND -eq 1 ]]; then
    # Synchronous: forward to spark, then tail the log until [end].
    "$SPARK" bash -lc "$INNER_CMD" 2>&1 | grep -v '^\[spark\]' | tail -3
    echo "[build-trtllm-spark] tailing $LOG until [end]..."
    until grep -q '^\[end\]' "$LOG" 2>/dev/null; do sleep 30; done
    echo
    grep '^\[end\]' "$LOG" | tail -1
else
    "$SPARK" bash -lc "$INNER_CMD" 2>&1 | grep -v '^\[spark\]' | tail -3
    echo
    echo "[build-trtllm-spark] running in background on Spark."
    echo "[build-trtllm-spark] monitor with:"
    echo "    tail -f $LOG"
    echo "[build-trtllm-spark] verify completion:"
    echo "    grep '^\[end\]' $LOG"
    echo "[build-trtllm-spark] verify install:"
    echo "    $SPARK python3 -c 'import tensorrt_llm; print(tensorrt_llm.__file__, tensorrt_llm.__version__)'"
fi
