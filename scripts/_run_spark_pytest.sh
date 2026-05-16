#!/bin/bash
# Reproducible wrapper: full pytest accuracy sweep on Spark with the
# cutlass-dsl 4.4.2 sidecar.  Auto-installs the sidecar if missing.
set -e
SIDECAR=/tmp/cutlass_4_4_2
if [ ! -d "$SIDECAR/nvidia_cutlass_dsl/python_packages" ]; then
  echo "[setup] installing cutlass-dsl 4.4.2 sidecar to $SIDECAR"
  pip3 install --target="$SIDECAR" --upgrade 'nvidia-cutlass-dsl==4.4.2' >/dev/null 2>&1
fi
export PYTHONPATH=$SIDECAR/nvidia_cutlass_dsl/python_packages:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton-pytest-$$
export B12X_CUTE_COMPILE_DISK_CACHE=0
mkdir -p "$TRITON_CACHE_DIR"
cd /home/farazkh_scratch/agentic-dev/b12x
echo "[setup] cutlass: $(python3 -c 'import cutlass; print(cutlass.__file__)')"
exec bash scripts/test_w4a16_dense_accuracy.sh both
