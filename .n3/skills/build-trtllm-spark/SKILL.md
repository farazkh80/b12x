---
name: build-trtllm-spark
description: >
  Build TensorRT-LLM from source on the Spark host (GB10/sm_121) inside the
  existing avo-b12x-spark persistent container, with ccache on NFS so
  rebuilds are warm. Use when the user has modified TRT-LLM source (e.g.
  patched a header, added a kernel, bumped to a new sha) and needs a
  freshly-built wheel that overrides the container's pre-baked rc12 install.
  Adapted from build-trtllm-local; turns the host-side docker dance into a
  command_on_spark.sh wrapper, redirects ccache to NFS, and pre-emptively
  clears the stale aarch64 binding that breaks `install_file`.
user-invocable: false
license: LicenseRef-NvidiaProprietary
metadata:
  author: AVO project
---

# build-trtllm-spark

Builds TensorRT-LLM via `scripts/build_wheel.py --install` inside the
persistent Spark container (`avo-b12x-spark-<user>`, image
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12`). Cold-cache wall-time is
**42 min on GB10 with `-j 12`**, dominated by:

- ~3 min CMake configure + Conan dependency install
- ~30 min cutlass/fmha/MoE CUDA kernel compilation (the slow templates)
- ~5 min CXX runtime + thop/ Python bindings (this is where our
  `cublasScaledMMLut.h` patches recompile)
- ~3 min `libtensorrt_llm.so` link + nanobind + wheel install

Warm-cache rebuilds (after a trivial header edit) are dramatically faster —
the populated `$WORKSPACE/.n3/ccache/` dir is shared across sessions.

## When to use

Trigger on:
- "rebuild TRT-LLM with my patches"
- "the bench used the old wheel — build my fork and reinstall"
- "compile TensorRT-LLM for SM121 / Blackwell"

Do **not** trigger for:
- Building b12x or its kernels (`b12x/scripts/build.sh` etc.) — different repo
- Building `tune_gemm` (use `cublas-gemm-tuning/scripts/build.sh`)
- Slurm cluster builds — use `trtllm-slurm-compile` instead

## Quick recipe

```bash
# Default: build for sm_121 (GB10/Spark) with the in-workspace TRT-LLM clone.
b12x/.n3/skills/build-trtllm-spark/scripts/build.sh

# Custom arch (also passes through to scripts/build_wheel.py --cuda_architectures):
B12X_TRTLLM_ARCH=120 b12x/.n3/skills/build-trtllm-spark/scripts/build.sh

# Custom repo dir (e.g. a sibling clone):
B12X_TRTLLM_REPO=/home/farazkh_scratch/some/other/TensorRT-LLM \
    b12x/.n3/skills/build-trtllm-spark/scripts/build.sh

# Re-run install only (after a previous build target succeeded but install_file failed):
b12x/.n3/skills/build-trtllm-spark/scripts/build.sh --install-only
```

The script orchestrates `command_on_spark.sh` itself — caller doesn't need
to wrap. Logs land at `$WORKSPACE/.n3/runs/build/sm<arch>-build.log` (NFS).

## Pre-flight (the script handles, listed for visibility)

```bash
chmod 777 $WORKSPACE/.n3 \
          $WORKSPACE/.n3/ccache \
          $WORKSPACE/.n3/runs \
          $WORKSPACE/.n3/runs/build
# rm the stale aarch64 binding the `install_file` step trips on (next gotcha)
rm -f $WORKSPACE/TensorRT-LLM/tensorrt_llm/bindings.cpython-*-aarch64-linux-gnu.so
```

## Build (inside Spark container)

```bash
cd $WORKSPACE/TensorRT-LLM &&
git config --global --add safe.directory '*' &&
rm -rf .venv-3.12 &&
export CCACHE_DIR=$WORKSPACE/.n3/ccache &&
( echo "[start] $(date)";
  python3 ./scripts/build_wheel.py --cuda_architectures '<arch>-real' \
      -j 12 --configure_cmake --use_ccache --install;
  echo "[end] $(date) exit=$?"
) > $LOG 2>&1 &
```

The `&` keeps the build alive past the SSH session. Monitor with `tail -f`
or grep for the `[end]` marker.

## Post-build verification

```bash
command_on_spark.sh python3 -c "
import tensorrt_llm
print('path:', tensorrt_llm.__file__)
print('version:', tensorrt_llm.__version__)
"
```

Expected output for a successful install:
- `path: /home/farazkh_scratch/.../TensorRT-LLM/tensorrt_llm/__init__.py`  ← **dev install**, NOT `/usr/local/lib/python3.12/dist-packages/...`
- `version: 1.3.0rc13` (or whatever your fork's `tensorrt_llm/version.py` says)

If `path` is still under `/usr/local/lib/python3.12/dist-packages/`, the
`--install` step didn't replace the container's pre-baked wheel — re-run
with `--install-only`.

## Known quirks (every one was hit during the 2026-05-04 verification run)

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: [Errno 8] Exec format error: '.venv-3.12/bin/python3'` | Stale aarch64 venv from a prior build | `rm -rf .venv-3.12` before invoking the build (script does this). |
| `fatal: detected dubious ownership` from any `git ...` inside container | UID mismatch host↔container | `git config --global --add safe.directory '*'` once per fresh container (script does this). |
| `bash: <log>: Permission denied` when redirecting build stdout | NFS root-squash on user-owned host dir | `chmod 777` `.n3/`, `.n3/ccache`, `.n3/runs`, `.n3/runs/build` BEFORE the container writes (script does this). |
| `PermissionError: [Errno 1] Operation not permitted: '.../tensorrt_llm/bindings.cpython-312-aarch64-linux-gnu.so'` at `install_file` step | Stale `bindings.cpython-*-aarch64-linux-gnu.so` owned by host UID; NFS root-squash blocks root-in-container chmod | `rm -f tensorrt_llm/bindings.cpython-*-aarch64-linux-gnu.so` before `--install` (script does this).  Also: the .x86_64 sibling is harmless — leave it alone. |
| `[end] exit=0` but `tensorrt_llm.__file__` still points at `/usr/local/lib/...` | The script's `--install` step crashed silently after `Built target build_wheel_targets` but BEFORE pip install of the wheel | Re-run the build with `--install-only` (skips the C++ rebuild, just runs the install part). |
| Build crawls past 30 min on first compile | Cold ccache (100 % miss); cutlass/fmha CUDA template TUs are slow | Patience. Subsequent builds with the populated `$WORKSPACE/.n3/ccache` are 5-10 min for header-only changes. |
| `ccache -s` shows 0 / 0 hits permanently | `CCACHE_DIR` env not propagating into the docker exec | Verify the script's `export CCACHE_DIR=$WORKSPACE/.n3/ccache` line is BEFORE `python3 ./scripts/build_wheel.py ...` and not stripped by SSH quoting. |

## What gets installed where

The script's `--install` flag:
1. Copies CMake build artifacts (`libtensorrt_llm.so`, `bindings.cpython-*.so`,
   plugin libs) from `cpp/build/...` into the in-source package dir
   `TensorRT-LLM/tensorrt_llm/{libs,bindings.cpython-*}.so`. **This is the
   step that hits the stale-bindings PermissionError.**
2. Builds a wheel: `tensorrt_llm-<version>-py3-none-any.whl`.
3. Runs `pip install <wheel>` into the container's Python env. Replaces the
   container's pre-baked `1.3.0rc12` install with the freshly-built one.

After install, the container's Python imports `tensorrt_llm` from the
**dev install** at `$WORKSPACE/TensorRT-LLM/tensorrt_llm/`, not from
`/usr/local/lib/python3.12/dist-packages/`. Restarts of the container won't
revert (the install path is in the workspace bind-mount, persistent).

## Time / disk footprint

Wall-times verified 2026-05-04 on GB10 with `-j 12`:

| scenario | total | what dominates |
|---|---:|---|
| **Cold build** (empty ccache, no prior build artifacts) | **~42 min** | ~30 min cutlass + fmha CUDA template TUs |
| **Warm rebuild** after a single header edit (e.g. `cublasScaledMMLut.h`) | **~13 min** | ~3 min CMake reconfigure, ~3 min compile (2 dependent TUs), **~7 min wheel packaging** (cutlass/fmha header copy + zip; runs twice due to TRT-LLM's build-then-install flow) |

**ccache contribution is ~zero on warm rebuild.** The speedup comes from
**ninja's incremental decisions** — for unmodified TUs ninja never invokes
nvcc, so there's nothing for ccache to short-circuit. ccache populates ~0.4
GiB during the cold build but warm-rebuild hit rate is <1 % (verified
`ccache -s`). The cache earns its keep when you DO need to recompile a
previously-built TU (e.g., revert a change), not on incremental builds.

The dominant cost on warm rebuilds is the **wheel-packaging phase**
(setuptools `bdist_wheel` copies ~30k cutlass headers/docs into the wheel
build tree, twice — once for the wheel, once for the editable-install
isolation env). This is independent of whether anything actually recompiled.

Disk:
- `$WORKSPACE/.n3/ccache/`: grows to ~0.4-1 GiB after one cold build, capped
  at 5 GiB by ccache default.
- `$WORKSPACE/TensorRT-LLM/cpp/build/`: ~10 GiB of CMake build artifacts.
- `$WORKSPACE/TensorRT-LLM/tensorrt_llm/libs/`: ~700 MB of installed `.so`s.

## Files in the skill

| file | purpose |
|---|---|
| `SKILL.md` | This file |
| `scripts/build.sh` | Entrypoint. Handles pre-flight chmods + stale-binding cleanup, runs the build inside the spark container, optionally `--install-only` to retry just the install step |

## See also

- `~/.claude/skills/build-trtllm-local/SKILL.md` — the original on-host docker
  flow this skill is adapted from.
- `~/.claude/projects/-home-farazkh-scratch-agentic-dev-avo-b12x-trt/memory/trtllm_nvfp4_lut_verified.md` — the 2026-05-04 verification run that surfaced every quirk above.
- `~/.claude/projects/-home-farazkh-scratch-agentic-dev-avo-b12x-trt/memory/nfs_run_paths.md` — where the build log of that verification run lives.
