---
name: cublas-gemm-tuning
description: >
  Tune cuBLAS / cuBLASLt for a specific GEMM shape and dtype combination on
  the current GPU. Returns the best correct algorithm config + measured
  timing as JSON. Use when AVO needs to (a) establish a strong baseline before
  optimizing a custom kernel, (b) decide whether to promote a custom kernel
  vs. cuBLASLt for a given shape, (c) generate a per-shape autotune table for
  a kernel-resolution registry.
  Supports FP16, BF16, FP8 e4m3/e5m2, INT8, and FP32 (full + tensor-op).
  Heuristic mode picks top-N candidates from cuBLASLt's built-in heuristic;
  enumerate mode iterates all algo IDs × tile/stage/splitK/swizzling configs.
user-invocable: false
license: LicenseRef-NvidiaProprietary
metadata:
  author: AVO project
  upstream: https://docs.nvidia.com/cuda/cublas/
---

# cuBLAS / cuBLASLt GEMM tuning

A standalone CLI binary `tune_gemm` (built from `scripts/tune_gemm.cu`) drives
all tuning. AVO calls it from the shell — there is no Python wrapper. Output
is JSON on stdout that AVO parses. The binary uses **cuBLASLt** (modern API,
heuristic + full algo enumeration). Classic cuBLAS via `cublasGemmEx` is a
thin fallback path inside the same binary, gated by `--api classic`.

## When to use

Trigger on these intents:
- "what's the best cuBLASLt config for `(M, K, N) = (...)` at `dtype = ...`"
- "establish the cuBLASLt floor for this shape before I optimize"
- "does cuBLASLt have anything faster than what `torch._scaled_mm` picked"
- "tune the production prefill GEMM for these 4 shapes"
- per-shape autotune table generation

Do NOT trigger for:
- non-GEMM ops (BatchNorm, Conv, attention) — wrong API
- shapes where you've already tuned and the result is in the artifact JSON

## Quick recipe

```bash
# Build once per container (idempotent; ~5 sec)
.n3/skills/cublas-gemm-tuning/scripts/build.sh

# Heuristic tune one shape (the common case)
.n3/skills/cublas-gemm-tuning/scripts/tune_gemm \
    --M 32 --K 5376 --N 5376 \
    --a-dtype e4m3 --b-dtype e4m3 --c-dtype bf16 --compute fp32 \
    --trans-a N --trans-b T \
    --warmup 10 --iters 20 --repeats 5 \
    --request-count 16 \
  > .n3/runs/tracks/<track>/step<N>/cublaslt_heuristic.json

# Full algo enumeration (slower, more thorough; use when heuristic plateaus)
.n3/skills/cublas-gemm-tuning/scripts/tune_gemm \
    --M 32 --K 5376 --N 5376 \
    --a-dtype e4m3 --b-dtype e4m3 --c-dtype bf16 --compute fp32 \
    --trans-a N --trans-b T \
    --enumerate \
    --warmup 5 --iters 10 --repeats 3 \
  > .n3/runs/tracks/<track>/step<N>/cublaslt_enumerate.json

# All commands MUST be wrapped in command_on_spark.sh from the AVO host:
.claude_docs/scripts/command_on_spark.sh \
    .n3/skills/cublas-gemm-tuning/scripts/tune_gemm --M 32 ...
```

## What the JSON contains

```json
{
  "device": "NVIDIA GB10",
  "compute_capability": "12.1",
  "shape": {"M": 128, "K": 2688, "N": 1},
  "dtype": {"a": "bf16", "b": "bf16", "c": "bf16", "compute": "fp32"},
  "transpose": {"a": "T", "b": "N"},
  "measurement": {"warmup": 5, "iters": 20, "repeats": 3},
  "heuristic_returned": 10,
  "candidates": [
    {
      "rank_measured": 0,           // index after sorting by measured median
      "rank_heuristic": 7,           // index in cuBLAS-Lt's predicted-perf order
      "algo_id": 13, "tile_id": 0, "stages_id": 0,
      "splitk_num": 9, "swizzling": 0, "reduction_scheme": 0,
      "workspace_bytes": 33554432,
      "wave_count": 4.0,
      "median_us": 16.192, "mean_us": 16.21, "min_us": 16.18,
      "spread_pct": 0.6,
      "speedup_vs_heuristic_baseline": 1.1383,
      "is_heuristic_baseline": false,
      "launch_ok": true
    },
    ...
  ],
  "heuristic_baseline": {           // rank_heuristic == 0 — what cuBLAS-Lt would auto-pick
    "rank_heuristic": 0, "rank_measured": 2,
    "algo_id": 13, "tile_id": 0, "stages_id": 0,
    "splitk_num": 1, "swizzling": 0,
    "median_us": 18.432
  },
  "best": {                         // fastest candidate after the sweep
    "rank_heuristic": 7, "rank_measured": 0,
    "algo_id": 13, "tile_id": 0, "stages_id": 0,
    "splitk_num": 9, "swizzling": 0,
    "median_us": 16.192,
    "speedup_vs_heuristic_baseline": 1.1383
  }
}
```

**Reading the result.** `heuristic_baseline` is what cuBLAS-Lt would pick on
its own without a sweep — the "stock" reference. `best` is the fastest
candidate among the top-N heuristic candidates after actually timing them.
`best.speedup_vs_heuristic_baseline` quantifies how much you gained by
sweeping. If `best.rank_heuristic == 0`, the heuristic was already optimal
and a sweep gives nothing — common for shapes the heuristic has good
coverage on. If `best.rank_heuristic > 0` (as in the example above), the
heuristic's predicted-perf order missed a faster config.

If the heuristic returns zero candidates, `heuristic_baseline` and `best`
are both `null` — escalate (different dtype, larger workspace, or check the
sm_120 FP8 caveat below).

## Workflow (AVO-side)

1. **Build once.** `scripts/build.sh` is idempotent and skipped if `tune_gemm`
   already exists and is newer than the source.
2. **Heuristic mode first** (`--request-count 16`). Fast — ~1 sec total. Picks
   the top-N candidates cuBLASLt's built-in heuristic ranks by predicted perf,
   times each, runs correctness vs FP32 reference, returns the best.
3. **If heuristic doesn't beat the target**, escalate to `--enumerate`. Full
   algo iteration: walks every valid (algo_id, tile_id, stages_id, splitk,
   swizzling) tuple within capability limits. Slow (1-30 min for big shapes),
   exhaustive.
4. **Persist the best config.** Write the JSON under
   `.n3/runs/tracks/<track>/step<N>/cublaslt_*.json`. The next iteration uses
   that same algo via `--force-algo <algo_id> --force-config tile=...,stages=...`
   which skips heuristic and reproduces the exact tuned config.

## Validation gates (built into the binary)

| Gate | Threshold | Source |
|---|---|---|
| Correctness vs FP32 reference | `cos ≥ 0.9995` | `oracle_cos` field |
| Max-abs vs FP32 reference | `max_abs ≤ 1% × ‖ref‖_∞` | `oracle_max_abs_ok` field |
| All outputs finite | `isfinite(all)` | `finite` field |
| Repeat-median spread | `≤ 2%` | `spread_pct` field |

A candidate is `passed_correctness: false` if any gate fails. The `best` field
is the fastest candidate that passed all gates. If nothing passes, `best` is
`null` and AVO must escalate.

## FP8 / NVFP4 / blockscaled specifics

For FP8 paths use `--compute fp32` (FP8 always accumulates in FP32) and the
canonical TN layout: `--trans-a T --trans-b N`. The binary auto-attaches A/B/D
scale pointers (unit scales, fp32) and sets `CUBLASLT_MATMUL_DESC_FAST_ACCUM=1`
when either input is FP8/INT8.

**Known caveat — sm_120 (GB10/Spark) FP8 returns 0 heuristic candidates with
status=7 (`INVALID_VALUE`).** This appears to be because cuBLASLt on sm_120
expects block-scaled `kind::mxf8f6f4` scale mode (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`)
rather than per-tensor `SCALAR_32F`. b12x's hand-written `fp8_dense_cuda_ext.cu`
hits the same hardware path with degenerate unit scales — its PTX is
`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32...ue8m0`.
To make `tune_gemm` work on sm_120 FP8, we'd need to:

1. Allocate properly-shaped UE8M0 scale tensors (1 byte per 32-element block
   along the K dim of A and B).
2. Set `CUBLASLT_MATMUL_DESC_A_SCALE_MODE = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`
   and `B_SCALE_MODE` similarly.
3. Initialize the scales to encode `2^0 = 1.0` (`0x7F` byte) for unit scaling.

This is a follow-up — see `references/fp8-and-blockscaled.md` for details.
Until then, `tune_gemm` works for **fp16, bf16, fp32, fp32_tf32** end-to-end
and for **fp8 on sm_89/sm_90** (Ada/Hopper) where per-tensor scales are
accepted by cuBLASLt's heuristic.

## When `tune_gemm` is the wrong tool

- For per-block-scaled NVFP4 with `vec_size=16` E4M3 SF: use
  `cublasLtMatmulMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BLOCK_SCALE_VECTOR_SIZE, ...)`.
  The binary supports this via `--scale-mode block_e4m3_vec16` but verify the
  PTX kind is `mxf8f6f4` post-tune.
- For mixed-precision FP8×BF16 (one operand FP8, the other BF16): supported
  via `--a-dtype e4m3 --b-dtype bf16`, but cuBLASLt heuristic coverage is thin
  on sm_120 — escalate to `--enumerate` immediately.
- Beyond `(M, K, N) = ~64K` per dim: cuBLASLt may run out of workspace.
  Bump `--max-workspace-bytes` (default 32 MB).

## References

| File | Purpose |
|---|---|
| `references/cublaslt-workflow.md` | Heuristic + algo iteration, with C code |
| `references/algo-config-attrs.md` | All `CUBLASLT_ALGO_CONFIG_*` knobs |
| `references/fp8-and-blockscaled.md` | FP8 / MXFP8 / NVFP4 setup specifics |

## Scripts

| File | Purpose |
|---|---|
| `scripts/tune_gemm.cu` | The C++ tuner (single source file, ~700 lines) |
| `scripts/build.sh` | One-line `nvcc` build (idempotent) |
| `scripts/tune_gemm` | Built binary (gitignored) |

## Common pitfalls

- **Don't run on the AVO host.** Spark host (p4242-0053) has the GB10 GPU.
  Always wrap with `.claude_docs/scripts/command_on_spark.sh`.
- **Workspace size matters.** Several fast algos need ≥16 MB. The binary
  defaults to 32 MB; reduce only if you're tuning many shapes in parallel
  and hitting OOM.
- **Heuristic may return zero candidates** for unusual shapes (very thin M,
  unusual K alignment). The binary reports `candidates: []` — escalate to
  `--enumerate` with `--allow-illegal-config-skip`.
- **L2 flush is mandatory for valid timing.** The binary always flushes
  unless `--no-l2-flush` is passed (don't pass it).
- **Tensor-Op alignment.** FP8 GEMMs need M/N/K alignment ≥ 16 for most
  algos; E5M2 requires K alignment ≥ 32. The binary detects misalignment
  and reports it instead of silently picking a slower fallback.
