# Cutlass Collective FP8 Deep-Dive (synthesized retroactively)

> Note: this campaign was triggered by an in-session prompt ("tune the gemm we
> want yourself") rather than an edit to `b12x/.n3/TASK.md`. The TASK.md and
> KICKOFF.md captured here are reconstructed from artifacts and the user's
> directive. The actual campaign body lives in `runs/tracks/cutlass_*` files.

## Goal

After the 4-track exploration concluded with no promotion, AVO was asked to
pivot from atom-swapping to a real warp-specialized + TMA + multi-stage
mainloop using `cutlass::gemm::collective::CollectiveBuilder` against the
sm_120 FP8 atom (`SM120_16x8x32_TN<e4m3, e4m3, float>`). Target: get within
striking distance of `torch._scaled_mm` (cuBLASLt nvjet) on the headline
shape `(M=32, K=5376, N=5376)`.

## Constraints

- Pure CUDA C++ extension (no CuTeDSL Python).
- Use `KernelTmaWarpSpecialized` schedule from public CUTLASS.
- Parameterize tile shape and stage count via env-var-controlled `-D` macros
  so we can sweep without re-editing source.
- Headline shape and validation gates inherited from the parent 4-track plan.

## What was actually done

- Built `b12x/gemm/fp8_dense_cutlass_collective_ext.cu` — JIT-loaded torch
  CUDA extension wrapping `CollectiveBuilder` with FP8 e4m3 inputs, bf16
  output, fp32 accumulate.
- Built `b12x/gemm/fp8_dense_cutlass_collective.py` — Python wrapper with
  `B12X_CUTLASS_TILE_{M,N,K}` and `B12X_CUTLASS_SCHED_STAGES` env vars baked
  into the build via `-D` macros.
- Ran `cutlass_collective_tune.py` (8 configs) and `cutlass_collective_tune2.py`
  (13 more configs) — total 21 distinct (TileM, TileN, TileK, Stages) tuples
  swept on the headline shape.

## Outcome

Best correct cutlass-collective config: **`(128, 64, 128, 2)`** at 88.10 µs
hot-cache vs `torch._scaled_mm` 79.57 µs (1.107× — 10% slower).

Did NOT use L2 flush — both kernels measured with hot cache between iterations.
The 1.107× ratio understates true cold-cache gap because torch fundamentally
hits L2 differently at small M.

Recommendation at the time: do not promote, keep cuBLASLt as production. Next
step suggested: deeper TMA + warp-specialized mainloop variants, custom
epilogue, persistent scheduling tweaks.

(That recommendation is what triggered the next campaign,
`2026-05-04-cublass-pipelining-push`.)

## Stop condition that fired

3 steps × 4 tracks = 12 iteration budget exhausted on the parent 4-track plan
plus 21 tune-sweep configs on this collective deep-dive. Final session token
budget approached the 1M ceiling, AVO emitted a stop and the user ended the
run manually.
