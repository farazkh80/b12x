# FP8 Dense GEMM — Autonomous Optimization

## Goal

A high-performance **FP8 dense GEMM** for b12x on SM121 (DGX Spark / GB10) that beats `torch._scaled_mm` / cuBLASLt on the production decode + prefill shapes.

## Status (read this first)

- **Inline-PTX kernel done.** `b12x/gemm/fp8_dense_cuda_ext.cu` works on Spark and beats `torch._scaled_mm` on most decode shapes. Uses sm_120 block-scaled FP8 mma (`kind::mxf8f6f4`) with degenerate unit scales. Stays in-tree as the parity floor and PTX reference. Do not delete or regress it.
- **Current focus: 4-track FP8 exploration.** AVO explores four implementation paths (CUDA C iteration, Cutlass C++ with `SM120_16x8x32_TN`, Triton, CuTeDSL+inline_asm) with a 3-step tuning budget per track and recorded handoff between them. Headline target: 70–80 µs on the chosen headline shape. **See `.n3/CUTEDSL_FP8_PLAN.md`** (now the multi-track plan) for the full brief, verified ground-truth analysis, per-track step structure, and recording schema.

## Pointers

- `.n3/CUTEDSL_FP8_PLAN.md` — the active sub-task brief.
- `.n3/FP8_DENSE_GEMM_BRIEF.md` — original implementation brief; still useful for shape table and API surface.
- `.n3/SPARK_QUIRKS.md` — must-read before any GPU work (nsys `-t cuda-sw` not `-t cuda`, etc.).
- `.n3/runs/baseline/fp8_dense_baseline.{md,json}` — target shapes + baseline scores.

## Autonomy contract (CRITICAL when running autopilot)

When invoked in autopilot mode (no `--chat`, with `task_planner` driving), AVO operates **without a user in the loop**.

1. **Do NOT use `AskUserQuestion` / `ask`.** Pick the option with highest expected information gain, write a one-paragraph justification to `.n3/runs/v${N}/hypothesis.md`, proceed.
2. **Never block on confirmation.** If a perf gain looks suspicious, run verification a second time at higher repeat count rather than asking.
3. **Hard blockers** (CUDA driver dead, repo corrupted, API key revoked) are the only legitimate stop conditions. Append to `.n3/workflow-alert.md` and emit `NO_TASK` from task_planner. Don't ask.
4. **Ambiguity in the brief** → pick the interpretation that best matches the success criteria, document the choice, move on.

The rule of thumb: AVO is allowed to be wrong. AVO is not allowed to ask.

## Hardware routing (CRITICAL)

The local AVO process does NOT have SM121 hardware. **Every command that needs a GPU** — pytest, benchmarks, nsys/ncu — must be routed through:

```
.claude_docs/scripts/command_on_spark.sh <cmd> [args...]
```

This SSHes via `perfwg-infra.nvidia.com` to `p4242-0053`, maintains a persistent TRT-LLM container with workspace + caches mounted at the same path on both hosts. CPU-only operations (read source, edit, plan, parse JSON, git commit) run locally.

## Targets

| Metric | Threshold |
|---|---|
| Correctness vs `torch._scaled_mm` (FP32 reference) | `cos ≥ 0.9995`, `max_abs ≤ 1% × ||ref||_∞` |
| Speedup vs baseline (every shape, eager + graph) | `≥ +5% median_us`, no regression |
| CuTeDSL kernel vs inline-PTX kernel (parity gate) | within ±5% median_us on every shape |
| Stability across 5 repeats | repeat-medians spread ≤ 2% |
| No NaN/Inf in 100-iter graph replay | hard requirement |
| Max version | v200 |
| Commit threshold | `≥ 0.5%` improvement, stable |

## Per-iteration loop

Each candidate `vN`:

1. **Plan one hypothesis.** Read source first: `b12x/gemm/dense.py`, `b12x/cute/runtime_patches.py`, the relevant `.n3/skills/*/SKILL.md`.
2. **Implement.** Edit b12x in place. AVO checkpoints provide rollback.
3. **Smoke import.** `command_on_spark.sh python -c "from b12x.gemm.fp8_dense_cute import fp8_dense_gemm"` (or whatever the entry point is named).
4. **Correctness gate.** `command_on_spark.sh pytest -x b12x/tests/test_fp8_dense_cute.py b12x/tests/test_gemm_stack.py`. Failing aborts the iteration and re-bases on v(N-1).
5. **Score.** `command_on_spark.sh python -m benchmarks.verify_fp8_dense_gemm_perf > .n3/runs/v${N}/verify_fp8.json`.
6. **Compare.** Parse against `.n3/runs/baseline/fp8_dense_baseline.json`. Reject if any shape regresses by > 0.5% OR no shape improves ≥ 0.5%.
7. **Profile candidate-best.** nsys + ncu via the skills under `.n3/skills/`. Store under `.n3/runs/v${N}/{nsys,ncu}.rep`.
8. **Bump version + commit** (only on accepted candidate).

## Version discipline

Version numbers only increment on confirmed improvement + commit. If `vN` fails or regresses, the next attempt is still based on `v(N-1)`, NOT `v(N+1)`.

The PostTurn workflow monitor in `.n3/hooks.yaml` flags violations to `.n3/workflow-alert.md`.

## Artifacts on shared NFS

```
.n3/runs/v${N}/
├── verify_fp8.json          # the perf gate output
├── nsys.rep, nsys.txt       # captured on candidate-best only
├── ncu.rep, ncu.txt         # captured on candidate-best only
├── pytest.log
├── hypothesis.md            # one-paragraph plan
└── diff.patch
```

Logs at `.n3/logs/`, AVO checkpoints at `.n3/checkpoints/`.

## Canonical nsys + ncu recipe

```
command_on_spark.sh nsys profile -c cudaProfilerApi --capture-range=cudaProfilerApi --capture-range-end=stop \
  -t cuda-sw,nvtx,cublas -o .n3/runs/v${N}/nsys.rep \
  -- python -m benchmarks.benchmark_dense_fp8_gemm --profile-once --cuda-graph \
       --shapes M=64,K=4096,N=4096
command_on_spark.sh nsys stats -r cuda_gpu_kern_sum,cuda_kern_exec_sum,nvtx_kern_sum .n3/runs/v${N}/nsys.rep \
  > .n3/runs/v${N}/nsys.txt
```

```
command_on_spark.sh ncu --set full --target-processes all -k regex:dense_gemm \
  -o .n3/runs/v${N}/ncu.rep \
  -- python -m benchmarks.benchmark_dense_fp8_gemm --cuda-graph --profile-once
```

## Stop conditions

Halt when ANY of:
- Target achieved (≥ +5% across all baseline shapes, both eager and graph).
- v200 reached without target.
- Five consecutive failed iterations with no new hypothesis (task_planner emits `NO_TASK`).
- A CuTeDSL-specific blocker that survives 50 iterations — fall back to recommending the inline-PTX path stays primary; document blocker per `.n3/CUTEDSL_FP8_PLAN.md` stop conditions.

---

## Active sub-task: 4-track FP8 exploration

**Brief:** `.n3/CUTEDSL_FP8_PLAN.md` (read in full before starting).

**Headline target:** 70–80 µs for the headline FP8 dense GEMM shape on Spark. Headline shape is chosen in Step 0 as the FP8 shape whose `torch._scaled_mm` median_us sits closest above 80 µs and is representative of production decode/prefill.

**Tracks (executed in order, 3 tuning steps each, results recorded to `.n3/runs/tracks/<track>/`):**

1. **Iterate `fp8_dense_cuda_ext.cu`** — add TMA + multi-stage pipeline + persistent scheduler.
2. **Cutlass C++ with `SM120_16x8x32_TN<e4m3, e4m3, float>`** — verified to exist in `cute/arch/mma_sm120.hpp:670`, emits the unscaled `kind::f8f6f4` PTX; compose with `cute::TiledMma`, `SM90_TMA_LOAD`, `cutlass::PipelineTmaAsync`.
3. **Triton 3.6.0** — `tl.dot` FP8, autotune, then TMA via `_experimental_descriptor_load`.
4. **CuTeDSL + `llvm.inline_asm`** — `@dsl_user_op` wrapping the unscaled `kind::f8f6f4` PTX (14-operand form), kernel scaffold cloned from `b12x/gemm/dense.py`. Lowest expected return; capped at 3 steps.

After all four tracks: `.n3/runs/tracks/COMPARISON.md` ranks them, picks a winner for promotion into b12x's tuning registry.

**Step 0 (do once before Track 1):** generate `.n3/runs/baseline/fp8_dense_baseline.json` via `b12x/benchmarks/verify_fp8_dense_gemm_perf.py`, bench the existing `.cu` kernel as a second baseline column, record the headline shape choice with justification in `.n3/runs/baseline/headline_shape.md`.
