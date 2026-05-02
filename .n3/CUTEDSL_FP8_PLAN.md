# FP8 Dense GEMM — Multi-Track Exploration Plan

> Replaces the previous "CuTeDSL only" framing. The corrected ground-truth analysis (see `## Ground truth` below) made it clear no single approach is the obvious winner, so AVO will explore four tracks in order, give each a 3-step tuning budget, and record results before moving on.
>
> **Performance target:** the headline FP8 dense GEMM shape (defined in Step 0 below) runs in **70–80 µs** on Spark (sm_121). Across all baseline shapes, no regression vs `torch._scaled_mm`; on the headline shape, push as close to 70 µs as the hardware allows.

## Ground truth (verified at source on Spark, 2026-05-02)

The previous AVO session burned itself out on a CuTeDSL FP8 path that doesn't exist. What actually exists:

1. **Existing `b12x/gemm/fp8_dense_cuda_ext.cu`** uses sm_120's block-scaled FP8 mma instruction with degenerate unit scales:
   ```ptx
   mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
   ```
   16 operands, two `0x7f7f7f7f` packed-`ue8m0` scale registers feeding "scale = 1.0" with zero selectors. This is empirically known to compile and run on sm_121.

2. **The "old" sm_89 FP8 mma** (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, no `kind::` prefix) is **gone on sm_120**. The `kind::` prefix is mandatory.

3. **Cutlass C++ has `SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>`** at `cute/arch/mma_sm120.hpp:670`, gated on `CUTE_ARCH_F8F6F4_MMA_ENABLED`. It emits the **unscaled** sm_120 FP8 mma (14-operand form):
   ```ptx
   mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
   ```
   Equivalent to the existing `.cu` kernel's instruction modulo the (degenerate) scale operands.

4. **Cutlass-DSL Python (4.3.4)** has zero FP8 entries in `cute/nvgpu/warp/`. FP8 atoms in cutlass-dsl exist only under `warpgroup/` (sm_90a-only Python check) and `tcgen05/` (sm_100f/sm_110f-only). The only Python-level mechanism for unsupported PTX is `llvm.inline_asm`.

5. **Triton 3.6.0** is in the container. `tl.dot` with FP8 dtypes lowers on sm_120 — the actual emitted PTX (unscaled `kind::f8f6f4` vs block-scaled `kind::mxf8f6f4`, vs falling back) is empirical and must be inspected per-track.

So we have **one verified-working PTX form** (`kind::mxf8f6f4` with unit scales, used by the existing `.cu`) and **one ready-made cutlass C++ atom** (`SM120_16x8x32_TN`, unscaled `kind::f8f6f4`). Everything else is a wrapper choice.

## Step 0 — Baseline + headline-shape selection (do this once, then start Track 1)

Before any track starts:

1. **Generate the baseline JSON.** Author `b12x/benchmarks/verify_fp8_dense_gemm_perf.py` per the schema in `.n3/runs/baseline/fp8_dense_baseline.md`. Run on Spark. Output to `.n3/runs/baseline/fp8_dense_baseline.json`. This is the parity floor every track grades against.
2. **Bench the existing `fp8_dense_cuda_ext.cu` kernel as a second baseline column** for every shape (eager + graph). Record into the same JSON under key `b12x_inline_ptx_*_us`.
3. **Pick the headline shape.** Choose the FP8 shape whose `torch._scaled_mm` median_us is *closest above* 80 µs and is representative of production decode/prefill. Record in `.n3/runs/baseline/headline_shape.md` along with one paragraph of justification. This shape is what every track's "push to 70-80 µs" benchmarking looks at.
4. Commit `verify_fp8_dense_gemm_perf.py` and the two new baseline files. **No track starts until Step 0 is committed.**

## Track structure (applies to T1–T4)

Each track gets exactly 3 tuning steps, then a write-up, then we move to the next track.

| Step | Goal | Gate |
|---|---|---|
| **Step 1: Working** | Smallest end-to-end kernel that produces correct output for the headline shape. Atom microbench passes if the path has one. | `cos ≥ 0.9995` vs `torch._scaled_mm`; runs without launch error |
| **Step 2: Tuned-nominal** | Autotune meta-parameters (BLOCK sizes, num_stages, num_warps, atom_layout, swizzle, persistent on/off). Pick the winning config. | Beats `torch._scaled_mm` on the headline shape |
| **Step 3: Push limits** | One targeted optimization based on Step 2's nsys/ncu profile (e.g. TMA prefetch depth, register-pressure trim, sub-tile epilogue, persistent scheduler tweaks, work-stealing). | As close to 70-80 µs as possible; document why we stopped |

After Step 3 of each track:

```
.n3/runs/tracks/<track>/SUMMARY.md
```

Must contain: best µs on the headline shape, µs-per-shape table for the full baseline, nsys-attributed bottleneck (memory-bound / compute-bound / latency-hidden), gap-to-target (`best_us - 70`), and one paragraph "what we'd try if given a 4th step." Then the track is closed.

After all four tracks finish:

```
.n3/runs/tracks/COMPARISON.md
```

A side-by-side ranked table across tracks, the winning kernel chosen for promotion into b12x's tuning registry, and decisions on which (if any) other track is worth a follow-up sprint.

## Per-step recording (under each track)

```
.n3/runs/tracks/<track>/step<N>/
├── verify_fp8.json     # full per-shape table
├── nsys.rep, nsys.txt  # always captured
├── ncu.rep, ncu.txt    # captured on Step 2 and Step 3
├── pytest.log
├── hypothesis.md       # one-paragraph plan for this step
├── diff.patch          # exactly what changed
└── result.md           # what happened, headline µs, decision to proceed/abort
```

`.n3/runs/tracks/<track>/SUMMARY.md` and `.n3/runs/tracks/COMPARISON.md` are the only files AVO carries forward; everything else stays per-step.

---

## Track 1 — Iterate the existing CUDA C extension (`fp8_dense_cuda_ext.cu`)

**Why first:** lowest-risk, already-passes-correctness baseline. We learn what the headline shape's true ceiling is before betting on more exotic frameworks.

- **Step 1**: instrument current kernel — capture nsys/ncu, identify dominant cost (likely cp.async + insufficient pipelining). Bench unmodified.
- **Step 2**: replace `cp.async`-style loads with TMA via cute_2x primitives (`cute::SM90_TMA_LOAD`), keep the existing inline-asm mma. Software-pipeline 2 → 3+ stages. Autotune `(TileM, TileN, TileK, num_stages)`.
- **Step 3**: add a persistent + work-stealing tile scheduler. Tighten epilogue (sub-tile bf16 store with shared-memory swizzle to avoid bank conflicts). Trim register usage to lift occupancy if ncu shows < 50%.

PTX path stays `kind::mxf8f6f4` block-scaled with unit scales (the existing form). No PTX change.

## Track 2 — Cutlass C++ with `SM120_16x8x32_TN` atom

**Why second:** the atom is ready-made and exercises the full cutlass infrastructure (`cute::TiledMma`, `CollectiveBuilder`, `PipelineTmaAsync`, persistent scheduler).

- **Step 1**: smoke-build a hello-world `fp8_dense_sm120_cute.cu` using `MMA_Atom<SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>>`, `SM90_TMA_LOAD`, `cutlass::PipelineTmaAsync`. Sanity-check `CUTE_ARCH_F8F6F4_MMA_ENABLED` is defined under `-arch=sm_120`. Single-tile correctness.
- **Step 2**: full GEMM with `CollectiveBuilder`. Autotune over tile sizes, cluster shape (try `(1,1,1)` first per the b12x NVFP4 lesson), num_stages, schedule kind (`KernelTmaWarpSpecialized` etc.).
- **Step 3**: pick the best Step-2 config, push it via custom epilogue (vectorized bf16 store, fused alpha scaling), or split-K if the headline shape is K-heavy. Examine ncu's pipeline utilization — if <70%, raise stages; if >90%, raise tile.

PTX path: unscaled `kind::f8f6f4`.

## Track 3 — Triton FP8 GEMM

**Why third:** highest dev velocity. Even if it doesn't win, it's a quick sanity check on whether `tl.dot` on sm_120 emits competitive PTX. Triton 3.6.0 is already installed.

- **Step 1**: minimal Triton FP8 GEMM kernel using `tl.dot` with `tl.float8e4m3`. Verify the generated PTX includes a `kind::f8f6f4` or `kind::mxf8f6f4` mma instruction (`triton-opt --target=cuda:120 --emit-ptx ...` or extract via `kernel.asm["ptx"]`). If Triton falls back to FP16 unpacking, abort the track and document.
- **Step 2**: standard Triton autotune over `(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)`, full grid persistent.
- **Step 3**: TMA via `tl._experimental_descriptor_load` (Triton's TMA path; works on sm_90+ in 3.5+). Compare with non-TMA. If Triton's TMA on sm_120 is unstable, drop back to Step 2's best config and document.

## Track 4 — CuTeDSL + `llvm.inline_asm`

**Why last:** lowest expected return per the analysis. Run it anyway — if it surprises us, fine; if it doesn't, the 3-step budget caps the cost.

- **Step 1**: `@dsl_user_op` Python helper wrapping the unscaled `kind::f8f6f4` PTX (14-operand form, constraint string `"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"`). Single-tile microbench.
- **Step 2**: clone `b12x/gemm/dense.py` (NVFP4) into `b12x/gemm/fp8_dense_cute.py`, drop scale-factor tensors, manually manage register fragments around the helper, keep `PipelineTmaAsync` + persistent scheduler. Autotune tile sizes.
- **Step 3**: same scheduler/pipeline tweaks as Track 1 Step 3, just expressed in CuTeDSL.

If Step 1's microbench can't hit `cos ≥ 0.9995` after the budgeted iterations, document the constraint-string blocker and skip Step 2/3.

---

## Stop conditions (per track and overall)

Per track:
- 3 steps used → write SUMMARY.md and move on (whether or not the target was hit).
- Step 1 fails to produce correct output after a reasonable attempt → document blocker in SUMMARY.md and skip remaining steps in this track.
- Track unexpectedly hits ≤ 70 µs at Step 2 → still do Step 3 to push further; mark in SUMMARY whether Step 3 helped or regressed.

Overall:
- All 4 tracks finished → write COMPARISON.md, propose the winning kernel for promotion, halt.
- Any track produces a kernel within 5% of the 70 µs target across **all** baseline shapes (not just the headline) before all 4 tracks finish → still finish remaining tracks because the comparison is part of the deliverable, but the bar for Step 3 in remaining tracks becomes "is there any chance of beating the leader, with one targeted change?" If not, abort that track's Step 3 and write SUMMARY.

## Validation gates that apply to every step in every track

| Gate | Threshold |
|---|---|
| Correctness vs `torch._scaled_mm` | `cos ≥ 0.9995`, `max_abs ≤ 1% × ‖ref‖_∞` on every shape in the baseline |
| No regression vs `torch._scaled_mm` | every shape, eager + graph |
| Stability across 5 repeats | repeat-medians spread ≤ 2% |
| No NaN/Inf in 100-iter graph replay | hard requirement |

## Implementation hygiene

- `b12x/gemm/fp8_dense_cuda_ext.cu` is **never deleted or regressed**; it stays as the parity floor and as the empirical PTX reference.
- Each track's kernel goes in its own file; the shared verifier (`verify_fp8_dense_gemm_perf.py`) takes a `--kernel` flag selecting which one to score.
- All build/run commands use `command_on_spark.sh`. CPU-only operations run locally.
- Per the autonomy contract in TASK.md, AVO does **not** ask the user between steps. If a step needs a judgment call, AVO writes the choice and reasoning into `step<N>/hypothesis.md` and proceeds.
