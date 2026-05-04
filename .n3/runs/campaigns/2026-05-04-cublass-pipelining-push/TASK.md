# cuBLASLt Pipelining Push — Beat cuBLASLt by More Than 2%

## One-line goal

Take the FP8 dense GEMM at the headline shape `(M=32, K=5376, N=5376)` from
the current `(64, 64, 256, 8)` config (180.35 µs, +2.15% faster than
`torch._scaled_mm`) to **as close to the 108 µs DRAM-bound floor as the
hardware allows**.

## Where we are (concrete starting state)

| metric | value | source |
|---|---:|---|
| `torch._scaled_mm` (cuBLASLt nvjet) | **184.32 µs** | clean L2-flushed bench, this campaign's `tracks/bench_cutlass_collective_best.py` |
| `nvjet_sm121_qqtst_mma_64x64x128_6_16x32x128_tmaAB_bz_TNNN` (the kernel cuBLASLt picks) | 100% of the GPU time | `tracks/probe_headline_nvjet.py` capture |
| **cutlass collective `(64, 64, 256, 8)`** | **180.35 µs** ← current best | this campaign |
| achieved DRAM bandwidth at 180 µs | 161 GB/s | 29.4 MB / 180 µs |
| DRAM peak on Spark / GB10 | 273 GB/s | published spec |
| **DRAM-bound floor for this shape** | **108 µs** | 29.4 MB / 273 GB/s |
| **headroom above floor** | **~72 µs** of pipeline / issue / sync overhead | |

So we're at 60.8% of peak DRAM. The remaining 72 µs is exposed pipeline
overhead — that's the optimization budget.

## Headline shape (FIXED — do not re-pick)

`M=32, K=5376, N=5376`, FP8 e4m3 × FP8 e4m3 → bf16, fp32 accumulate, TN
canonical layout (A K-major, B K-major, output M-major), per-tensor unit scales.
This is the production nsys profile shape and the hand-tuned point of
`b12x/gemm/fp8_dense_cuda_ext.cu`.

## Constraints already learned (don't re-discover)

1. **Public CUTLASS Collective hits a smem-budget wall** at TileK=384, TileK=512,
   and TileM=128 with the existing scaffold — `SchedulerPipelineStageCount`
   collapses to 1 and the `Stages >= 2` static_assert fires. To push past, the
   `.cu` source must override `KernelTmaWarpSpecializedPingpongSm120` to a more
   generous dispatch policy, OR write a custom `CollectiveBuilder` arguments
   set, OR drop to a CUTLASS 2.x kernel.

2. **TileM=32 fails** because the Sm90 warp-specialized epilogue requires
   `EPI_TILE_M | CTA_M`, default EPI_TILE_M=64 doesn't divide 32. Override
   `EpilogueTile` to `cute::Shape<_32, _32>` to allow TileM=32 — that recovers
   half the smem budget and eliminates M-padding waste at this shape.

3. **MMA atom is `SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>`**
   (cute/arch/mma_sm120.hpp:670). Emits `kind::f8f6f4` PTX. Block-scaled
   `kind::mxf8f6f4` exists too but cutlass C++ doesn't expose an unscaled-only
   wrapper for it. nvjet uses block-scaled with degenerate scales.

4. **CuTeDSL Python is dead-end** for native FP8 atoms on sm_120. Don't
   re-attempt it — the previous campaign's T4 SUMMARY documents this.

5. **Triton 3.6 silently upcasts FP8 → FP16** for `tl.dot` on sm_120. Don't
   re-attempt. Documented in T3 SUMMARY.

## Specific avenues to push (ranked by expected impact)

### Tier 1 — directly attack pipeline overhead (high expected value)

**A1. Override `EpilogueTile` to `<_32, _32>` and unlock TileM=32.**

  - Modify `b12x/gemm/fp8_dense_cutlass_collective_ext.cu` to pass
    `EpilogueTile = cute::Shape<_32, _32>` into the
    `cutlass::epilogue::collective::CollectiveBuilder` template.
  - Then sweep `(32, 64, 256, S)` for S in {3..10}. M=32 is exactly the
    headline M, so no padding waste. Smem budget should accommodate
    deeper pipelining.
  - **Expected**: 2-5% additional gain over current best.

**A2. Try larger TileK by overriding the dispatch policy.**

  - The current build dies at TileK=384/512 because the default Sm120
    pingpong dispatch policy reserves smem for too many epilogue stages.
  - Reduce epilogue stages (`StagesC`) or shrink `SmemLayoutAtomD` to free
    budget for TileK=384/512 with Stages=2.
  - **Expected**: large K-tile means fewer K-iterations (10 instead of 21),
    fewer mbarrier waits. 1-3% gain.

**A3. Cluster shape `(2, 1, 1)` — multi-CTA TMA load broadcast.**

  - `MainloopSm120TmaWarpSpecialized` accepts a `ClusterShape` template arg.
  - With cluster (2,1,1), 2 CTAs in the M dim share TMA loads of B (the
    big matrix) via SM-to-SM async copy. Cuts B's effective DRAM load
    in half for those CTAs.
  - **Expected**: 2-8% gain if it works on sm_121 (Hopper-class clustering;
    untested on Spark).

**A4. Persistent + work-stealing tile scheduler.**

  - Currently the kernel launches `ceil(M/TileM) × ceil(N/TileN) = 1 × 84
    = 84 CTAs`. With a persistent scheduler and N_persistent ≈ num_SMs,
    each CTA drains a work queue, kernel launch overhead amortized once.
  - cute::gemm::PersistentTileScheduler exists; need to wire it into
    the Collective gemm.
  - **Expected**: 1-3% on this shape (84 CTAs / 48 SMs ≈ 1.75 waves —
    persistent helps tail tiles).

### Tier 2 — shave overhead inside the existing scaffold (medium value)

**B1. Sweep epilogue policy variants.**

  - `cutlass::epilogue::collective::CollectiveBuilder` has multiple policies:
    `NoSmemWarpSpecialized`, `TmaWarpSpecialized`, `TmaWarpSpecializedCooperative`,
    `TmaWarpSpecializedPingpong`. Default is pingpong; cooperative may handle
    small-M better.
  - **Expected**: 0.5-2% if the default is suboptimal for this M.

**B2. Fuse alpha scaling into mainloop.**

  - Current epilogue applies `alpha = scale_a * scale_b * 1.0` after MMA
    accumulation. Fusing it into the K-loop's last iteration removes one
    pass over the accumulator registers.
  - **Expected**: <1% on this shape (epilogue is tiny vs mainloop).

**B3. ncu-driven micro-optimization.**

  - Run ncu --set full on the current best, identify top 3 stall reasons,
    address each. Likely stalls: mbarrier waits, smem bank conflicts on
    register loads, register spills.
  - **Expected**: 0.5-3% per addressable stall.

### Tier 3 — outside the box (high variance)

**C1. Stream-K decomposition.**

  - Split each (M=32, N=64) tile across the K dim so multiple CTAs work
    on the same output tile and reduce-add at the end. Better load
    balancing on shapes where M*N tile count < num_SMs.
  - **Expected**: variable — depends on whether per-iteration overhead from
    reduction outweighs the parallelism gain at M=32.

**C2. Direct `mma.sync.kind::mxf8f6f4 .block_scale` PTX.**

  - Match nvjet's exact PTX path (block-scaled with unit `0x7F` scales)
    instead of cutlass's unscaled `kind::f8f6f4`. Currently `b12x/gemm/
    fp8_dense_cuda_ext.cu` uses this PTX path and hits ~109 µs hot-cache.
  - **Expected**: probably no gain — cuBLASLt also uses this path and the
    bottleneck isn't the MMA op.

**C3. Custom CUTLASS 2.x kernel.**

  - Drop the Collective abstraction and write the mainloop directly using
    raw cute primitives: TMA loads, mbarrier-managed pipeline, manual
    register staging, custom epilogue.
  - Most flexibility, biggest implementation cost.
  - **Expected**: research-grade. Could be net negative if the public
    primitives are tuned worse than the Collective wrapper.

## Validation gates (every step, every config)

| gate | threshold |
|---|---|
| Correctness vs `torch._scaled_mm` (FP32 reference) | `cos ≥ 0.9995`, `max_abs ≤ 1% × ‖ref‖_∞` |
| **Strict no-regression vs `torch._scaled_mm`** | every shape in `runs/baseline/fp8_dense_baseline.json`, eager + graph |
| L2-flushed cold-cache median (this shape) | record AND beat 180.35 µs |
| Repeat-median spread (5 repeats × 80+ iters) | ≤ 2% |
| 100-iter graph replay finite | hard requirement |

The bench harness `tracks/bench_cutlass_collective_best.py` already produces
all of these. Use it. Don't re-implement timing.

## Per-step recording (under `runs/campaigns/2026-05-04-cublass-pipelining-push/tracks/`)

```
tracks/
├── bench_cutlass_collective_best.py     ← already in tree; keep
├── probe_headline_nvjet.py              ← already in tree; keep
├── A1_epitile_override/                 ← one dir per attempt
│   ├── hypothesis.md
│   ├── diff.patch
│   ├── verify_fp8.json
│   ├── ncu.txt                          ← only on candidate-best
│   ├── nsys.txt                         ← only on candidate-best
│   └── result.md
├── A2_tilek_512/
├── A3_cluster_2_1_1/
└── ...
```

After 5 productive attempts (or AVO judgment that further iteration is
hitting diminishing returns), write `tracks/SUMMARY.md` and update
`meta.json`'s `outcome_summary` field.

## Stop conditions

Halt when ANY of:
- Headline measured ≤ 130 µs (within 20% of DRAM floor) AND no-regression on every other shape AND stable spread ≤ 2%. **This is the ship-it line.**
- Headline measured ≤ 150 µs AND no-regression. Promote, but document remaining headroom.
- 15 attempts with no further improvement over 180.35 µs. Document, ship the existing best, and write a "limits of public CUTLASS on sm_121" lessons-learned note in `tracks/SUMMARY.md`.

## Autonomy contract (CRITICAL when running autopilot)

When invoked in autopilot mode (no `--chat`, with `task_planner` driving),
AVO operates without a user in the loop.

1. **Do NOT use `AskUserQuestion` / `ask`.** Pick the option with highest
   expected information gain, write a one-paragraph justification to the
   attempt's `hypothesis.md`, proceed.
2. **Never block on confirmation.** If a perf gain looks suspicious, run
   verification a second time at higher repeat count rather than asking.
3. **Hard blockers** (CUDA driver dead, repo corrupted, API key revoked) are
   the only legitimate stop conditions. Append to `.n3/workflow-alert.md`
   and emit `NO_TASK` from task_planner.
4. **Ambiguity in the brief** → pick the interpretation that best matches
   the success criteria, document the choice, move on.

## Hardware routing (reminder)

The local AVO process does NOT have a GPU. Every command that needs a GPU
(pytest, benchmarks, nsys, ncu, the JIT-compiled extension build) must be
routed through:

```
.claude_docs/scripts/command_on_spark.sh <cmd> [args...]
```

CPU-only operations (read source, edit, plan, parse JSON, git commit) run
locally on the AVO host.

## Pointers (read in order)

1. This file.
2. `b12x/.n3/runs/baseline/headline_shape.md` — the fixed headline target.
3. `b12x/.n3/runs/baseline/fp8_dense_baseline.json` — full no-regression table.
4. `b12x/.n3/runs/campaigns/2026-05-04-cublass-pipelining-push/tracks/bench_cutlass_collective_best.py`
   — the bench harness AVO uses.
5. `b12x/gemm/fp8_dense_cutlass_collective_ext.cu` and `.py` — the kernel
   to modify.
6. Previous campaign summaries under `runs/campaigns/2026-05-02-*/` for prior
   art and what's already been tried.
