You are running in AUTOPILOT mode (no user in the loop, stdin is /dev/null).

You are starting the cuBLASLt-pipelining-push campaign. Read these in order:

  1. .n3/TASK.md  (this file's local copy is at .n3/runs/campaigns/2026-05-04-cublass-pipelining-push/TASK.md)
  2. .n3/runs/baseline/headline_shape.md
  3. .n3/runs/baseline/fp8_dense_baseline.md
  4. .n3/runs/baseline/nsys_top_kernels.md
  5. .n3/runs/campaigns/2026-05-02-cutlass-collective-deepdive/TASK.md
     and that campaign's tracks/cutlass_collective_tune2.log (so you don't
     re-try configs already swept)

Start with attempt **A1: override EpilogueTile to (32, 32) and unlock TileM=32**.

This is the highest-EV first move because:
- TileM=32 lets the kernel match the headline M exactly (no padding waste).
- The CUTLASS Sm90 warp-specialized epilogue's static_assert
  `EPI_TILE_M | CTA_M` is the only reason TileM=32 doesn't compile today.
  It's a one-line override.
- Frees ~half the smem budget vs TileM=64, opening the path to deeper
  pipelining (Stages=10+) or larger TileK (≥384) at TileM=32.

Concretely:
- Edit b12x/gemm/fp8_dense_cutlass_collective_ext.cu — pass an explicit
  EpilogueTile of cute::Shape<_32, _32> into
  cutlass::epilogue::collective::CollectiveBuilder.
- Rebuild with B12X_CUTLASS_TILE_M=32 B12X_CUTLASS_TILE_N=64
  B12X_CUTLASS_TILE_K=256 B12X_CUTLASS_SCHED_STAGES=8 (start), then sweep
  STAGES ∈ {6, 7, 8, 9, 10}.
- Run bench_cutlass_collective_best.py against each, record results under
  tracks/A1_epitile_override/sweep_<config>.{json,log}.
- If A1 produces ≥ 1% improvement over 180.35 µs, lock the winner and
  proceed to A2 (TileK=384/512 with relaxed dispatch policy).
- If A1 produces no improvement, document the negative result in
  tracks/A1_epitile_override/result.md and proceed to A2.

After every attempt:
- Write tracks/<attempt-name>/result.md with: hypothesis, diff applied,
  config swept, headline median µs, vs-cuBLASLt ratio, decision
  (proceed / abort / lock).
- The bench harness already produces a JSON; copy it into the attempt dir.

Never use AskUserQuestion / ask tools. The user will read JSON outputs,
SUMMARY.md, and git history when the loop terminates.

Stop conditions are documented in TASK.md "## Stop conditions". When any
fires, write tracks/SUMMARY.md, emit NO_TASK, and let the loop terminate.
