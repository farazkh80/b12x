You are running in AUTOPILOT mode (no user in the loop, stdin is /dev/null).

Read these in order, then begin work:
  1. .n3/TASK.md  (especially the "Status" and "Autonomy contract" sections — never use AskUserQuestion / ask tools)
  2. .n3/CUTEDSL_FP8_PLAN.md  (this is the active plan; despite the filename it now describes the 4-track FP8 exploration, not just CuTeDSL)
  3. .n3/SPARK_QUIRKS.md
  4. .n3/runs/baseline/fp8_dense_baseline.md  (shape inventory + baseline measurement schema)
  5. .n3/runs/baseline/nsys_top_kernels.md  (nvjet kernel inventory)

The work is structured as Step 0 → Track 1 → Track 2 → Track 3 → Track 4 → COMPARISON.

Begin with Step 0:
  - Author b12x/benchmarks/verify_fp8_dense_gemm_perf.py per the schema in
    .n3/runs/baseline/fp8_dense_baseline.md.
  - Run on Spark via .claude_docs/scripts/command_on_spark.sh; output to
    .n3/runs/baseline/fp8_dense_baseline.json.
  - Bench the existing b12x/gemm/fp8_dense_cuda_ext.cu kernel on every shape
    (eager + graph) and record into the same JSON under key b12x_inline_ptx_*_us.
  - Pick the headline shape (FP8 shape whose torch._scaled_mm median_us sits
    closest above 80 µs and is representative of production decode/prefill).
    Write justification to .n3/runs/baseline/headline_shape.md.
  - Commit before starting any track.

Then proceed through the tracks in order (T1 → T2 → T3 → T4), 3 tuning steps
each, recording per-step artifacts under .n3/runs/tracks/<track>/step{1,2,3}/
and writing SUMMARY.md after each track. After T4 SUMMARY, write
.n3/runs/tracks/COMPARISON.md.

Performance target: 70–80 µs on the headline shape. Across all baseline shapes,
no regression vs torch._scaled_mm.

For every choice you face: pick, justify in
.n3/runs/tracks/<track>/step<N>/hypothesis.md (or for Step 0,
.n3/runs/baseline/headline_shape.md), and proceed. Never use
AskUserQuestion / ask tools. The user will read JSON outputs, the SUMMARY/
COMPARISON files, and git history when the loop terminates.

Stop conditions are documented in CUTEDSL_FP8_PLAN.md "## Stop conditions"
and TASK.md "## Stop conditions". When any fires, emit NO_TASK and let the
loop terminate.
