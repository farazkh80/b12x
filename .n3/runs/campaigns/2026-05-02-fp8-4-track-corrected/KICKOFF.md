You are running in AUTOPILOT mode (no user in the loop, stdin is /dev/null).

Read these in order, then begin work:
  1. .n3/TASK.md  (especially the "Status" and "Autonomy contract" sections — never use AskUserQuestion / ask tools)
  2. .n3/CUTEDSL_FP8_PLAN.md  (this is the active plan; despite the filename it now describes the 4-track FP8 exploration, not just CuTeDSL)
  3. .n3/SPARK_QUIRKS.md
  4. .n3/runs/baseline/fp8_dense_baseline.md  (shape inventory + baseline measurement schema)
  5. .n3/runs/baseline/nsys_top_kernels.md  (nvjet kernel inventory)

The work is structured as Step 0 → Track 1 → Track 2 → Track 3 → Track 4 → COMPARISON.

Step 0 is DONE: baseline JSON exists at
  .n3/runs/baseline/fp8_dense_baseline.json and headline_shape.md has been
  authoritatively set to:

      *** HEADLINE SHAPE: (M=32, K=5376, N=5376) ***

This is FIXED. Do NOT re-pick a different headline. It is the shape captured in
the production nsys profile (profile_super_nvfp4_disable_b12x.1.nsys-rep), the
shape b12x/gemm/fp8_dense_cuda_ext.cu was hand-tuned for, and the only shape
where the existing inline-PTX kernel is competitive (102.464 µs vs cuBLASLt
99.408 µs per b12x/README.md tuning notes).

Earlier autopilot iterations picked (M=1, K=4096, N=4096) by following an
outdated "closest to 80 µs" rule and produced wrong-shape results in
.n3/runs/tracks/T1/step{1,2}/. Those results target a shape the kernel was
never tuned for and are archived rather than deleted. **Restart Track 1 from
step 1 against the correct headline shape.**

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
