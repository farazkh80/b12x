# Track 3 SUMMARY — Triton FP8 GEMM

## Outcome

Track 3 stopped at Step 1. Triton's `tl.dot` accepted FP8 tensors and produced correct output for a smoke shape, but it did not lower to native SM120 FP8 MMA.

The generated PTX contained FP16 tensor-core MMA:

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

It did not contain either required native-FP8 instruction form:
- `kind::f8f6f4`
- `kind::mxf8f6f4`

Per the Track 3 plan, Step 2 autotuning and Step 3 TMA exploration were skipped because they would tune a fallback FP16-MMA path rather than the requested FP8 path.

## Step status

| step | status | notes |
| --- | --- | --- |
| Step 1 | correctness pass, PTX lowering fail | `max_abs=0.0`, `cos=1.0`, but PTX uses FP16 MMA |
| Step 2 | skipped | autotune not applicable after Step 1 lowering fail |
| Step 3 | skipped | TMA not applicable without viable native-FP8 Triton kernel |

## Artifacts

- `.n3/runs/tracks/T3/step1/pytest.log`
- `.n3/runs/tracks/T3/step1/verify_fp8.json`
- `.n3/runs/tracks/T3/step1/diff.patch`
- `.n3/runs/tracks/T3/step1/result.md`
- `.n3/runs/tracks/T3/step2/result.md`
- `.n3/runs/tracks/T3/step3/result.md`

## Decision

Do not carry Track 3 forward. Native-FP8 lowering is the blocker, not tile tuning.
