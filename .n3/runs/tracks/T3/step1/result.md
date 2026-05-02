# T3 Step 1 result

Candidate: minimal Triton FP8 GEMM using `tl.dot`.

Smoke correctness passed exactly on `M=16,K=32,N=64`:
- `max_abs=0.0`
- `mean_abs=0.0`
- `cos=1.0`

PTX lowering failed the native-FP8 requirement. The compiled PTX did not contain `kind::f8f6f4` or `kind::mxf8f6f4`; it emitted FP16 tensor-core instructions, for example:

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

Decision: abort Track 3 per plan. Step 2 autotune and Step 3 TMA would optimize a fallback FP16-MMA path rather than native SM120 FP8 MMA.
