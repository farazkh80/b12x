# T4 Step 1 hypothesis

Add a CuTeDSL `@dsl_user_op` helper for the unscaled SM120 FP8 MMA instruction:

```text
mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

First compile smoke only, without routing the production CuTeDSL FP8 kernels to it. The validation target is that existing FP8 tests still pass, proving the helper is syntactically loadable and does not break existing CuTeDSL paths. If direct integration into `fp8_dense_gemm_mma` is needed, it will be attempted after this smoke.
