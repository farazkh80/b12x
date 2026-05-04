# T4 Step 1 result

Added `f8f6f4_mma_m16n8k32_f32_e4m3` in `b12x/cute/fp4.py` as a CuTeDSL `@dsl_user_op` wrapper for:

```text
mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

Validation:

```text
tests/test_fp8_dense_gemm.py: 7 passed
```

This proves the helper is syntactically loadable without breaking existing FP8 paths. Step 2 will attempt to route the existing CuTeDSL FP8 MMA prototype through this unscaled helper.
