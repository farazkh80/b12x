# T4 Step 2 result

Routed the non-staged CuTeDSL FP8 MMA prototype through `f8f6f4_mma_m16n8k32_f32_e4m3`.

Validation:

```text
tests/test_fp8_dense_gemm.py: 7 passed
```

This validates the unscaled helper on the small CuTeDSL MMA path.
