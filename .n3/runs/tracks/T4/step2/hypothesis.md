# T4 Step 2 hypothesis

Route the non-staged CuTeDSL FP8 MMA prototype through the new unscaled `kind::f8f6f4` helper. This removes the `mxf8f6f4` unit-scale operands and should match the CUTLASS C++ atom semantics while preserving the existing CuTeDSL fragment layout.

Validation: run `tests/test_fp8_dense_gemm.py`; this covers the non-staged 16x32x32 CuTeDSL MMA path.
