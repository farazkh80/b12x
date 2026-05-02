# T4 Step 3 hypothesis

Route the staged CuTeDSL FP8 MMA prototype through the new unscaled `kind::f8f6f4` helper as well. This is the closest CuTeDSL equivalent to the Track 1 hot path because it uses 128 threads, staged K, and the production-shaped staged kernel when `M,N` are multiples of 32 and `K` is a multiple of 256.

Validation: run `tests/test_fp8_dense_gemm.py`, which includes staged/layout regression coverage.
