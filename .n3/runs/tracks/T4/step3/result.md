# T4 Step 3 result

Routed the staged CuTeDSL FP8 MMA prototype through `f8f6f4_mma_m16n8k32_f32_e4m3`.

Validation:

```text
tests/test_fp8_dense_gemm.py: 7 passed
```

Headline smoke benchmark with `warmup=5,iters=10,repeats=3`:

| path | eager us | graph us |
| --- | ---: | ---: |
| torch._scaled_mm | 183.264 | 185.712 |
| CuTeDSL unscaled staged | 206.704 | 207.152 |

Correctness passed: `cos=0.99999857`, `max_abs=0.01623`, limit `0.05216`.

Decision: correct but slower than `torch._scaled_mm`; do not promote.
