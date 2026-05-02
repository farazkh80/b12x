# Track 4 SUMMARY — CuTeDSL + llvm.inline_asm

## Outcome

Track 4 produced a working CuTeDSL unscaled SM120 FP8 MMA helper and routed the existing CuTeDSL FP8 MMA prototypes through it. Correctness tests passed, but the headline smoke benchmark is slower than `torch._scaled_mm`, so this track is not promoted.

## Best T4 candidate

Candidate: staged CuTeDSL FP8 MMA with `f8f6f4_mma_m16n8k32_f32_e4m3`.

Headline `(M=32,K=5376,N=5376)`, smoke benchmark with `warmup=5,iters=10,repeats=3`:

| path | eager us | graph us |
| --- | ---: | ---: |
| torch._scaled_mm | 183.264 | 185.712 |
| CuTeDSL unscaled staged | 206.704 | 207.152 |

Correctness: passed with `cos=0.99999857`, `max_abs=0.01623`, limit `0.05216`.

## Steps

### Step 1 — helper smoke

Added `f8f6f4_mma_m16n8k32_f32_e4m3` in `b12x/cute/fp4.py` using `llvm.inline_asm` and the unscaled 14-operand `kind::f8f6f4` instruction.

Validation: `tests/test_fp8_dense_gemm.py` passed.

### Step 2 — non-staged prototype

Routed the non-staged CuTeDSL FP8 MMA path in `b12x/gemm/fp8_dense.py` to the unscaled helper.

Validation: `tests/test_fp8_dense_gemm.py` passed.

### Step 3 — staged prototype and headline smoke

Routed the staged CuTeDSL FP8 MMA path to the unscaled helper and measured the fixed headline shape.

Validation: `tests/test_fp8_dense_gemm.py` passed; headline correctness passed.

## Decision

Do not promote Track 4. The unscaled inline-asm helper works and is useful as a CuTeDSL reference, but the current CuTeDSL staged prototype is slower than the PyTorch baseline and slower than Track 1. A competitive CuTeDSL path would need a larger rewrite around TMA/persistent scheduling rather than just changing the MMA helper.
