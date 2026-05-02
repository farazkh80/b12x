# FP8 dense GEMM track comparison

Fixed headline: `M=32,K=5376,N=5376`.

## Reference baseline correction

The promotion reference is the original headline reference from `.n3/runs/baseline/headline_shape.md`, not the slower simultaneous `torch._scaled_mm` measurements observed during later verifier runs.

Original headline references:

| reference | median us |
| --- | ---: |
| Production cuBLASLt / `torch._scaled_mm` | 99.408 |
| Existing b12x inline PTX parity floor | 102.464 |
| User-recalled starting reference | ~120 |
| Stretch target | 70-80 |

The later 178-186 us `torch._scaled_mm` values are only per-run simultaneous controls. They should not be used as the promotion baseline.

## Summary table

| track | best candidate | measured headline eager us | measured headline graph us | vs original 99.408 us | vs ~120 us | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T1 CUDA C inline PTX | Step 2 `TileM=32,TileN=64,StageK=256` | 179.200 | 178.992 | +80.3% slower | +49.3% slower | do not promote |
| T2 CUTLASS/CuTe C++ atom | Step 1 `SM120_16x8x32_TN` smoke | 362.496 | 360.592 | +264.7% slower | +202.1% slower | do not promote |
| T3 Triton FP8 GEMM | Step 1 minimal `tl.dot` | n/a | n/a | n/a | n/a | abort: PTX lowered to FP16 MMA |
| T4 CuTeDSL + `llvm.inline_asm` | Step 3 staged unscaled helper | 206.704 | 207.152 | +107.9% slower | +72.3% slower | do not promote |

## Winner

No candidate met the `70-80 us` target, the original `99.408 us` production baseline, the existing `102.464 us` inline-PTX parity floor, or the user-recalled ~120 us starting reference.

The best measured headline number was T1 Step 2 at `179.200 us` eager. It was only near parity against a slower simultaneous verifier control and is not a valid improvement against the original reference. It also failed the graph/full-shape no-regression gate, so it is not promoted.

## Evidence by track

- T1: `.n3/runs/tracks/T1/SUMMARY.md`
  - Best headline: `179.200 us` eager, `178.992 us` graph.
  - Worse than original `99.408 us` production reference and worse than ~120 us starting reference.
  - Bottleneck: latency/occupancy limited with memory-pipeline underutilization.
- T2: `.n3/runs/tracks/T2/SUMMARY.md`
  - CUTLASS atom was correct but slow: `362.496 us` eager on headline.
  - Larger row/column topology probes failed correctness.
- T3: `.n3/runs/tracks/T3/SUMMARY.md`
  - Triton accepted FP8 tensors and was correct on smoke, but PTX emitted `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, not `kind::f8f6f4` or `kind::mxf8f6f4`.
- T4: `.n3/runs/tracks/T4/SUMMARY.md`
  - CuTeDSL unscaled inline asm works and passed tests, but staged headline was `206.704 us` eager.
  - Worse than original `99.408 us` production reference and worse than ~120 us starting reference.

## Promotion recommendation

Do not promote a new kernel from these tracks. Keep the existing CUDA inline-PTX implementation as the empirical parity floor. The next productive optimization direction is not another local tile tweak; it is a real TMA/warp-specialized mainloop or a production CUTLASS collective path that can keep enough CTAs resident while avoiding repeated global/shared-memory staging overhead.
