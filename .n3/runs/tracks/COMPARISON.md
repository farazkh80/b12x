# FP8 dense GEMM track comparison

Fixed headline: `M=32,K=5376,N=5376`.

## Summary table

| track | best candidate | headline eager us | headline graph us | correctness | no-regression vs `torch._scaled_mm` | decision |
| --- | --- | ---: | ---: | --- | --- | --- |
| T1 CUDA C inline PTX | Step 2 `TileM=32,TileN=64,StageK=256` | 179.200 | 178.992 | passed | failed graph/full-shape | do not promote; restore original parity floor |
| T2 CUTLASS/CuTe C++ atom | Step 1 `SM120_16x8x32_TN` smoke | 362.496 | 360.592 | passed | failed | do not promote |
| T3 Triton FP8 GEMM | Step 1 minimal `tl.dot` | n/a | n/a | smoke passed | n/a | abort: PTX lowered to FP16 MMA |
| T4 CuTeDSL + `llvm.inline_asm` | Step 3 staged unscaled helper | 206.704 | 207.152 | passed | failed headline | do not promote |

## Winner

No candidate met the `70–80 us` target or the no-regression gate across baseline shapes.

The best measured headline number was T1 Step 2 at `179.200 us` eager, but it is not promoted because graph timing was slightly slower than the simultaneous `torch._scaled_mm` baseline and full-shape no-regression failed. The code was restored to the original inline-PTX hot path as the parity floor.

## Evidence by track

- T1: `.n3/runs/tracks/T1/SUMMARY.md`
  - Best headline: `179.200 us` eager, `178.992 us` graph.
  - Bottleneck: latency/occupancy limited with memory-pipeline underutilization.
- T2: `.n3/runs/tracks/T2/SUMMARY.md`
  - CUTLASS atom was correct but slow: `362.496 us` eager on headline.
  - Larger row/column topology probes failed correctness.
- T3: `.n3/runs/tracks/T3/SUMMARY.md`
  - Triton accepted FP8 tensors and was correct on smoke, but PTX emitted `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, not `kind::f8f6f4` or `kind::mxf8f6f4`.
- T4: `.n3/runs/tracks/T4/SUMMARY.md`
  - CuTeDSL unscaled inline asm works and passed tests, but staged headline was `206.704 us` eager.

## Promotion recommendation

Do not promote a new kernel from these tracks. Keep the existing CUDA inline-PTX implementation as the empirical parity floor. The next productive optimization direction is not another local tile tweak; it is a real TMA/warp-specialized mainloop or a production CUTLASS collective path that can keep enough CTAs resident while avoiding repeated global/shared-memory staging overhead.
