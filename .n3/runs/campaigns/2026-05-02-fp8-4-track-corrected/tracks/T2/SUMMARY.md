# Track 2 SUMMARY — CUTLASS/CuTe SM120 atom

## Outcome

Track 2 produced a correct CUTLASS/CuTe atom smoke extension, but it is not a viable performance candidate.

Best correct Track 2 candidate: Step 1 `sm120_cute_atom`, `tile_m=16`, `tile_n=32`, `block=32`.

Headline `(M=32,K=5376,N=5376)`:

| path | eager us | graph us |
| --- | ---: | ---: |
| torch._scaled_mm | 186.304 | 186.304 |
| T2 Step 1 CUTLASS atom smoke | 362.496 | 360.592 |

Correctness: Step 1 passed all 22 verifier shapes. Step 2 and Step 3 topology probes failed smoke correctness and were rejected before profiling. The `186.304 us` row is the slower simultaneous verifier control, not the original promotion reference; against the original `99.408 us` production baseline and ~120 us starting reference, the `362.496 us` candidate is a clear regression.

## Steps

### Step 1 — CUTLASS atom smoke

Implemented `b12x/gemm/fp8_dense_sm120_cute_ext.cu` and `b12x/gemm/fp8_dense_sm120_cute.py`. The kernel uses `cute::SM120_16x8x32_TN<float_e4m3_t,float_e4m3_t,float>` in place of handwritten unscaled MMA while preserving the Track 1 staging/layout.

Artifacts:
- `.n3/runs/tracks/T2/step1/pytest.log`
- `.n3/runs/tracks/T2/step1/verify_fp8.json`
- `.n3/runs/tracks/T2/step1/nsys.nsys-rep`
- `.n3/runs/tracks/T2/step1/nsys.txt`
- `.n3/runs/tracks/T2/step1/ncu.ncu-rep`
- `.n3/runs/tracks/T2/step1/ncu.txt`
- `.n3/runs/tracks/T2/step1/ncu_details.txt`
- `.n3/runs/tracks/T2/step1/result.md`

NCU headline notes: 41.62% memory throughput, 12.04% compute throughput, 0.50 waves/SM, 29.17% theoretical occupancy, 11.37% achieved occupancy, L1TEX scoreboard stalls, uncoalesced global/shared accesses, and shared bank conflicts.

### Step 2 — row-warp enlargement probe

Candidate: `tile_m=32`, `tile_n=32`, `stage_k=256`, `block=128`.

Result: failed smoke correctness on `M=32,K=32,N=32` with 992/1024 mismatched elements and max absolute difference 47.75. Profiling skipped.

### Step 3 — column-warp enlargement probe

Candidate: `tile_m=16`, `tile_n=64`, `stage_k=128`, `block=64`.

Result: failed smoke correctness on `M=16,K=32,N=64` with 1006/1024 mismatched elements and max absolute difference 1880.0. Profiling skipped.

## Decision

Do not carry Track 2 forward. The standalone CUTLASS atom path proves the atom is callable and numerically compatible for a single `16x32` warp tile, but simple CTA topology scaling breaks the assumed fragment layout. The only correct topology is slower than both `torch._scaled_mm` and Track 1 on every important shape.
