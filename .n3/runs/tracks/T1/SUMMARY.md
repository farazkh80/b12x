# Track 1 SUMMARY — CUDA C inline PTX

## Best headline result

Best measured Track 1 headline result was Step 2 (`TileM=32, TileN=64, StageK=256`):

| Metric | torch._scaled_mm | Track 1 best |
|---|---:|---:|
| Headline eager median | 180.336 us | 179.200 us |
| Headline graph median | 178.000 us | 178.992 us |
| Correctness | passed | passed |

This is parity-level only against the slower simultaneous verifier baseline: eager was 0.6% faster than that control, graph was 0.6% slower, and full-shape no-regression failed. It is not parity against the original headline reference (`99.408 us` production cuBLASLt / `102.464 us` existing inline PTX, with user-recalled starting reference ~120 us). Against those references, `179.200 us` is a regression. The stretch-target gap is `179.200 - 70 = 109.200 us`.

## Per-step outcome

| Step | Candidate | Headline eager | Headline graph | Decision |
|---|---|---:|---:|---|
| 1 | Original `TileN=32, StageK=256` | 183.136 us | 183.200 us | baseline/profile |
| 2 | `TileN=64, StageK=256` | 179.200 us | 178.992 us | best headline, but not carried forward |
| 3 | `TileN=32, StageK=128` | 204.208 us | 203.712 us | rejected regression |

## Full-shape table (decode + decode_tp4)

Values are eager medians in microseconds from each step's `verify_fp8.json`.

| Shape group | M | K | N | baseline | Step 1 | Step 2 | Step 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 4096 | 4096 | 121.296 | 270.336 | 286.176 | 266.272 |
| decode | 1 | 4096 | 5376 | 140.016 | 331.712 | 333.872 | 320.000 |
| decode | 1 | 5376 | 4096 | 140.288 | 331.744 | 323.584 | 327.680 |
| decode | 4 | 4096 | 5376 | 140.224 | 337.344 | 331.792 | 317.376 |
| decode | 32 | 4096 | 5376 | 140.192 | 141.248 | 144.384 | 165.376 |
| decode | 32 | 5376 | 5376 | 181.664 | 183.136 | 179.200 | 204.208 |
| decode | 80 | 4096 | 5376 | 172.064 | 456.640 | 448.432 | 446.464 |
| decode_tp4 | 1 | 4096 | 1024 | 31.712 | 190.464 | 186.336 | 182.288 |
| decode_tp4 | 1 | 4096 | 1344 | 37.792 | 206.848 | 186.400 | 186.368 |
| decode_tp4 | 1 | 5376 | 1024 | 37.888 | 270.224 | 243.712 | 241.648 |
| decode_tp4 | 4 | 4096 | 1344 | 37.904 | 208.768 | 186.368 | 186.368 |
| decode_tp4 | 32 | 4096 | 1344 | 39.360 | 45.056 | 59.328 | 67.600 |
| decode_tp4 | 32 | 5376 | 1344 | 49.328 | 59.392 | 75.760 | 108.544 |
| decode_tp4 | 80 | 4096 | 1344 | 40.960 | 190.432 | 192.416 | 186.368 |

Prefill shapes are also present in the JSON artifacts; the inline PTX path is much slower than cuBLASLt there and is not a viable no-regression candidate.

## Bottleneck attribution

Step 1 ncu: memory throughput about 41-45%, compute throughput about 23-25%, grid 168 CTAs (`0.70` waves/SM), 95 registers/thread, achieved occupancy about 30-33%, and very low eligible warp issue rate. Step 2 halved CTA count to 84 and reduced redundant A staging, but ncu showed only `0.44` waves/SM and about 15% achieved occupancy, so the small event-timing gain was fragile. Step 3 reduced shared memory but added K-loop synchronization/staging overhead, dropping memory throughput to about 30% and regressing runtime.

Classification: latency/occupancy limited with memory-pipeline underutilization, not tensor-core compute-bound.

## If given a fourth step

I would not continue local tweaks to this hand-written cp/st.shared pipeline. The next meaningful CUDA C step would be a real TMA or warp-specialized pipeline that keeps the 32x32 tile count high while prefetching B and avoiding repeated A staging, but implementing that correctly is larger than the one-step budget. Track 2's CUTLASS/CuTe C++ path is a better place to test TMA + pipeline scheduling without rewriting this kernel from scratch.

## Carry-forward decision

Track 1 is closed. Restore and carry forward the original inline-PTX hot path (`TileM=32, TileN=32, StageK=256`) as the parity floor; Step 2 remains an archived measurement, not the promoted implementation, because it is slower than the original headline reference and fails the graph/full-shape no-regression gate.
