# Track 1 Step 2 result

Tried `TileM=32, TileN=64, StageK=256` on the hot path while keeping the existing inline PTX unchanged.

- Full verifier headline: baseline eager `180.336 us`, baseline graph `178.000 us`; candidate eager `179.200 us`, graph `178.992 us`; correctness passed (`cos=0.99999857`, max abs within limit). This is a small event-timing improvement versus Step 1 (`183.136 us -> 179.200 us` eager) but still does not beat the simultaneous graph baseline and is far above the 70-80 us target.
- nsys headline profile: candidate median `178.304 us`; nvjet median `178.912 us` in the same run. This is approximately parity, not a decisive win.
- ncu regressed occupancy: grid `84` CTAs (`0.44` waves/SM), dynamic shared memory `24.58 KiB`, `96` registers/thread, achieved occupancy about `14-15%`, memory throughput about `34-36%`, compute throughput about `19-20%`, duration commonly `184-193 us`.

Decision: do not build Step 3 on top of `TileN=64`. It reduces redundant A loads but underfills Spark too much and cuts achieved occupancy in half. Step 3 will revert the hot path to `TileN=32` and instead try a targeted occupancy/register-pressure reduction by lowering `StageK` to 128, which increases CTA waves and dynamic-smem residency while preserving the proven 32x32 tile shape.
