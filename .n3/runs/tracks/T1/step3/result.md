# Track 1 Step 3 result

Tried TileM=32, TileN=32, StageK=128 after reverting the Step 2 TileN=64 experiment.

- Full verifier headline: baseline eager 181.344 us, baseline graph 181.232 us; candidate eager 204.208 us, graph 203.712 us; correctness passed (cos=0.99999857, max abs within limit), but repeat spread was 2.72%, above the 2% stability gate.
- nsys headline profile: candidate median 200.608 us; nvjet median 182.464 us in the same run. This is a clear regression.
- ncu: dynamic shared memory fell to 8.19 KiB, but runtime worsened (~242-250 us in profiled passes), memory throughput fell to about 30%, compute throughput to about 18%, and achieved occupancy remained about 28-30%. More K-loop synchronization/staging overhead dominated any residency benefit.

Decision: reject Step 3. Track 1 best measured candidate remains Step 2 by headline median (179.200 us eager), but it does not beat torch._scaled_mm across all modes/shapes and has worse ncu occupancy. For code carry-forward, restore the original hot path (TileN=32, StageK=256) as the parity floor before starting Track 2.
