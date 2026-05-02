# Track 1 Step 1 result

Measured the unmodified CUDA inline-PTX kernel on the fixed headline shape after adding `(32, 5376, 5376)` to the verifier shape set.

- Full verifier headline: baseline eager `181.664 us`, baseline graph `181.392 us`; inline PTX eager `183.136 us`, graph `183.200 us`; correctness passed (`cos=0.99999857`, `max_abs=0.01623 <= 0.05216`).
- nsys headline profile: inline kernel median `179.200 us` over 12 instances; cuBLASLt nvjet median `177.616 us` over 12 instances. The L2-flush event dominates total profile time, so kernel medians are the relevant comparison.
- ncu: `Duration 159.58-174.94 us`, memory throughput `40.85-44.77%`, compute throughput `22.82-25.01%`, grid `168` CTAs, `0.70` waves/SM, `95` registers/thread, theoretical occupancy `41.67%`, achieved occupancy about `30-33%`.

Decision: proceed to Step 2. The dominant actionable issue is under-filled grid/low eligible warp issue rate plus redundant A-panel staging across 168 N-tiles. The first nominal tuning attempt will widen the headline N tile from 32 to 64 so each CTA computes two adjacent N tiles with all four warps, reducing CTA count and redundant A loads while preserving total compute-warps.
