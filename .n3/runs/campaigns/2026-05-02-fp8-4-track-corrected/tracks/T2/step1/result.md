# T2 Step 1 result

Candidate: separate CUTLASS/CuTe SM120 atom smoke extension using `cute::SM120_16x8x32_TN<float_e4m3_t,float_e4m3_t,float>` inside the existing shared-memory/register staging pattern.

Correctness: passed for all 22 verifier shapes. The verifier JSON stores this candidate in the `b12x_inline_ptx_*` fields and marks `candidate=sm120_cute_atom`.

Headline `(M=32,K=5376,N=5376)`:

| path | eager us | graph us |
| --- | ---: | ---: |
| torch._scaled_mm | 186.304 | 186.304 |
| CUTLASS atom smoke | 362.496 | 360.592 |

Nsight Systems headline median kernel time was about 280 us for the profiled kernel. Nsight Compute reported 430.40 us duration for the first profiled replay, 41.62% memory throughput, 12.04% compute throughput, 0.50 waves/SM, 29.17% theoretical occupancy, and 11.37% achieved occupancy. Main reported issues were uncoalesced global/shared accesses, shared bank conflicts, and L1TEX scoreboard stalls.

Decision: correct but not performant. Continue only with small topology probes; do not carry this as a production candidate.
