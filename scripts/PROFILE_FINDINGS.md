# v5 prefill kernel — Spark profiling findings

Source: `scripts/profile_v5_nsys.py` + ncu sections, run on Spark
(NVIDIA GB10, SM121, 48 SMs).  Captures live at `/tmp/v5_*.nsys-rep`
on the Spark container.

## Method

The profile script wraps a steady-state inner loop in
`cudaProfilerStart/Stop` with NVTX markers per iter.  10 warmup +
30 captured iters.  Run via the standard `command_on_spark.sh`
wrapper inside the persistent `b12x-w4a16-fkhoubsirat` container.

## nsys results (system-level)

### o_proj (K=4096, N=2688, M=2048) — v5/Marlin = 1.52×

| Metric | Value |
|---|---|
| Kernels captured | 1 monolithic v5 `@cute.kernel` |
| Median per-call | 881 µs |
| Min / Max | 877 / 1188 µs |
| StdDev | 66 µs (7%) |
| `cudaLaunchKernelExC` overhead | 4 µs avg (0.5% of total API time) |
| `cuda_gpu_mem_time_sum` | empty (no host↔device transfers in inner loop) |
| `nsys analyze` rules fired | none |
| GPU gaps > 500 ms | none |
| GPU utilization | > 50 % throughout |

### mamba_in_proj (K=2688, N=10304, M=2048) — v5/Marlin = 3.17×

Same picture: 1 monolithic kernel, 2589 µs median, max 3371 µs,
StdDev 232 µs (9%).  No system-level overhead.

**nsys verdict:** The kernel is the whole story.  No memcpy, no API
stalls, no GPU idle gaps.  FP4 staging is fused inside the kernel
(cooperative gmem→reg→smem decode interleaved with MMA in a single
`@cute.kernel` body) — nsys cannot break it down further.

## ncu results (kernel-internal)

### o_proj — SOL%

| Launch | Duration | Compute SM % | Memory % | L1/TEX % | L2 % |
|---|---|---|---|---|---|
| 0 | 1.12 ms | 38.20 | 43.60 | 42.76 | 21.43 |
| 1 | 1.12 ms | 38.40 | 43.84 | 42.09 | 21.52 |
| 2 | 1.11 ms | 38.61 | 44.10 | 42.23 | 21.64 |

**Classification:** Compute < 40%, Memory < 60% → **latency-bound**
per the skill's threshold table.  ncu's rule confirms: "Achieved
compute throughput and/or memory bandwidth below 60.0% of peak
typically indicate latency issues."

### o_proj — Memory Workload

| Metric | Value |
|---|---|
| Mem Busy | 43.87 % |
| Max Bandwidth | 24.20 % |
| L1/TEX Hit Rate | 60.68 % |
| L2 Hit Rate | **92.54 %** |
| L2 Persisting Size | 4.72 MB (= the W4A16 FP4 weight) |
| Mem Pipes Busy | 24.20 % |
| Local / Shared Memory Spilling | 0 (no spills) |

The FP4 weight tensor fits in L2 cache.  L1/TEX hit rate is moderate
(60 %).  Only 24 % of peak gmem bandwidth is used.  **Memory is not
the bottleneck.**

### o_proj — Occupancy & Scheduler

| Metric | Value |
|---|---|
| Block Size | 288 threads (= 9 warps: 8 MMA + 1 DMA) |
| Grid Size | 144 (persistent CTAs) |
| Registers Per Thread | **168** |
| Dynamic SMem Per Block | **66.56 KB** |
| **Block Limit Registers** | **1** ← register-bound |
| **Block Limit Shared Mem** | **1** ← smem-bound |
| Block Limit Warps | 5 (slack — irrelevant) |
| Theoretical Active Warps / SM | 9 |
| **Theoretical Occupancy** | **18.75 %** |
| Achieved Occupancy | 18.70 % (hits the ceiling) |
| Active Warps / Scheduler | 2.26 (vs hardware max of 12) |
| Eligible Warps / Scheduler | 0.26 |
| **No Eligible** | **78.94 %** of cycles |
| Warp Cycles / Issued Instruction | 10.72 |
| Issued Warp / Scheduler | 0.21 (= 1 instr every 4.7 cycles) |

**This is the root cause.**  168 registers × 288 threads = 48 K
registers per block.  66.5 KB smem per block.  Both pressures cap
the SM at 1 block.  With only 9 warps per SM (2.26 per scheduler),
when warps stall on the smem barriers between FP4 staging and MMA,
there are no other warps to switch to.  The scheduler is idle 78.9 %
of cycles.

### mamba_in_proj — comparison

Same shape autotune picks `tile_K=32` (vs o_proj's `tile_K=64`).
That cuts smem in half:

| Metric | o_proj (tK=64) | mamba_in (tK=32) |
|---|---|---|
| Block Limit Registers | 1 | **2** |
| Block Limit SMem | 1 | **2** |
| Theoretical Occupancy | 18.75 % | **37.50 %** |
| Achieved Occupancy | 18.70 % | **30.89 %** |
| Active Warps / Scheduler | 2.26 | **3.71** |
| No Eligible | 78.94 % | **61.18 %** |
| Memory Throughput | 43.60 % | 53.67 % |
| L2 Hit Rate | 92.54 % | 94.08 % |

**The autotune's tile_K=32 choice for K≤3712 is structurally validated**
by ncu: it doubles the per-SM block count and roughly halves the
scheduler-idle fraction.  Not just empirical — it's literally about
fitting 2 blocks per SM.

## Conclusion

### v5 is NOT bottlenecked by FP4 staging itself

- Memory bandwidth: only 24-37 % of peak used.
- L2 hit rate: 92-94 % (weights fit in cache).
- No memory spills, no system-level overhead, no GPU idle gaps.

### v5 IS bottlenecked by occupancy starvation

- 18.75 % theoretical occupancy at tile_K=64 (1 block/SM).
- 78.9 % of scheduler cycles have no eligible warp.
- Caused by **both** register pressure (168/thread) AND smem
  pressure (66.5 KB/block).
- The few warps per SM stall on smem barriers between FP4 staging
  and MMA, and there's no spare warp to hide that latency.

### Why the autotune helped — and what's left

The autotune's `tile_K=32` cuts smem by half, doubling occupancy to
~37 % for K ≤ 3712 shapes.  That's a structural fix, not a tuning
artifact.  But for K ≥ 4096 (o_proj, mamba_output_proj) the autotune
still picks `tile_K=64` — keeping the kernel stuck at 18.75 %
occupancy.

### Optimization directions

| Direction | Cost | Expected lift |
|---|---|---|
| Shrink tile (e.g., (64, 64, 64)) | 2× grid → CTA overhead | Should push occupancy to 2-3 blocks/SM |
| Drop a warp (4 MMA + 1 DMA) | Halve warp count → ~halve perf per CTA | Halves register footprint; double blocks/SM |
| Reduce accumulator footprint | Refactor MMA per-warp tiling | Cuts registers; complex |
| Use WGMMA (warpgroup MMA) | Architectural rewrite | Lower per-warp register pressure for accumulator; SM90+/SM100+ feature |
| Targeted fix for K ≥ 4096 only | Add tile_K=32 path for these shapes | Match mamba_in_proj's occupancy gain |

The "FP4 staging path optimization" direction the PR description
mentioned isn't where the perf is actually hiding.  The structural
finding is **occupancy**.  Marlin presumably runs at much higher
occupancy with a different per-CTA work split.

## Reports

Located in `/tmp/` on the Spark container `b12x-w4a16-fkhoubsirat`
(regenerable from `scripts/profile_v5_nsys.py`):

* `v5_oproj_M2048.nsys-rep`
* `v5_mamba_in_M2048.nsys-rep`
* ncu output captured to terminal (no .ncu-rep saved — re-run with
  `-o report_name` flag to persist).
