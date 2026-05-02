---
name: autopilot-roadmap
description: >
  Two-phase autopilot planning for iterative performance goal optimization.
  Phase 1 collects a WorkloadAssessment (hardware profile, workload
  characteristics, user constraints, per-operator roofline SOL%). Phase 2
  produces an OptimizationRoadmap with 9 ordered stages prioritized by
  bottleneck severity. Only for autopilot mode where a measurable
  performance target is given. Triggers: "make this 2x faster",
  "optimize to 80% SOL", "achieve N% GPU utilization",
  performance goal loop with iterative analysis-optimization cycles.
  NOT for one-shot optimization or analysis requests.
user-invocable: false
license: LicenseRef-NvidiaProprietary
tags:
  - autopilot
  - planning
  - optimization
  - assessment
metadata:
  author: NVIDIA Corporation
  documentation: https://gitlab-master.nvidia.com/wkong/perf-bot
---

# Autopilot Optimization Roadmap

## Principles

1. **Assess before planning.** Never generate a roadmap without measured data.
   Phase 1 (WorkloadAssessment) must complete before Phase 2 (roadmap).
2. **Fix the biggest category first.** The 9 stages are ordered so that
   high-level bottlenecks (idle GPU, wasted ops) are resolved before
   micro-optimizations (fusion, parallelism). Skipping ahead wastes effort.
3. **Ground every decision in SOL%.** SOL% (Speed-of-Light percentage) is the
   ratio of theoretical minimum time to actual time. It quantifies how much
   headroom remains for each operator.
4. **Respect user constraints.** Constraints (batch size, precision, memory)
   are hard limits that override optimization suggestions.
5. **Iterate, measure, advance.** Each autopilot loop iteration runs one
   action, measures impact, then decides whether to stay in the current
   stage or advance to the next.

## Optimization Principles

These principles guide stage selection and optimization decisions:
- **Pipeline**: Overlap compute, memory, and communication.
- **Parallelism**: Scale across GPUs with the right strategy (TP, PP, DP, FSDP).
- **Locality**: Minimize data movement.
- **Vectorization**: Maximize parallel utilization (SIMD, tensor cores).
- **Fusion**: Combine operations to reduce kernel launch overhead.
- **Precision**: Use lower precision (FP16, BF16, FP8) where safe.
- **Batching**: Amortize fixed costs with larger work units.
- **Async**: Eliminate synchronization points to keep all units busy.

## Phase 1: Workload Assessment

Collect a `WorkloadAssessment` in five steps, in order.

### Step 1: Extract User Constraints

Parse the user's goal for hard limits:
- Batch size ("keep BS=32"), precision ("stay in FP32")
- Memory ("must fit in 40GB"), correctness ("accuracy within 1%")

Output as a list of constraint strings.

### Step 2: Query Hardware

Use `nvidia-smi --query-gpu` or `torch.cuda.get_device_properties()`:
- GPU model, compute capability, GPU count
- Peak FP16/FP32 TFLOPS, peak memory bandwidth (HBM), memory capacity
- NVLink/InfiniBand bandwidth (if multi-GPU)
- CPU core count, NUMA topology

### Step 3: Identify Workload Characteristics

Inspect the training script and/or run a short profiling pass:
- Model type (transformer, CNN, RNN), framework, parameter count
- Batch size, sequence length, primary dtype (FP32/FP16/BF16/FP8)
- Distributed setup and parallelism strategy (DP, TP, PP, FSDP)

### Step 4: Profile One Iteration for Per-Operator SOL

Run one training iteration with profiler enabled (PyTorch profiler, nsys, or
ncu) to capture operator-level data.

**Operator grouping:** Work at the **natural operator level** -- a PyTorch
operation (e.g., `Linear`, `LayerNorm`, `SDPA`) that may launch multiple
CUDA kernels. Group kernels by parent operator, separate forward vs backward.

For each operator:
1. Identify problem size from arguments (M/N/K for Linear, shape for norms)
2. Group CUDA kernels by forward/backward pass
3. Compute per-pass SOL time: `sol_time = max(FLOPs / peak_FLOPS, bytes / peak_BW)`
4. Derive SOL%: `sol_pct = sol_time / actual_time * 100`
5. Classify: compute-bound if `FLOPs/peak_FLOPS > bytes/peak_BW`, else memory-bound
6. Combine passes: `total_sol_time = fwd.sol_time + bwd.sol_time`
7. Count occurrences per iteration (e.g., 24x for 24-layer transformer)

**Roll up to iteration level:**

```
iteration_SOL% = sum(op.total_sol_time_us * op.count) /
                 sum(op.total_duration_us * op.count) * 100
```

Example: operators with actual 20ms + 30ms, SOL 10ms + 25ms ->
iteration SOL% = 35/50 * 100 = 70%.

Rank top bottlenecks by time contribution (duration * count) and SOL gap.

### Step 5: Output WorkloadAssessment

Output as JSON. See [Output Formats](#workloadassessment) below.

## Phase 2: Roadmap Planning

Use the WorkloadAssessment to produce an `OptimizationRoadmap`.

1. **Review assessment** -- hardware, workload, constraints, SOL gaps.
2. **Select stages** -- include only stages relevant to this workload's
   bottlenecks. Skip inapplicable stages (e.g., Pipeline Parallelism for
   single-GPU). See the stage overview table and `references/stage-catalog.md`
   for full details.
3. **Tailor plans** -- customize profiling and optimization plans per stage.
   Write rationale grounded in measured SOL gaps, not generic guidance.
4. **Output roadmap** as JSON. See [Output Formats](#optimizationroadmap) below.

### Stage Overview

Stages are ordered: **fix the biggest bottleneck category first, then refine.**

| # | Stage | Goal | Always Include? |
|---|-------|------|-----------------|
| 1 | GPU Utilization | GPU active ~100% of the time | Unless GPU util >95% |
| 2 | Redundancy Elimination | No wasted or duplicate ops | When op count seems high |
| 3 | Hardware Feature Exploitation | Use Tensor Cores, AMP, layout | Unless already mixed precision |
| 4 | Heavy Operator Efficiency | GEMM/Conv/MHA near peak SOL% | Always |
| 5 | Operator Fusion | Reduce memory traffic between ops | When many small ops in trace |
| 6 | Pipelining & Overlap | Overlap compute/comm/prefetch across streams | Distributed or prefetch opportunities |
| 7 | Parallelism Strategy | Right-size TP/PP/DP/FSDP for hardware topology | Multi-GPU workloads |
| 8 | Resource Trade-offs | Rebalance compute/memory/comm | When resource imbalance detected |
| 9 | Opportunistic | Domain-specific opportunities | Always (final sweep) |

Each stage has profiling actions, ordered optimizations, and selection guidance.
Full details: `references/stage-catalog.md`.

## Iteration Protocol

During the autopilot loop, repeat each iteration:

1. **Check current stage** in the roadmap.
2. **Execute next action** -- run the next profiling or optimization step.
3. **Record findings** -- what was discovered or applied.
4. **Advance or stay** -- if the stage goal is met or no candidates remain,
   advance to the next pending stage.
5. **Report metric** -- always report the current target metric value.

## Output Formats

### WorkloadAssessment

```json
{
  "hardware": {
    "gpu_name": "NVIDIA H100 SXM",
    "compute_capability": "9.0",
    "gpu_count": 1,
    "peak_fp16_tflops": 989.0,
    "peak_fp32_tflops": 67.0,
    "peak_memory_bw_gbps": 3350.0,
    "memory_capacity_gb": 80.0,
    "nvlink_bw_gbps": null,
    "ib_bw_gbps": null,
    "cpu_cores": 64,
    "numa_nodes": 2
  },
  "workload": {
    "model_type": "transformer",
    "framework": "pytorch",
    "batch_size": 32,
    "sequence_length": 2048,
    "parameter_count": 125000000,
    "dtype": "fp32",
    "distributed": false,
    "parallelism_strategy": null
  },
  "user_constraints": ["keep batch_size=32"],
  "operator_sol": [
    {
      "name": "Linear",
      "category": "gemm",
      "problem_size": {"M": 4096, "N": 4096, "K": 1024},
      "forward": {
        "kernels": ["ampere_fp16_s1688gemm"],
        "duration_us": 100.0,
        "sol_time_us": 45.0,
        "achieved_flops": 50e12,
        "achieved_bw_gbps": 1200.0,
        "sol_pct": 45.0,
        "bottleneck": "memory"
      },
      "backward": {
        "kernels": ["gemm_bwd_kernel"],
        "duration_us": 120.0,
        "sol_time_us": 50.4,
        "achieved_flops": 45e12,
        "achieved_bw_gbps": 1100.0,
        "sol_pct": 42.0,
        "bottleneck": "memory"
      },
      "total_duration_us": 220.0,
      "total_sol_time_us": 95.4,
      "sol_pct": 43.4,
      "bottleneck": "memory",
      "count": 24
    }
  ],
  "iteration_sol_pct": 38.5,
  "top_bottlenecks": [
    "Linear (fwd): 45% SOL, memory-bound (24x per iter, 62% of total time)"
  ],
  "assessment_summary": "Running FP32 on H100, Tensor Cores completely unused..."
}
```

### OptimizationRoadmap

```json
{
  "workload_summary": "GPT-2 125M, FP32, BS=32, single H100",
  "initial_assessment": "38.5% iteration SOL, GEMM ops memory-bound at 43%",
  "target_metric": "sol_pct",
  "target_value": 80.0,
  "assessment_summary": "Iteration SOL at 38.5%, GEMM ops memory-bound at 45%",
  "iteration_sol_pct": 38.5,
  "stages": [
    {
      "id": 3,
      "name": "Hardware Feature Exploitation",
      "goal": "Enable Tensor Cores via mixed precision",
      "profiling_plan": ["ncu Tensor Core utilization check"],
      "optimization_plan": ["Enable AMP (BF16)", "Verify loss scaling"],
      "rationale": "FP32 on H100 leaves Tensor Cores entirely idle; AMP alone should double throughput"
    },
    {
      "id": 4,
      "name": "Heavy Operator Efficiency",
      "goal": "Linear ops above 70% SOL",
      "profiling_plan": ["Per-op SOL% with ncu after AMP"],
      "optimization_plan": ["FlashAttention for MHA", "Pad M/K to multiples of 128"],
      "rationale": "Linear at 43% SOL with 62% time share -- largest headroom"
    }
  ]
}
```

## Error Handling and Edge Cases

**When to skip stages:**
- Skip Stage 6 (Pipelining & Overlap) for single-GPU workloads with no
  prefetch opportunity.
- Skip Stage 7 (Parallelism Strategy) for single-GPU workloads.
- Skip Stage 2 (Redundancy Elimination) if operator count matches the
  model architecture with no duplicates.
- Skip Stage 8 (Resource Trade-offs) if all resources are balanced and
  no user constraints block rebalancing.

**Iteration stalls:**
- If two consecutive iterations show <1% improvement on the target metric
  within the same stage, advance to the next stage.
- If the target metric regresses after an optimization, revert the change
  and try the next optimization in the stage's plan.

**When to abort:**
- If all stages are exhausted and the target is not met, report the best
  achieved metric and explain which bottlenecks remain.
- If user constraints make the target mathematically unreachable (e.g.,
  "stay FP32" on a memory-bound workload needing 2x throughput), report
  this during Phase 2 roadmap planning with an explanation.

**Incomplete profiling data:**
- If ncu is unavailable, estimate SOL% from nsys kernel durations and
  known hardware peak specs. Flag estimates as approximate.
- If GPU hardware specs are not in the database, use `nvidia-smi` reported
  clocks and compute capability to derive peak FLOPS.
