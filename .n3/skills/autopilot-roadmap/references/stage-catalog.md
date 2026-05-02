<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

# Stage Catalog

Detailed profiling actions, optimizations, and selection guidance for each
of the 9 optimization stages. Stages are ordered: fix the biggest bottleneck
category first, then refine.

## Stage 1: GPU Utilization

**Goal:** GPU utilization close to 100%. If the GPU is idle, the bottleneck
is host-side -- fix that before optimizing kernels.

**Profiling:**
- GPU utilization sampling via nvidia-smi/pynvml (quick check)
- CPU-GPU timeline analysis via nsys (identifies idle gaps, sync points)
- Host-side hotspot profiling via Python cProfile/py-spy (if CPU-bound)

**Optimizations (ordered by complexity):**
1. CPU-core binding -- numactl to reduce scheduling jitter and NUMA penalties
2. Data loader optimization -- prefetch, pin memory, increase num_workers
3. Synchronization removal -- eliminate unnecessary cuda.synchronize(), .item(),
   .cpu() calls
4. CUDA Graphs -- capture and replay static kernel sequences
5. Horizontal fusion -- reduce kernel count when CUDA Graphs are impractical

**Selection guidance:** Always include unless GPU utilization is already >95%.

## Stage 2: Redundancy Elimination

**Goal:** No wasted computation -- every operator necessary and executed once.

**Profiling:**
- Operator dependency analysis (trace dataflow graph)
- Compare operator list against model architecture

**Optimizations:**
1. Dead code elimination -- remove ops whose outputs are unused
2. Common subexpression elimination -- cache repeated computations
3. Constant folding -- pre-compute ops on static inputs

**Selection guidance:** Include when the operator trace shows suspiciously
high operator count or duplicated patterns.

## Stage 3: Hardware Feature Exploitation

**Goal:** Fully utilize hardware capabilities before writing custom kernels.

**Profiling:**
- Tensor Core utilization via ncu (sm__pipe_tensor_op_hmma metrics)
- Memory access pattern analysis (alignment, coalescing, L2 hit rate)
- Data layout inspection (NCHW vs NHWC)

**Optimizations (ordered by effort):**
1. Mixed precision -- FP16/BF16 via AMP; FP8/MXFP8 via Transformer Engine
   for Hopper+
2. Memory alignment -- pad tensor dimensions to multiples of 128 bytes
3. Channels-last layout -- NCHW to NHWC for convolution-heavy models
4. L2 cache fetch granularity -- set cudaLimitMaxL2FetchGranularity to
   128 bytes via cudaDeviceSetLimit

**Selection guidance:** Always include unless already running mixed precision
with good Tensor Core utilization.

## Stage 4: Heavy Operator Efficiency

**Goal:** GEMM, Conv, and MHA near peak SOL%.

SOL% = min(achieved_FLOPS / HW_peak_FLOPS, achieved_BW / HW_peak_BW) * 100

**Profiling:**
- Per-operator SOL% via ncu kernel metrics
- Identify operators below 70% SOL as candidates
- Roofline analysis to classify compute-bound vs memory-bound

**Optimizations:**
1. Library-level tuning -- cuBLAS heuristics, cuDNN algorithm selection,
   FlashAttention for MHA
2. Custom kernel generation -- send low-SOL operators to Triton/CuTe DSL;
   iterate with ncu until SOL% converges
3. Shape optimization -- pad matrix dimensions to tile-friendly sizes
   (multiples of 64/128)

**Selection guidance:** Always include -- heavy operators dominate runtime.

## Stage 5: Operator Fusion

**Goal:** Fuse remaining operators to reduce memory traffic and launch
overhead.

**Profiling:**
- Identify fusion candidates: chains of element-wise/reduction/normalization ops
- Measure per-operator bytes-per-FLOP ratios
- Detect horizontal fusion candidates (independent ops at same level)

**Optimizations:**
1. Vertical fusion -- fuse producer-consumer chains into single kernel
2. Horizontal fusion -- fuse independent ops of similar shape
3. Fusion engine dispatch -- send groups to Triton/CuTe DSL

**Selection guidance:** Include when there are many small operators between
heavy ops in the trace.

## Stage 6: Pipelining & Overlap

**Goal:** Overlap independent work across multiple CUDA streams to hide
latency and keep all hardware units busy.

**Profiling:**
- Operator dependency analysis (build DAG, find independent subgraphs)
- Per-operator GPU resource utilization
- nsys timeline analysis for sequential segments and idle gaps

**Optimizations (ordered by impact):**
1. Data prefetch overlap -- next batch on separate stream
2. Compute-communication overlap -- all-reduce concurrent with backward
3. Compute-compute overlap -- independent kernels on separate streams
4. Memory-compute overlap -- prefetch weights/activations during compute

**Selection guidance:** Include for distributed training or when nsys shows
sequential segments with idle gaps. Skip for single-GPU workloads unless
there are clear prefetch opportunities.

## Stage 7: Parallelism Strategy

**Goal:** Right-size the parallelism strategy (TP, PP, DP, FSDP) for the
hardware topology and model architecture.

**Profiling:**
- Communication-to-computation ratio per iteration
- Per-GPU memory utilization and headroom
- NVLink/IB bandwidth utilization vs peak
- All-reduce and all-gather collective sizes and durations

**Optimizations (ordered by impact):**
1. Tensor Parallelism (TP) sizing -- match TP degree to NVLink topology
   (e.g., TP=8 within a node, not across nodes)
2. Pipeline Parallelism (PP) partitioning -- balance stage compute times
   to minimize bubble overhead
3. Data Parallelism (DP) / FSDP tuning -- shard parameter/gradient/optimizer
   states; tune bucket sizes for communication efficiency
4. Sequence Parallelism (SP) -- distribute activation memory for long sequences
5. Expert Parallelism (EP) -- for MoE models, balance expert placement

**Selection guidance:** Include for multi-GPU workloads. Skip for single-GPU.
Ensure Stage 6 (overlap) is addressed first -- parallelism strategy changes
affect communication patterns that overlap optimizations depend on.

## Stage 8: Resource Trade-offs

**Goal:** Rebalance compute/memory/communication for maximum utilization.

**Profiling:**
- Resource utilization breakdown (compute, memory capacity/BW, communication)
- Identify imbalanced resources

**Optimizations:**
1. Recomputation vs memory -- activation checkpointing + larger batch size
2. Precision vs accuracy -- lower precision for more throughput
3. Communication vs computation -- gradient compression, local accumulation

**Selection guidance:** Include when profiling shows resource imbalance
(e.g., GPU memory underutilized, or communication-dominated).

## Stage 9: Opportunistic

**Goal:** Workload-specific optimizations not covered by systematic stages.

**Approach:**
- Review all profiling data for anomalies or unexploited patterns
- Consult knowledge bases for domain best practices
- Propose and validate optimizations following backup-modify-validate-revert

No fixed optimization menu. The LLM has full context from prior stages.

**Selection guidance:** Always include as final stage -- there may be
domain-specific opportunities.
