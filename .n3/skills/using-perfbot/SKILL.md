---
name: using-perfbot
description: >
  Use when the user's task involves GPU performance, profiling, kernel
  development, or deep learning optimization. Routes to the right
  PerfBot skill or specialist agent based on user intent.
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
  documentation: https://gitlab-master.nvidia.com/wkong/perf-bot
---

You have **PerfBot** -- an AI toolkit for deep learning performance analysis and optimization on NVIDIA GPUs.

## When to Use PerfBot

Use PerfBot skills and agents when the task involves:
- GPU profiling, performance analysis, or bottleneck diagnosis
- Writing or optimizing GPU kernels (Triton, CuDeepy/CuTe, TileIR)
- CUDA Graph compatibility analysis or migration
- Deep learning training/inference performance optimization
- Questions about GPU performance concepts, CUDA, Triton, or profiling tools

## Operating Modes

Choose the mode based on user intent:

**Query**: Answer questions about performance concepts.
- Search knowledge bases with `search_docs` and synthesize answers.

**Analysis**: Understand performance status and bottlenecks.
- Use the `performance-analysis` skill for workflow guidance.
- Delegate profiling to `profiling-specialist` for actual measurements.
- Key metrics: throughput, latency, DRAM bandwidth, MFU, SOL%, GPU utilization.

**Optimization**: Apply specific performance improvements.
- Use the `performance-optimization` skill for routing guidance.
- Delegate to the appropriate specialist for implementation and validation.

**Autopilot**: Achieve a performance goal through iterative optimization.
- Loop: Profile (via `profiling-specialist`) -> Optimize (via specialist) -> Validate until goal met.
- Track progress with a checklist across iterations.

## Specialist Subagents

The following specialist subagents are available via the Task tool. Delegate specialist work to them — do not perform it yourself:

| Subagent | Expertise |
|---|---|
| `profiling-specialist` | GPU profiling (nsys, ncu, torch.profiler), metric collection, trace analysis |
| `torch-cuda-graph-specialist` | CUDA Graph API selection, compatibility analysis, capture workflows, and sync elimination |
| `triton-specialist` | Triton kernel development, operator routing, autotune configuration |
| `tileir-specialist` | TileIR optimization for existing Triton kernels on Blackwell GPUs (sm_100+) |
| `cute-dsl-specialist` | CuTe DSL kernel writing (NOT Triton) — GEMM, attention, element-wise, reduction. Writes kernels from CUTLASS examples or falls back to CuDeePy CLI |
| `ctm-specialist` | CTM (Close-to-Metal) bare-metal kernel development — `import ctm` with direct PTX/NVVM access via `ctm.nvvm.*` (TMA, mbarrier, tcgen05, wgmma). NOT CuTe DSL tile-level abstractions |
| `cuda-kernel-specialist` | Raw CUDA C/C++ kernel development (.cu files) with pybind11 bindings and torch.utils.cpp_extension |
| `gpu-memory-specialist` | Memory optimization, fragmentation reduction, gradient checkpointing |
| `mixed-precision-specialist` | Mixed precision training — fp16/bf16 autocast, FP8 (TransformerEngine), manual precision control |
| `distributed-specialist` | Multi-GPU communication patterns, NCCL optimization |

**CTM vs CuTe DSL:** Both share `@cute.kernel`/`@cute.jit` decorators but operate at different abstraction levels. CTM (`import ctm`, `ctm.nvvm.*`) gives direct hardware control; CuTe DSL (`cutlass.cute`) provides tile-level abstractions. If unclear which the user needs, ask.

The operating mode skills (`performance-analysis`, `performance-optimization`, `autopilot-roadmap`) contain detailed routing logic for which subagent to delegate to based on the task.

## Delegation Principles

- **Route, don't execute**: Coordinate and route requests; don't perform analysis or profiling directly.
- **Profile before optimizing**: Always delegate profiling to `profiling-specialist` before applying optimizations.
- **Don't read files for analysis**: Delegate file reading and workload understanding to specialists.
- **Let specialists own output**: Never specify output paths; specialists manage their own artifact directories.
- **Sequential when dependent**: Each step in a workflow depends on the previous step's output. Never run dependent steps in parallel.

## Remote Cluster Execution

When a task requires running on a remote SLURM cluster (user mentions a cluster
name, remote GPUs, or the workload isn't available locally):

1. **Read config**: Look for `slurm-cluster-*.md` in the project root. If none
   exists, invoke the `remote-slurm` skill to create one.
2. **Set up allocation**: Invoke the `remote-slurm` skill to establish SSH and
   create a persistent allocation (Recipe 6, Steps 1-3).
3. **Delegate with context**: Include a **Remote Execution Context** block in
   every specialist delegation prompt:

   ```
   ## Remote Execution Context
   Cluster: <cluster> | Host: <ssh_host>
   Jobid: <jobid> | Container: perfbot-<jobid>
   Remote CWD: <remote_cwd>
   Mounts: <mounts>

   Run commands:  ssh <host> "srun --jobid=<jobid> --container-name=perfbot-<jobid> --container-mounts=<mounts> <command>"
   Per-node ops:  add --ntasks-per-node=1
   Write files:   ssh <host> "cat > <remote_cwd>/file.py" << 'PYEOF' ... PYEOF
   Read results:  ssh <host> "tail -n 100 <remote_cwd>/file" (NEVER full cat)
   Binary files:  scp <host>:<remote_cwd>/file.nsys-rep .
   ```

4. **Reuse across specialists**: Pass the same context block to every specialist
   in the workflow. The allocation persists across delegations.
5. **Clean up**: After all specialist work is done, release the allocation
   (Recipe 6, Step 5).

## Key Principles

1. **Measure first** -- profile before optimizing; never guess at bottlenecks.
2. **Data over assumptions** -- all metrics must come from tool/script output, not LLM generation.
3. **Safe modifications** -- back up code before modifying, validate after, revert on failure.
4. **Use `search_docs`** -- the MCP tool searches PerfBot knowledge bases for documentation lookups.
