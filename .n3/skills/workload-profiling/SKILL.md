---
name: workload-profiling
description: >
  Code instrumentation for timing workloads. Two scenarios:
  (1) Training loop — inject PerfGauge or manual timing to report
  per-iteration latency, throughput (samples/sec), data load time, and GPU
  hardware metrics. (2) Standalone kernel/op — write CUDA event timing code
  with warmup, per-iteration statistics, and anti-pattern avoidance. Also
  covers NVTX annotation for labeling profiler timelines.
  NOT for: running or analyzing profiler tools (nsys, ncu, Nsight Systems,
  Nsight Compute), writing kernels (Triton, CuTe, CUDA), applying
  optimizations (CUDA Graphs, gradient checkpointing, fusion), or
  interpreting roofline/SOL% metrics.
  Triggers: "measure throughput", "benchmark this function", "time my
  training loop", "samples per second", "NVTX annotate", "instrument my
  dataloader", "data load time", "kernel timing", "how do I time".
user-invocable: false
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
  documentation: https://gitlab-master.nvidia.com/wkong/perf-bot
---

# Workload Profiling

## Quick Reference

**First**, check if dltools is available: `python -c "from dltools.gauge.pth import PerfGauge; print('ok')"`. If it fails, skip the entire "Loop Workloads — PerfGauge" section and use manual `torch.cuda.synchronize()` + `time.perf_counter()` timing with warmup for training loops instead.

Pick ONE path based on the workload type:

| Workload | Approach | Section |
|----------|----------|---------|
| Training loop with PyTorch DataLoader | Wrap with `PerfGauge(dataloader, global_batch_size=N)` (requires dltools) | Loop Workloads — PerfGauge |
| Training loop with custom iterator | Use `PerfGaugeBase` lifecycle hooks (requires dltools) | Loop Workloads — PerfGauge (custom iterator template) |
| Training loop (no dltools) | Manual `torch.cuda.synchronize()` + `time.perf_counter()` with warmup | Non-Loop Workloads — CUDA Event Benchmarking (adapt pattern) |
| Single kernel or op | Write CUDA event benchmark (pre-allocate, warmup, event pairs) | Non-Loop Workloads — CUDA Event Benchmarking |
| Add timeline labels for nsys | Use `@nvtx.annotate` decorator or context manager | NVTX Reference |

## Principles

- **Measure, don't guess.** Every performance claim must trace back to profiler output or structured measurement data. Never invent metrics.
- **Isolate steady-state.** Warmup costs (CUDA context init, cuDNN autotuning, JIT compilation) distort measurements. Always exclude warmup iterations before collecting data.
- **Use hardware timing.** CUDA events measure GPU time precisely. CPU timers (`time.perf_counter()`) include host overhead and miss asynchronous execution.
- **No sync inside measurement loops.** Each `torch.cuda.synchronize()` adds 10-50us overhead. Record CUDA events asynchronously, sync once at the end.
- **Pre-allocate everything.** Tensors, events, compiled kernels — all before the timing loop. For CuTe DSL kernels, pre-compile with `cute.compile()`.
- **Minimize profiler interference.** Start with lightweight measurement (PerfGauge latency/throughput) and escalate to heavier tools (Kineto, nsys, ncu) only when lighter tools cannot answer the question.

## Loop Workloads — PerfGauge

For training loops and iterative workloads, use dltools PerfGauge to measure per-iteration latency, throughput, data load time, and GPU metrics.

### How It Works

PerfGauge wraps a dataloader iterator. It handles:
- Warmup (metrics reset after warmup iterations)
- Per-iteration timing (batch time, data load time)
- Throughput calculation (samples/sec from global batch size)
- GPU monitoring via DeviceMonitor (power, temperature, clocks, utilization)
- Profiler lifecycle management (Kineto, memory profiler, cProfile)
- Automatic `cudaProfilerStart`/`cudaProfilerStop` at profiling boundaries

Because PerfGauge calls `cudaProfilerStart`/`cudaProfilerStop` internally, a PerfGauge-instrumented script works directly with `nsys profile --capture-range=cudaProfilerApi` — no separate instrumentation needed.

### Injection Template

Read the user's training script, understand the dataloader and loop structure, then inject PerfGauge wrapping. Configuration is via the `DLTOOLS` environment variable, not Python kwargs.

```python
import os
os.environ["DLTOOLS"] = "5,30;--warmup=5;--dmon;--no_exit_on_stop;--gbs=BATCH_SIZE"

from dltools.gauge.pth import PerfGauge

gauge = PerfGauge(dataloader, global_batch_size=BATCH_SIZE)
for batch in gauge:
    # ... existing training loop body ...
```

**Important:** Set `DLTOOLS` env var BEFORE importing PerfGauge. The `PerfGauge` constructor only accepts `(iterable, global_batch_size)` — all other config goes through `DLTOOLS`.

For custom iterators (not PyTorch DataLoader), use `PerfGaugeBase` with explicit `next()` so `on_dataloader_begin/end` brackets the actual data fetch:

```python
import os
os.environ["DLTOOLS"] = "5,30;--warmup=5;--no_exit_on_stop;--gbs=BATCH_SIZE"

from dltools.gauge.base import PerfGaugeBase

gauge = PerfGaugeBase(global_batch_size=BATCH_SIZE)
gauge.on_app_begin()
for epoch in range(num_epochs):
    gauge.on_epoch_begin()
    it = iter(custom_iterator)
    while True:
        gauge.on_iteration_begin()
        gauge.on_dataloader_begin()
        try:
            batch = next(it)          # data fetch measured here
        except StopIteration:
            break
        gauge.on_dataloader_end()     # ← .to(device) goes AFTER this line
        batch = batch.to(device)      # H2D transfer counted in compute, not data
        train_step(batch)
        gauge.on_iteration_end()
    gauge.on_epoch_end()
gauge.on_app_end()
```

**Why explicit `next()`?** With `for batch in iterator:`, Python calls `__next__` *before* the loop body runs — so `on_dataloader_begin/end` in the body would miss the actual data fetch. Using `next(it)` inside the hooks ensures data loading time is measured correctly.

### DLTOOLS Config Reference

Configuration is passed via `DLTOOLS` environment variable. Format: `start,stop;--flag;--key=value` separated by semicolons.

**Example:** `DLTOOLS="5,30;--warmup=5;--dmon;--no_exit_on_stop;--gbs=128"`

| Flag | Default | Effect |
|------|---------|--------|
| `start,stop` | 5,30 | Profiling iteration range (positional, e.g. `5,30`) |
| `--warmup N` | 5 | Warmup iterations (metrics reset after warmup) |
| `--gbs N` | -1 | Global batch size (overrides constructor arg) |
| `--dmon` | off | DeviceMonitor: power, temp, SM/memory clocks, utilization |
| `--kineto` | off | PyTorch Kineto profiler for operator-level breakdown |
| `--memory_profiler` | off | CUDA memory allocation tracking |
| `--cprofiler` | off | Python cProfile for CPU-side bottleneck identification |
| `--no_exit_on_stop` | exit | Don't exit process after profiling range completes |
| `--log_interval N` | 1 | Print metrics every N iterations |

### Output Format

PerfGauge prints per-iteration metrics to stdout as log lines:

```
0:[PerfGauge][INFO]: [0005]: iter 125.34 ms, data 12.45 ms, fps 256.32
0:[PerfGauge][INFO]: [0006]: iter 123.89 ms, data 11.98 ms, fps 259.41
0:[PerfGauge][INFO]: Average: iter 124.62 ms, data 12.22 ms, fps 257.86
```

Fields: `iter` = total iteration time (ms), `data` = dataloader time (ms), `fps` = throughput (samples/sec). The `Average` line is the summary over the profiling range.

With `--dmon`, DeviceMonitor metrics are logged separately showing power (W), clocks (MHz), utilization (%), and temperature (C).

Parse these log lines to extract metrics. Do not invent numbers.

### Interpreting Results

- **iter (ms)**: Total wall-clock time per iteration (includes compute + data loading + communication)
- **data (ms)**: Time spent in dataloader. If `data / iter > 0.2`, data loading is a bottleneck.
- **fps**: Global throughput in samples/second. Use with known FLOPs-per-sample to compute MFU.
- **DeviceMonitor metrics**: If GPU utilization is low (<80%) but iter time is high, suspect CPU bottleneck or data loading issues. If power is well below TDP, GPU may be under-utilized.

### Limitations

PerfGauge reports **aggregate** timing (iter, data) — not per-sub-phase breakdown (forward, backward, optimizer). When the user asks **where time is spent within compute**:

1. Add `torch.cuda.synchronize()` + `time.perf_counter()` around each sub-phase for a one-off diagnosis, OR
2. Add NVTX annotations and run with `nsys profile` for timeline visualization.

**Do NOT derive compute time as `iter - data`.** GPU operations run asynchronously — `iter - data` shows CPU launch overhead, not actual GPU execution time. This is especially misleading when data loading dominates (e.g., data=320ms, iter=324ms → `iter - data = 4ms` but actual GPU compute may be different). Report PerfGauge's `iter` and `data` fields directly; do not compute derived metrics from them.

## Non-Loop Workloads — CUDA Event Benchmarking

For single kernels, one-shot inference, or standalone operations, write CUDA event benchmarking code directly.

### PyTorch: Simple (Mean Only)

```python
import torch

def benchmark(fn, warmup=50, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms per iteration
```

### PyTorch: Detailed (Per-Iteration Stats)

```python
import torch
import statistics

def benchmark_detailed(fn, warmup=50, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }
```

### Anti-Patterns

| Anti-Pattern | Problem |
|--------------|---------|
| `torch.cuda.synchronize()` before AND after each iteration | Adds ~10-50us overhead per iteration |
| `time.perf_counter()` for GPU timing | Measures CPU time, misses async GPU execution |
| Missing warmup | First iterations include JIT, clock ramp-up, context init |
| Allocating tensors inside measurement loop | Allocation overhead pollutes timing |
| Reporting only mean | Hides variance, outliers, bimodal distributions |

For additional benchmarking templates (CUDA Graph, CuTe DSL, Triton, Raw CUDA), see [references/benchmarking-patterns.md](references/benchmarking-patterns.md).

## NVTX Reference

NVTX (NVIDIA Tools Extension) adds named annotations to profiler timelines. Use NVTX to label phases (forward, backward, optimizer) for readability in nsys — not for measurement.

```python
import nvtx

# Decorator — annotates every call
@nvtx.annotate("training_step", color="blue")
def training_step():
    ...

# Context manager — annotates a code block
with nvtx.annotate("data_loading", color="green"):
    batch = next(dataloader)
```

- **Do** annotate training phases (forward, backward, optimizer, data loading) for nsys timeline clarity.
- **Do not** annotate for measurement — use PerfGauge or CUDA events instead.
- **Do not** over-annotate — too many fine-grained ranges add visual clutter and minor overhead.

For NVTX domains, categories, payloads, and legacy API details, see [references/nvtx-api.md](references/nvtx-api.md).

## References

- [references/benchmarking-patterns.md](references/benchmarking-patterns.md) — CUDA Graph, CuTe DSL, Triton, Raw CUDA templates; warmup guidance; GPU hardware properties; reporting format
- [references/nvtx-api.md](references/nvtx-api.md) — Domains, categories, payloads, legacy push/pop API
- [references/pytorch-profiler-api.md](references/pytorch-profiler-api.md) — PyTorch 2.0+ profiler API changes (`device_time` vs deprecated `cuda_time`)
