---
name: kernel-validation
description: >
  Verify GPU kernel numerical correctness against a reference implementation.
  Runs verify_kernel.py to compare outputs with configurable tolerance for
  mixed-precision dtypes (fp16/bf16/fp8/int8). Auto-discovers kernel callable
  (run_kernel, forward, fused_*), generates test harness, reports max absolute
  and relative differences. Supports TileIR backend detection via ENABLE_TILE.
  Triggers: verify kernel, check correctness, numerical accuracy, tolerance,
  allclose, max diff. Use after any kernel write, fuse, or optimization --
  always verify correctness BEFORE benchmarking (see workload-profiling skill).
user-invocable: false
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
  documentation: https://gitlab-master.nvidia.com/wkong/perf-bot
---

# Kernel Validation

## Principles

**Correctness before performance.** Never benchmark a kernel that fails
correctness. A fast wrong answer is worthless. Always verify first, then
benchmark with the workload-profiling skill.

**Tolerance model.** Numerical tolerance depends on dtype and operation:

| Dtype | Recommended rtol | Recommended atol | Notes |
|-------|-----------------|-----------------|-------|
| float32 | 1e-5 | 1e-5 | Default for most operations |
| float16 | 1e-3 | 1e-3 | Standard for mixed-precision |
| bfloat16 | 1e-2 | 1e-2 | Lower mantissa precision |
| float8_e4m3fn | 5e-2 | 5e-2 | Very low precision, expect larger diffs |
| int8/int32 | 0 | 0 | Exact match expected |

Reductions and long chains accumulate error -- widen tolerance by 10x if
needed. If `max_abs_diff` is close to `atol`, the kernel is borderline;
investigate before accepting.

**Compare against the correct reference.** Always pass the ORIGINAL reference
code (e.g., `F.silu(gate) * up`) to `verify_kernel.py`, not the Triton/CuTe
kernel wrapper. If `max_abs_diff` reports 0.00 for FP16/BF16, the result is
suspicious -- a realistic diff for FP16 is 1e-4 to 1e-2.

## Workflow

Run `verify_kernel.py` after writing or modifying any GPU kernel:

```bash
python scripts/verify_kernel.py \
    --kernel-path kernel.py \
    --reference-code "def ref(x): return x * 2" \
    --input-shapes '{"x": [1024]}' \
    --input-dtypes '{"x": "float32"}' \
    --rtol 1e-3 --atol 1e-3
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--kernel-path` | Yes* | Path to Python file containing the kernel |
| `--reference-code` | Yes* | Python code defining the reference function |
| `--input-shapes` | Yes* | JSON dict mapping input names to shape lists |
| `--input-dtypes` | Yes* | JSON dict mapping input names to dtype strings |
| `--rtol` | No | Relative tolerance (default: 1e-3) |
| `--atol` | No | Absolute tolerance (default: 1e-3) |
| `--env-vars` | No | JSON dict of environment variables |
| `--timeout` | No | Execution timeout in seconds (default: 60) |
| `--mock` | No | Return mock data for testing (*not required with --mock) |

Output:
```json
{
  "correct": true,
  "max_abs_diff": 1.2e-7,
  "max_rel_diff": 3.4e-6,
  "backend_info": "enable_tile=0,actual=cuda",
  "details": "All outputs match within tolerance (rtol=1e-3, atol=1e-3)"
}
```

**Stop if `correct: false`.** Fix the kernel before benchmarking.

The script auto-discovers the kernel's callable (`run_kernel`, `forward`, or
`fused_*` prefix) and the reference function from the provided code. Supported
dtypes: float16, float32, bfloat16, float64, int8, int16, int32, int64, uint8,
float8_e4m3fn, float8_e5m2.

### TileIR Backend Verification

To verify a kernel works with both standard Triton and TileIR:

```bash
# Standard Triton
python scripts/verify_kernel.py ... --env-vars '{"ENABLE_TILE": "0"}'

# TileIR backend
python scripts/verify_kernel.py ... --env-vars '{"ENABLE_TILE": "1"}'
```

The `backend_info` field in the output confirms which backend was active.

## Error Handling

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `correct: false`, `max_abs_diff` ~ 1e-3 | Tolerance too tight for dtype | Widen `--rtol`/`--atol` per tolerance table above |
| `correct: false`, `max_abs_diff` > 1.0 | Algorithmic bug | Check kernel logic, reduction order, index math |
| `correct: false` only for large shapes | Overflow or accumulator precision | Use higher-precision accumulator, check intermediate dtypes |
| `No wrapper function found` | Kernel file missing callable | Ensure kernel exports `run_kernel`, `forward`, or a `fused_*` function |
| `Verification timed out` | Infinite loop or deadlock | Check kernel for sync issues; increase `--timeout` |
| `max_abs_diff` is 0.00 for FP16/BF16 | Reference function may be wrong | Verify reference code matches the original PyTorch operation |
| `name 'F' is not defined` | Missing import in reference code | Use full `torch.nn.functional.xxx` instead of `F.xxx` |
