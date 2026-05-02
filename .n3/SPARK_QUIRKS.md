# SM121 / DGX Spark — known quirks AVO must respect

NV-internal environmental issues that don't appear in any public doc. AVO **cannot infer these**; both are sourced from `#sm121-dgx-spark`-style Slack threads (Luke Alonso, Brian Ryu, Xiaodong Huang, late April 2026).

## 1. CUTLASS DSL version pinning + arch-check patch

### Symptom
With `nvidia-cutlass-dsl==4.4.2` (libs-base + libs-cu13 also 4.4.2), MoE/FP8 GEMM tests fail on Spark with:
```
Unexpected instruction types specified for '_mma'
```
on the inline PTX:
```
_mma.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

### Working combinations
1. **4.4.1 + manually remove the sm_120a admissible_archs check in cutlass.cute.nvgpu.warp.mma** — confirmed working by Xiaodong (Apr 2026).
2. **4.4.2 with `pip install --no-deps`** — Luke's confirmed alternative (avoids whatever sub-package gets pulled in by default that breaks the lookup).

### What to remove
File (inside the cutlass-dsl wheel): `cutlass/cute/nvgpu/warp/mma.py`, around line 132–138:
```python
class MmaSM120BlockScaledOp(...):
    admissible_archs = [
        "sm_120a",
    ]
```
Add `"sm_121a"` to that list, or just delete the `admissible_archs` check entirely. Verify the patch is in place before running any FP8 / NVFP4 MMA.

The check is **gone in 4.5** but 4.5 wasn't on PyPI as of late Apr 2026, so we patch 4.4.x in place.

### What `build-sm121.sh` does

`.claude_docs/scripts/build-sm121.sh` now:
1. Installs `nvidia-cutlass-dsl==4.4.1` (and the two libs- packages) with `--force-reinstall --no-deps`.
2. After install, sed-patches `mma.py` to add `sm_121a` to the admissible archs.
3. Runs the smoke pytest (`test_cutlass_runtime_patches.py` + `test_fp4_quantization.py`) to confirm. If the smoke fails with the `_mma` error, the patch didn't take and the script aborts.

### Failure mode AVO will hit if it ignores this

- Any `import b12x` path that touches the GEMM kernel triggers JIT compile.
- JIT emits PTX that the assembler rejects (`Unexpected instruction types`).
- The error surfaces deep in the cute compile chain — easy to misread as "kernel bug" when it's actually a version/check problem.
- **If AVO sees that exact error string, the fix is THIS file's recipe, not a code change in b12x.**

## 2. `nsys profile -t cuda` produces empty kernel captures on Spark

### Symptom
```
nsys profile -t cuda-sw,nvtx ... -- python script.py
nsys stats -r cuda_gpu_kern_sum profile.nsys-rep
```
returns an empty kernel table. There's a Spark driver bug where the standard CUDA tracing backend doesn't get any kernel records.

### Workaround
Replace `-t cuda` with `-t cuda-sw` in every nsys invocation. The "software" backend works.

### Where this matters in the AVO loop

**Every** nsys command in the BRIEF and TASK.md must be updated. Specifically:

- Phase 0 calibration (`calibrate_baseline.py` capture).
- Per-iteration candidate-best nsys capture.
- ncu (uses its own profiler, not affected — leave ncu commands alone).

Canonical Spark-aware nsys recipe:
```
command_on_spark.sh nsys profile -c cudaProfilerApi --capture-range=cudaProfilerApi --capture-range-end=stop \
  -t cuda-sw,nvtx,cublas \
  -o .n3/runs/v${N}/nsys.rep \
  -- python -m benchmarks.benchmark_dense_fp8_gemm --profile-once --cuda-graph
```

Stats invocation is unchanged (`-t` is a profile-time flag, not a stats-time one):
```
command_on_spark.sh nsys stats -r cuda_gpu_kern_sum,cuda_kern_exec_sum,nvtx_kern_sum \
  .n3/runs/v${N}/nsys.rep > .n3/runs/v${N}/nsys.txt
```

### Failure mode AVO will hit if it ignores this

- nsys appears to succeed (exits 0, produces a `.nsys-rep` of normal size).
- `nsys stats -r cuda_gpu_kern_sum` returns an empty table.
- AVO concludes "no kernels ran, my benchmark must be broken" — wastes an iteration.
- **If AVO sees an empty kernel summary, the fix is `-t cuda` → `-t cuda-sw`, not a benchmark-script change.**

## Source

These quirks are from internal Slack discussion (`Xiaodong Huang`, `Luke Alonso`, `Brian Ryu`, late Apr 2026), surfaced to me by the project owner during the AVO chat-mode validation. Until 4.5 ships and a driver fix lands, this file is authoritative for the Spark host (`p4242-0053`, NVIDIA GB10 / SM121).
