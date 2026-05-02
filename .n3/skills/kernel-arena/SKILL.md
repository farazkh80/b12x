---
name: kernel-arena
description: >
  Search and contribute to the centralized Kernel Arena — a git-backed repository
  of GPU kernel exemplars. Before writing a kernel from scratch, search the arena
  for the closest existing solution. After producing a faster kernel, upload it
  back. Triggers: kernel reuse, arena search, arena upload, find existing kernel,
  share kernel, kernel exemplar, optimization reuse.
user-invocable: false
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
  documentation: https://gitlab-master.nvidia.com/wkong/perf-bot
---

# Kernel Arena

The Kernel Arena is a centralized repository of GPU kernel exemplars at
`perf-bot.nvidia.com`. It stores **optimization strategies** (not math formulas)
— one exemplar per optimization skeleton, which you adapt to your specific case.

## Pre-Gate: Arena Search (MANDATORY)

**Execute this BEFORE writing any kernel.** This is not optional.

1. Fetch the catalog:

```bash
python scripts/arena_client.py catalog
```

2. If the arena server is unreachable, log the error and proceed from scratch.
   Do not block kernel generation on arena availability.

3. Scan the returned problems — find the closest match for your task.
   Match by operation type first, then variant, then arch/API.

4. If match found, pull the exemplar:

```bash
python scripts/arena_client.py pull --exemplar-id <id>
```

5. Use the pulled code as your starting point. The exemplar follows the
   fixed-name contract (`kernel_fn`, `reference_fn`, `get_inputs()`).
   For CUDA exemplars, the response includes `companion_files` — write
   all companion files (`.cu`, `.cpp`) to disk alongside `kernel.py`.

6. If no match found, proceed from scratch using your kernel-writing skill.
   Note this as a "new problem candidate" for the Post-Gate.

## Post-Gate: Arena Upload Evaluation (MANDATORY)

**Execute this AFTER writing a kernel.** This is not optional.

**CRITICAL: You MUST attempt to run the kernel before deciding to skip upload.**
Do not assume GPUs are unavailable — always try `python kernel.py` or the
verification script first. Only if execution fails with an actual error
(e.g., `CUDA error`, `no CUDA GPUs are available`) may you skip upload.
Writing `.arena-upload-failed.json` without attempting execution is a protocol
violation.

1. Check for `.arena-uploaded.json` in your artifact directory — skip if already uploaded.

2. **Verify and benchmark the kernel** (MANDATORY before any upload decision):
   a. Run the kernel's verification script or `python kernel.py` to confirm
      correctness against the reference implementation
   b. If verification fails with an execution error, fix the kernel or log
      the actual error and proceed to step 5

3. If improving an existing exemplar:
   a. Benchmark BOTH old and new kernels on the same hardware using the
      relevant kernel-writing skill's `benchmark_kernel.py`
   b. Confirm new is at least **1% faster**
   c. If faster, submit with provenance:

```bash
python scripts/arena_client.py submit --payload-file upload.json
```

   d. If not faster, skip upload and log the reason.

4. If this is a new problem:
   a. Apply the **Adaptation Test** (see `references/adaptation-test.md`)
      to confirm no existing problem in the catalog fits
   b. Submit as new (set `based_on_exemplar_id` to null):

```bash
python scripts/arena_client.py submit --payload-file upload.json
```

5. On successful upload, write `.arena-uploaded.json` in the artifact directory:

```json
{
  "exemplar_id": "<uuid>",
  "version_id": "<uuid>",
  "uploaded_at": "<ISO-8601>"
}
```

6. On **server** failure (server unreachable, network timeout), write
   `.arena-upload-failed.json`:

```json
{
  "error": "<error message>",
  "attempted_at": "<ISO-8601>"
}
```

   This applies ONLY to network/server errors during `arena_client.py submit`.
   Do NOT use this for skipping verification — you must always attempt to run
   the kernel first (step 2).

## Upload Payload Format

Write a JSON file `upload.json` with this structure:

```json
{
  "problem": {
    "operation": "<optimization-pattern-name>",
    "variant": "<variant-name>",
    "description": "<human-readable description>"
  },
  "exemplar": {
    "kernel_api": "<triton|cute_dsl|cuda|ctm|nv_triton>",
    "arch_family": "<ampere|hopper|blackwell>",
    "traits": ["<free-form-tag>", "..."],
    "description": "<what makes this exemplar distinct>"
  },
  "kernel_source": "<full kernel.py file content>",
  "companion_files": {
    "kernel.cu": "<CUDA kernel source (optional, for CUDA kernels)>",
    "kernel_binding.cpp": "<pybind11 binding source (optional, for CUDA kernels)>"
  },
  "provenance": {
    "based_on_exemplar_id": "<uuid-or-null>",
    "based_on_git_sha": "<sha-or-null>",
    "change_summary": "<description of optimization delta>"
  },
  "benchmark": {
    "device": "<GPU-model>",
    "old_time_us": null,
    "new_time_us": 50.0,
    "correctness_verified": true
  },
  "uploaded_by": "<user>",
  "session_id": "<session-id>"
}
```

## Adaptation Test

Before creating a new problem, apply the Adaptation Test:

> Can you adapt an existing exemplar by changing only the computation logic
> (the math in the inner loop), while keeping the optimization skeleton
> (tiling, memory access pattern, synchronization, pipeline) intact?

- If **yes** → reuse the existing problem
- If **no** → create a new problem

See `references/adaptation-test.md` for detailed examples.
