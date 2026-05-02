# FP8 Dense GEMM — Implementation Brief for AVO

## Goal

Add a high-performance **FP8 dense GEMM** to b12x for SM121 (DGX Spark / GB10), modeled on b12x's existing NVFP4 block-scaled dense GEMM. The new kernel must beat the current FP8 dense GEMM baseline (whatever cuBLAS/cuBLASLt path PyTorch invokes today) on the shapes that actually fire in the production serving paths captured in the nsys profile at the workspace root.

## Reference template — DO NOT reinvent, mirror

b12x already has a complete, production-grade **NVFP4** block-scaled dense GEMM. Use it as the structural template for every new file you write. AVO must read each of these before producing any code:

| Concern | Reference file | What to lift |
|---|---|---|
| Kernel & launcher | `b12x/b12x/gemm/dense.py` | `class DenseGemmKernel`, `class _DenseGemmLaunch`, `_get_compiled_dense_gemm`, `dense_gemm()` (line 1767). The wrapper signature is the surface AVO mirrors. |
| Public exports | `b12x/b12x/gemm/__init__.py` | Module re-exports: `DenseGemmKernel`, `dense_gemm`. Add the FP8 equivalent here. |
| Bench harness | `b12x/benchmarks/benchmark_dense_gemm.py` | L2-flush, CUDA-event repeated trials, comparison against FlashInfer-CUTLASS (`mm_fp4`), cosine-similarity correctness gate (`COSINE_THRESHOLD=0.999999`), `BatchSize`/`TimingStats` schema. Mirror this for FP8. |
| Correctness oracle | `b12x/tests/test_gemm_stack.py` | `_make_quantized_operand`, `_run_dense_gemm`, alpha/scale handling, `quantize_grouped_nvfp4_torch` analogue. The pattern: `(a, sfa) × (b, sfb) → c` plus per-tensor `alpha` scalar. |
| Oracle tolerance | The NVFP4 path uses cosine ≥ 0.999999 (`COSINE_THRESHOLD` in `benchmark_dense_gemm.py`). FP8 e4m3 has ~3 mantissa bits, so the FP8 oracle should accept a looser bound — propose `cos ≥ 0.9995` and `max_abs ≤ 0.01 × ||ref||_∞` and validate empirically against PyTorch `torch._scaled_mm`. |
| CuTeDSL utilities | `b12x/b12x/cute/utils.py` | `current_cuda_stream`, `cutlass_to_torch_dtype`, `get_cutlass_dtype`, `get_max_active_clusters`, `get_num_sm`, `make_ptr`, swizzle helpers. Reuse, don't duplicate. |
| Runtime patches | `b12x/b12x/cute/runtime_patches.py` | The `MmaSM120BlockScaledOp` arch-check patch already extends to `Arch.sm_121a`. Don't touch. |
| Cutlass MMA atom | `cute.MMA_Atom<cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<...>>` (visible in nsys output) is the FP4-style atom. The FP8 equivalent is **`SM120_FP8_MMA`** (or whatever cutlass-dsl 4.4.x exposes for sm120/121 FP8 — search `cutlass.cute.nvgpu.warp.mma` and the cutlass-examples in `.claude_docs/cutlass-examples/CuTeDSL/blackwell_geforce/dense_gemm.py`). |

## Existing `dense_gemm()` API surface (the contract to mirror)

```python
# b12x/b12x/gemm/dense.py:1767
def dense_gemm(
    lhs: Tuple[torch.Tensor, torch.Tensor],   # (a, sfa)  — NVFP4: packed e2m1 + e4m3 block scales
    rhs: Tuple[torch.Tensor, torch.Tensor],   # (b, sfb)
    out: Optional[torch.Tensor] = None,
    *,
    ab_dtype: str,                             # "float4_e2m1fn" today; AVO adds "float8_e4m3fn" / "float8_e5m2"
    sf_dtype: str,                             # "ufloat8_e4m3" for NVFP4 block scales; FP8 typically uses "float32" per-tensor or per-row
    c_dtype: str,                              # "bfloat16", "float16", or "float32"
    sf_vec_size: int,                          # 16 for NVFP4 block scales; 1 for FP8 per-tensor; or per-row N
    sm_count: Optional[int] = None,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    alpha: Optional[torch.Tensor] = None,      # scalar = 1/(scale_a * scale_b)
    alpha_dtype: Optional[str] = None,
) -> torch.Tensor: ...
```

**Decision for AVO:** extend this single function to accept FP8 (preferred — keeps one entry point, one tuner, one bench), or fork a `dense_gemm_fp8()` if the FP8 mainloop differs enough that one signature is awkward. Recommendation: try the extend path first, fork only if the kernel really needs different launch parameters.

## Phase 0 — calibration is mandatory

Before Phase 0's `verify_fp8_dense_gemm_perf.py` is considered final, AVO MUST validate that the problem shapes it picks reproduce the per-kernel timings observed in the nsys baseline.

Mechanism:
1. Write a tiny `b12x/benchmarks/calibrate_baseline.py` that, for each candidate `(M, K, N)`:
   - Generates random FP8 inputs.
   - Runs `torch._scaled_mm(a, b.T, scale_a, scale_b, out_dtype=torch.bfloat16)` 20 iters wrapped in `cudart.cudaProfilerStart/Stop`.
2. Run it under nsys with capture-range:
   ```
   command_on_spark.sh nsys profile -c cudaProfilerApi --capture-range=cudaProfilerApi --capture-range-end=stop \
     -t cuda-sw,cublas -o .n3/runs/baseline/calibrate.nsys-rep \
     -- python -m benchmarks.calibrate_baseline
   ```
3. `nsys stats -r cuda_gpu_kern_sum` over the result. For each `(M, K, N)`:
   - The kernel name (e.g. `nvjet_sm121_tst_mma_16x8x256_8_*`) must match the production nsys's name.
   - The median per-instance time must match within ±10% of production median.
4. If a shape doesn't reproduce its expected nsys kernel/timing, drop it or annotate it. Only confirmed shapes go into `fp8_dense_baseline.json`.

The expected mapping (from `nsys_top_kernels.md`):

| Problem (M, K, N) | Expected kernel | Expected median µs |
|---|---|---|
| (1, 4096, 4096) | `tst_16x8x256_8_*` | ~17 |
| (1, 4096, 5376) | `tst_16x8x256_8_*` or `tst_32x8x64_16_*` | 17–175 |
| (1, 5376, 4096) | `tst_16x8x768_2_*` | ~297 |
| (1, 4096, 1024) (KV TP=4) | `tst_32x8x64_16_*` | ~175 |
| (M ∈ {4,32,80}, 4096, 5376) | `tst_32x8x256_*` or `tst_64x8x256_*` | 39–70 |
| (8192, 4096, 5376) | `qqtst_192x48x128_3_*` (NVFP4 fallback) | ~339 |

Calibration tells AVO whether `torch._scaled_mm` is actually the right baseline. If no shape produces a `tst_*` family kernel, then `torch._scaled_mm` on this PyTorch build doesn't route through the same cuBLASLt heuristic the production model used — that's a real finding, not a failure, and AVO should flag it.

## Baseline shape inventory — already enumerated

The shape set is fully determined by:
- The 9 unique `nvjet_sm121_*_mma_*` tile sizes in the existing nsys profile (parsed at `.n3/runs/baseline/nsys_top_kernels.md`).
- The Nemotron 3 Super architecture constants in `b12x/benchmarks/benchmark_dense_gemm.py:33-34`: `HIDDEN=4096`, `INTERMEDIATE=5376`, plus standard attention projections and TP=4 sharding.

**No nsys re-capture is required.** The full target shape table — decode + prefill regimes, with TP=1 and TP=4 variants — is committed at `.n3/runs/baseline/fp8_dense_baseline.md`. Read that file before writing any kernel code.

The dtype-tag interpretation of `qqtst` / `tst` / `tss` is documented in that file. Brief summary: `tst` and `tss` families are the likely FP8 dense targets; `qqtst` is the NVFP4-block-scaled fallback (cuBLASLt picking up b12x's NVFP4 path when b12x is disabled). If you want to confirm the dtype tag during Phase 0, monkey-patch `torch._scaled_mm` and `torch.matmul` to log `(M, K, N, dtypes)` for one inference call — but this is optional, not blocking.

## Deliverable spec

AVO must produce, in order, in the same b12x style:

1. **`b12x/b12x/gemm/dense.py`** — extended to accept `ab_dtype="float8_e4m3fn"` (and `"float8_e5m2"` if cheap). Keep all existing NVFP4 callers working — gate on `ab_dtype` inside the kernel selection / mainloop.
2. **`b12x/benchmarks/benchmark_dense_fp8_gemm.py`** — mirror `benchmark_dense_gemm.py`. Reference backend: `torch._scaled_mm` (or whatever the baseline turns out to be). Same flags (`--warmup 10 --iters 20 --repeats 5 --flush-l2 --cuda-graph`). Same JSON schema:
   ```json
   {
     "shapes": {"<M>x<K>x<N>": {
       "eager_median_us": ..., "eager_min_us": ...,
       "graph_median_us": ..., "graph_min_us": ...,
       "ref_median_us": ..., "speedup_vs_ref": ...,
       "oracle": {"max_abs": ..., "rmse": ..., "mean_abs": ..., "cos": ...}
     }}
   }
   ```
3. **`b12x/benchmarks/verify_fp8_dense_gemm_perf.py`** — fixed-shape perf gate (mirror `verify_moe_perf.py`). Output JSON is what the AVO loop parses to score each iteration.
4. **`b12x/tests/test_fp8_dense_gemm.py`** — correctness tests (mirror `test_gemm_stack.py`). At least: per-tensor scale, per-row scale, BF16 output, FP16 output, two cluster shapes, range of `(M, K, N)` covering the production shapes.
5. **`b12x/b12x/gemm/tuning/` (new dir)** — register a `MaxActiveClustersPolicy` ladder for FP8 dense GEMM keyed by `(regime="dense", backend="fp8")`. Pattern from `b12x/b12x/moe/tuning/`.
6. **No changes to `b12x/cute/runtime_patches.py`** unless ncu/nsys evidence forces it.

## Success criteria (AVO loop scores against this)

For each shape that appears in the nsys baseline (the production set), AVO's kernel must satisfy:

- **Correctness:** oracle metrics pass (cos ≥ 0.9995, max_abs ≤ 1% of `||ref||_∞`). Validate against `torch._scaled_mm` and against an FP32 reference computed with `(a.float() * scale_a) @ (b.float() * scale_b)`.
- **Performance:** ≥ +5% sustained reduction in `graph_median_us` vs the current baseline (cuBLAS/cuBLASLt nvjet) on every measured shape, no regression on any shape, stable across 5 repeats.
- **Numerical safety:** no NaN/Inf in graph-mode replay across 100 iterations of random inputs.

## Per-iteration loop (AVO autopilot)

Same shape as `b12x/.n3/TASK.md`'s iteration loop, but with FP8-specific commands:

1. Plan one hypothesis (read source first: `dense.py`, the bench, the relevant cutlass-examples notebook).
2. Implement.
3. Smoke import: `command_on_spark.sh python -c "from b12x.gemm.dense import dense_gemm"`.
4. Correctness gate: `command_on_spark.sh pytest -x b12x/tests/test_fp8_dense_gemm.py b12x/tests/test_gemm_stack.py`.
5. Score: `command_on_spark.sh python -m benchmarks.verify_fp8_dense_gemm_perf > .n3/runs/v${N}/verify_fp8.json`.
6. Compare to baseline: parse JSON; if any shape regresses ≥ 0.5% OR no shape improves ≥ 0.5%, reject and re-base on v(N-1).
7. Profile candidate-best: nsys + ncu via the recipes in `.n3/skills/nsight-systems/SKILL.md` and `.n3/skills/nsight-compute-analysis/SKILL.md`.
8. Bump version + commit.

## Pointers AVO should always have open

- `.n3/skills/cute-kernel-writing/SKILL.md` — CuTe DSL idioms.
- `.n3/skills/kernel-validation/SKILL.md` — correctness-before-perf protocol.
- `.n3/skills/nsight-systems/SKILL.md` — nsys recipes for the baseline analysis.
- `.n3/skills/nsight-compute-analysis/SKILL.md` — SOL%, roofline, classification once a candidate is best.
- `.claude_docs/cutlass-examples/CuTeDSL/blackwell_geforce/dense_gemm.py` — canonical SM120/121 dense GEMM example (FP4/FP8 patterns).
- `.claude_docs/cutlass-examples/CuTeDSL/notebooks/tour_to_sol_gemm.ipynb` — tour from naive to SOL GEMM in CuTeDSL.

## Out of scope

- Grouped GEMM (MoE) — that's a separate kernel family in b12x.
- FP8 attention — already exists in `b12x/attention/`, not part of this task.
- Quantization itself — b12x consumes pre-quantized FP8 inputs; quantization happens upstream.
- Recipe / model integration — once the kernel is fast and correct, plumbing it into `serve/model/recipe_*.py` is a follow-up.
