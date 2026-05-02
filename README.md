
`b12x` is an SM120/SM121 CuTe DSL kernel library for (primarily) NVFP4 LLM inference.

It is intentionally narrow. This is not a generic CUDA kernel collection or a
full model-serving stack. It does not intend to target any other GPU architectures,
including SM100. It is a focused package for a small number of high-performance
kernels plus the runtime glue needed to launch them cleanly from `sglang`/`vllm`.

Currently supported kernels:
- NVFP4 fused MoE GEMM
- NVFP4 dense GEMM
- BF16/FP8 paged attention
- Sparse MLA attention (for DSA/NSA only).

## FP8 dense GEMM tuning notes

For the DGX Spark / SM121 proxy shape `(M,K,N) = (32,5376,5376)`, the current best custom CUDA FP8 dense GEMM hot path is `TileM=32`, `TileN=32`, `StageK=256`, launched as `168` CTAs with `128` threads/CTA and `16384` bytes dynamic shared memory per CTA. The first two warps compute the `32x32` output tile while all four warps participate in shared-memory staging. Latest measured result was `102.464 us` median versus `99.408 us` median for `torch._scaled_mm` / cuBLASLt NVJet, with `max_abs_diff=0.015625` and `mean_abs_diff=3.26636e-7` against the PyTorch reference.

Rejected tuning points: `StageK384` was correct but slower at `135.776 us` median; `StageK128` was correct but slower at `134.288 us`; `StageK224` and `StageK448` failed with illegal memory accesses in the benchmark path; `TileM16,TileN32,StageK384` was correct but regressed to `181.456 us`; split-K was correct but regressed to about `156.7 us` because the partial/reduction overhead dominated.

```bash
pip install b12x
```

Ask your friendly neighborhood AI agent for further information on how to use this library.

## Hardware tuning notes (RTX PRO 6000 vs DGX Spark)

The kernels target SM120/SM121, but the bulk of the autotuning was done on
**RTX PRO 6000 Blackwell (SM120, ~188 SMs)**. DGX Spark (SM121, ~96 SMs) is
supported via targeted patches layered on top, not via independent tuning.

Evidence:
- `b12x/moe/tuning/decode.max_active_clusters.py` ladders top out at
  `MAX_ACTIVE_CLUSTERS = 188` (exactly RTX PRO 6000's SM count). On Spark these
  are clamped at runtime by `get_max_active_clusters(device)`; the ladder values
  themselves are not Spark-derived.
- Spark-specific code paths gate on `get_num_sm(...) <= 96` (see
  `b12x/integration/tp_moe.py:2165` and `:2395`) and apply a separate cap via
  `B12X_RELU2_BS1_SPARK_MICRO_CAP` (default 48), introduced for the
  bandwidth-limited single-token ReLU² decode path.
- `b12x/cute/runtime_patches.py:_patch_cutlass_sm120_blockscaled_arch_check`
  extends CUTLASS's `MmaSM120BlockScaledOp` to also accept `Arch.sm_121a`.
- Relevant commits: `7757627` (DGX Spark fixes), `99c0a80` (Optimize
  single-token relu2 MoE decode for DGX Spark), `dadf95f` (bump Spark relu2
  bs=1 MAC cap 42 → 48), `14b4214` (SM121 fixes), `73bfd50` (revert suboptimal
  sm121 changes).

Re-tuning Spark further is a sweep-and-regenerate exercise, not kernel surgery:
- `scripts/sweep_moe_decode_max_active_clusters.py` →
  `scripts/generate_moe_decode_mac_tuning.py` regenerates the MoE decode ladder.
- The tuning registry (`b12x/moe/tuning/registry.py`) is keyed by
  `(regime, backend)`; extending the key to include SM count would let a
  Spark-specific ladder coexist with the RTX 6000 one without touching kernels.
- Cutover thresholds (`B12X_STATIC_COMPACT_CUTOVER_PAIRS`,
  `B12X_MICRO_COMPACT_CUTOVER_PAIRS`) and dynamic knobs
  (`B12X_DYNAMIC_ENABLE_MULTICTA`, `B12X_DYNAMIC_CHUNK_MULTIPLIER`) are env-
  overridable and worth sweeping on Spark.
- MMA tilings (`mma_tiler_mn`, `output_tile_count_n`) are not currently swept;
  adding that would be net-new harness work.
- No `regime="prefill"` ladders exist yet — only decode is tuned.
