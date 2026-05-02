# FP8 Dense GEMM Baseline — Shape Inventory

This is the baseline AVO scores against. **No nsys re-capture is needed** — the existing profile (`profile_super_nvfp4_disable_b12x.1.nsys-rep`) plus the b12x model architecture constants give us every shape we need.

## Source of truth

- **Kernel inventory:** `b12x/.n3/runs/baseline/nsys_top_kernels.md` (parsed from the nsys profile).
- **Architecture constants:** `b12x/benchmarks/benchmark_dense_gemm.py:33-34` and `benchmark_moe.py:255-263`.
- **Target model:** `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (`benchmark_moe.py:262`).

## Architectural shape constants

| Symbol | Value | Source |
|---|---|---|
| `NEMOTRON_HIDDEN_SIZE` | 4096 | `benchmark_dense_gemm.py:34` |
| `NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE` | 5376 | `benchmark_dense_gemm.py:33` |
| `TP_SIZE` (default) | 4 | `benchmark_moe.py:54` |
| Decode regime batches | `[1, 4, 32, 80]` | `verify_moe_perf.py:DEFAULT_BATCH_SIZES` |
| Prefill regime batches | `[8192, 16384, 24576, 32768]` | `benchmark_moe.py:CHUNKED_PREFILL_BATCH_SIZES` |
| Eager-prefill | `[16384, 32768]` | `benchmark_moe.py:EAGER_PREFILL_BATCH_SIZES` |

## nvjet kernel inventory (from baseline nsys)

Nine unique cuBLASLt heuristic kernels are present. Tile naming convention encodes `(outer_tile_MNK)_<stages>_(inner_tile_MNK)_tmaAB_bz_TNNN`:

| Time% | Kernel name | Outer MNK | Stages | Bucket |
|---|---|---|---|---|
| 14.4 | `nvjet_sm121_qqtst_mma_192x16x128_3_192x8x128_*` | 192×16×128 | 3 | qqtst, decode-ish |
| 14.0 | `nvjet_sm121_qqtst_mma_192x48x128_3_96x16x128_*` | 192×48×128 | 3 | qqtst, prefill-ish |
| 7.9 | `nvjet_sm121_qqtst_mma_64x64x128_6_16x64x128_*` | 64×64×128 | 6 | qqtst, square |
| 4.8 | `nvjet_sm121_tst_mma_16x8x768_2_16x8x768_*` | 16×8×**768** | 2 | tst, deep-K thin |
| 3.8 | `nvjet_sm121_tst_mma_64x8x256_2_16x8x256_*` | 64×8×256 | 2 | tst, decode |
| 2.3 | `nvjet_sm121_tst_mma_32x8x256_4_16x8x256_*` | 32×8×256 | 4 | tst, decode |
| 2.2 | `nvjet_sm121_tss_mma_32x8x256_4_16x8x256_*` | 32×8×256 | 4 | tss, decode |
| 1.5 | `nvjet_sm121_tst_mma_32x8x64_16_16x8x64_*` | 32×8×64 | 16 | tst, very thin |
| 0.6 | `nvjet_sm121_tst_mma_16x8x256_8_16x8x256_*` | 16×8×256 | 8 | tst, decode |

Total ~52% of GPU time across these. The `_TNNN` suffix = transposed-A, normal-B, normal-acc, normal-C layout.

### Dtype-tag interpretation

The `qq` / `t` / `s` letters in the prefix encode operand types. NVIDIA does NOT publish this map externally, but cross-referencing what the model *can* feed into a dense GEMM:

- **`qqtst`** (3 letters of dtype): plausibly NVFP4-A × NVFP4-B → BF16-C with FP8 block scales (the prefix `qq` for "quantized pair"). These would be NVFP4 dense GEMMs falling back to cuBLASLt because b12x is disabled. **Not the FP8 target.**
- **`tst`**: 3-letter dtype, plausibly FP8-A × FP8-B → BF16-C, no block scales. **This is the FP8 dense GEMM target.**
- **`tss`**: similar, possibly FP8-A × BF16-B → BF16-C (mixed-precision).

**This needs runtime confirmation.** AVO Phase-0 step (5 min): monkey-patch `torch._scaled_mm` and `torch.matmul` to log `(name, M, K, N, a.dtype, b.dtype, scale_a.dtype, scale_b.dtype)` per call during one inference pass; correlate with `nsys stats -r nvtx_kern_sum` to map kernel-name → call site.

## Target shape set for the new b12x FP8 dense GEMM

Combining the nvjet tile sizes with `K=4096` (Nemotron hidden) and standard projection dimensions:

### Decode-regime targets (M small, GEMV-ish)

| Layer | Shape (M × K × N) | TP | matches nvjet tile |
|---|---|---|---|
| Q proj (decode bs=1) | 1 × 4096 × 4096 | TP=4 → 1×4096×1024 | `tst_16x8x256_8_*` (very thin) |
| K/V proj (decode bs=1) | 1 × 4096 × 1024 | TP=4 → 1×4096×256 | `tst_32x8x64_16_*` |
| O proj (decode bs=1) | 1 × 4096 × 4096 | TP=4 → 1×1024×4096 | `tst_16x8x256_8_*` |
| Shared expert gate/up | 1 × 4096 × 5376 | TP=4 → 1×4096×1344 | `tst_16x8x256_*` |
| Shared expert down | 1 × 5376 × 4096 | TP=4 → 1×1344×4096 | `tst_16x8x768_2_*` (deep K) |
| Decode bs=4 / 32 / 80 | M ∈ {4,32,80} variants of above | | `tst_32x8x256_*`, `tst_64x8x256_*` |

### Prefill-regime targets (M large)

| Layer | Shape (M × K × N) | TP | matches nvjet tile |
|---|---|---|---|
| Shared expert down (prefill chunk) | 8192 × 5376 × 4096 | TP=4 → 8192×1344×4096 | `qqtst_192x48x128_3_*` |
| Shared expert gate/up (prefill chunk) | 8192 × 4096 × 5376 | TP=4 → 8192×4096×1344 | `qqtst_192x48x128_3_*` |
| Eager prefill chunk | 16384 / 32768 × 4096 × N | TP=4 | `qqtst_64x64x128_6_*` (larger square) |

### Canonical benchmark shapes for `verify_fp8_dense_gemm_perf.py`

Mirror `verify_moe_perf.py`'s pattern. AVO writes `verify_fp8_dense_gemm_perf.py` to measure these:

```python
# Decode regime
SHAPES_DECODE = [
    (1,  4096, 4096),   # Q/O proj, full TP=1
    (1,  4096, 5376),   # gate/up, full TP=1
    (1,  5376, 4096),   # down, full TP=1
    (4,  4096, 5376),
    (32, 4096, 5376),
    (80, 4096, 5376),
]
# Prefill regime
SHAPES_PREFILL = [
    (8192,  4096, 5376),
    (8192,  5376, 4096),
    (16384, 4096, 5376),
    (32768, 4096, 5376),
]
# TP=4 sharded (production deployment)
SHAPES_DECODE_TP4 = [(M, K, N // 4) for (M, K, N) in SHAPES_DECODE]
SHAPES_PREFILL_TP4 = [(M, K, N // 4) for (M, K, N) in SHAPES_PREFILL]
```

## Baseline measurement plan

For each `(M, K, N)` in the shape set, AVO produces a baseline JSON entry:

```json
{
  "M": 1, "K": 4096, "N": 4096,
  "ab_dtype": "float8_e4m3fn",
  "c_dtype": "bfloat16",
  "scaling": "per_tensor",
  "baseline_kernel": "torch._scaled_mm",
  "baseline_eager_median_us": <measured>,
  "baseline_graph_median_us": <measured>,
  "baseline_oracle": {"max_abs": <vs fp32>, "rmse": ..., "cos": ...}
}
```

Baseline measurement **does not require the model** — it only requires:
1. PyTorch's `torch._scaled_mm` (already in the container; `pip show torch → 2.11.0a0+...`).
2. Random `(M, K)` and `(K, N)` FP8 tensors with random per-tensor scales.
3. The same L2-flush + CUDA-event timing harness as `benchmark_moe.py`.

## Phase-0 deliverable for AVO

A single file `verify_fp8_dense_gemm_perf.py` that:
1. Generates random FP8 inputs at every shape in the table.
2. Times `torch._scaled_mm` (the baseline) on each shape, eager + graph mode, 10 warmup / 20 iters / 5 repeats, with L2 flush.
3. Validates against `(a.float() * scale_a) @ (b.float() * scale_b)` FP32 reference: `cos`, `max_abs`, `rmse`.
4. Emits JSON with the schema above to `.n3/runs/baseline/fp8_dense_baseline.json`.

Once that JSON exists, iteration v1 starts: AVO writes the b12x FP8 dense kernel and runs the same verifier against its own kernel.

## What we deliberately deferred

- Mapping every `nvjet_*` to a specific Python call site (would require nsys re-capture with backtrace + NVTX). Not needed because we've already enumerated the dense GEMM call types from the model architecture.
- Cross-validating `qq` vs `t` vs `s` against documented cuBLASLt naming. We treat the `tst`/`tss` family as FP8 candidates (because cuBLASLt has separate kernel families for FP8 and NVFP4 on Blackwell), and the `qqtst` family as NVFP4-block-scaled (matching the Nemotron weight type). If the runtime monkey-patch trace contradicts this, we revise.
