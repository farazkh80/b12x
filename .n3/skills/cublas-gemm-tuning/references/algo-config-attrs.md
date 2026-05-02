# cuBLASLt algo config attributes

These are the attributes cuBLASLt's heuristic returns and that
`tune_gemm` extracts into the JSON output. They're also the keys you'd set
when reproducing a tuned algo without re-running heuristic.

## Per-algo attributes

| Attribute | Type | Meaning |
|---|---|---|
| `CUBLASLT_ALGO_CONFIG_ID` | int32 | Algorithm class. Each ID is a different kernel family (e.g. cublasLt-internal nvjet variants). 0..127 typical. |
| `CUBLASLT_ALGO_CONFIG_TILE_ID` | int32 | Tile shape index. Per-algo capability list via `CUBLASLT_ALGO_CAP_TILE_IDS`. |
| `CUBLASLT_ALGO_CONFIG_STAGES_ID` | int32 | Pipeline stage count index. Per-algo via `CUBLASLT_ALGO_CAP_STAGES_IDS`. |
| `CUBLASLT_ALGO_CONFIG_SPLITK_NUM` | int32 | Number of K-splits for split-K reduction. 1 = no split. Larger values trade extra reduction kernel for more parallelism on small problems. |
| `CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME` | int32 | Reduction scheme for split-K (NONE / IN_PLACE / OUT_OF_PLACE / COMPUTE_TYPE / MASK). 0 = no reduction. |
| `CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING` | int32 | CTA scheduling order. 0 = default linear, 1 = swizzle (better L2 reuse for some shapes). |
| `CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION` | int32 | Algo-specific knob; rarely user-relevant. |
| `CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID` | int32 | Inner mma shape index (Hopper+). |
| `CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID` | int32 | Thread block cluster size (Hopper+). |

## Capability attributes (read from algo, not configured)

Use `cublasLtMatmulAlgoCapGetAttribute` to ask which configs an algo supports.

| Attribute | Returns | Meaning |
|---|---|---|
| `CUBLASLT_ALGO_CAP_TILE_IDS` | int[] | Valid `TILE_ID` values for this algo |
| `CUBLASLT_ALGO_CAP_STAGES_IDS` | int[] | Valid `STAGES_ID` values |
| `CUBLASLT_ALGO_CAP_SPLITK_SUPPORT` | int | 0/1 — does algo support splitK? |
| `CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT` | int | 0/1 |
| `CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK` | int | bitmask of supported reduction schemes |
| `CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX` | int | upper bound for `CUSTOM_OPTION` |
| `CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT` | int | 0/1 — D = A·B (vs in-place A·B → A or B) |
| `CUBLASLT_ALGO_CAP_TILE_M_SHIFT/.../TILE_N_SHIFT` | int | tile dim hints |

`tune_gemm` doesn't use these — they're for if you want to manually iterate
the full algo×tile×stage×... space (overkill since heuristic already covers it).

## Tile_ID decoding

Tile IDs map to symbolic names via `cublasLtMatmulTile_t` enum in `cublasLt.h`.
Examples on Hopper / Blackwell:

```
CUBLASLT_MATMUL_TILE_64x64    = 6
CUBLASLT_MATMUL_TILE_128x64   = 8
CUBLASLT_MATMUL_TILE_64x128   = 9
CUBLASLT_MATMUL_TILE_128x128  = 10
CUBLASLT_MATMUL_TILE_256x128  = 14
CUBLASLT_MATMUL_TILE_128x256  = 15
```

The numeric tile_id in `tune_gemm`'s JSON is the enum value; cross-reference
against `cublasLt.h` to read the MNK tile shape.

## Reading what cuBLASLt picked

For every entry in `tune_gemm`'s `candidates` JSON list, the
`(algo_id, tile_id, stages_id, splitk_num, swizzling, reduction_scheme)`
tuple uniquely identifies the kernel cuBLASLt would launch. Two heuristic
results with identical tuples are duplicates (will time identically).
