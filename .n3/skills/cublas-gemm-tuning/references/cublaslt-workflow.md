# cuBLASLt heuristic tuning workflow

The whole skill is a wrapper around two cuBLASLt API calls. There is no
custom algo enumeration — `cublasLtMatmulAlgoGetHeuristic` already returns
the top-N candidates ranked by predicted performance over the full
algo×tile×stages×splitK×swizzle space.

## The two API calls

```c
// 1) Ask for top-N candidates (1..256). Heuristic is fast (~1 ms) — call once.
cublasLtMatmulPreference_t pref;
cublasLtMatmulPreferenceCreate(&pref);
cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                     &workspace_bytes, sizeof(workspace_bytes));

cublasLtMatmulHeuristicResult_t results[100];
int returned = 0;
cublasLtMatmulAlgoGetHeuristic(handle, op_desc, A_layout, B_layout, C_layout, D_layout, pref,
                               /*request_count=*/100, results, &returned);

// 2) Run each one. Pick fastest.
for (int i = 0; i < returned; ++i) {
    // warmup + L2 flush + iters × cudaEventRecord around cublasLtMatmul(..., &results[i].algo, ...)
    // record median_us for results[i]
}
```

## Heuristic result fields

```c
typedef struct {
    cublasLtMatmulAlgo_t algo;          // pass to cublasLtMatmul
    size_t              workspaceSize;  // bytes needed; bound by pref MAX
    cublasStatus_t      state;          // CUBLAS_STATUS_SUCCESS if usable
    float               wavesCount;     // cuBLASLt's predicted GPU wave count
    int                 reserved[4];
} cublasLtMatmulHeuristicResult_t;
```

`wavesCount < 1.0` is "fits in one wave"; >1 means the kernel issues that many
waves of CTAs across all SMs. For decode-shaped GEMMs (small M) you'll often
see `wavesCount=0.04..0.1` and the candidates differ mainly by tile/swizzle.

## Reading per-algo config

After heuristic returns, read what cuBLASLt picked using
`cublasLtMatmulAlgoConfigGetAttribute` — these are the same knobs we'd otherwise
sweep manually:

```c
int v;
cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &v, sizeof(v), nullptr);  // algo class
cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &v, sizeof(v), nullptr);
cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &v, sizeof(v), nullptr);
cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &v, sizeof(v), nullptr);
cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &v, sizeof(v), nullptr);
cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &v, sizeof(v), nullptr);
```

These are the keys that go into `tune_gemm`'s JSON output, and the keys you'd
write into a kernel-resolution registry to reproduce the picked algo.

## Reproducing a tuned algo

To reuse a tuned `(algo_id, tile_id, stages_id, splitk_num, swizzling)` without
re-running the heuristic:

```c
cublasLtMatmulAlgo_t algo;
cublasLtMatmulAlgoInit(handle, compute, scale_type, a_dt, b_dt, c_dt, d_dt, algo_id, &algo);
cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(tile_id));
cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages_id, sizeof(stages_id));
cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk_num, sizeof(splitk_num));
cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle));

cublasLtMatmulHeuristicResult_t check;
cublasLtMatmulAlgoCheck(handle, op_desc, la, lb, lc, ld, &algo, &check);  // validates + reports workspace
// Now use &algo with cublasLtMatmul.
```

## When the heuristic returns 0 candidates

| Status | Meaning | What to try |
|---|---|---|
| 7 (`INVALID_VALUE`) | layout/dtype combo cuBLASLt rejects | check leading-dim alignment (FP8 needs ld % 16 == 0); try different transpose; check scale mode for FP8/INT8 |
| 14 (`NOT_INITIALIZED`) | handle/desc not initialized | confirm `cublasLtCreate` succeeded; matmul desc + all 4 layouts created |
| 15 (`NOT_SUPPORTED`) | dtype combo not supported on this device | mixed dtypes are fine for fp16+fp16+fp16, less so for fp8+fp8+fp32 on some arches; consult cuBLAS support matrix |

If you bumped `--request-count` to 256 and still get 0, you likely have a
fundamentally unsupported combination, not a heuristic miss.
