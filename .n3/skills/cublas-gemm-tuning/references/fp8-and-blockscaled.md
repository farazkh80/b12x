# FP8 and block-scaled GEMM specifics

Quick reference for the FP8 setup that cuBLASLt requires beyond the standard
fp16/bf16 path.

## What FP8 needs (any arch)

| Setup | Value |
|---|---|
| A dtype | `CUDA_R_8F_E4M3` or `CUDA_R_8F_E5M2` |
| B dtype | `CUDA_R_8F_E4M3` or `CUDA_R_8F_E5M2` |
| C/D dtype | `CUDA_R_16BF` (most common), `CUDA_R_16F`, `CUDA_R_32F`, or one of the FP8 types |
| Compute | `CUBLAS_COMPUTE_32F` (mandatory; FP8 epilogue accumulates in FP32) |
| Scale type | `CUDA_R_32F` |
| Layout | TN canonical: `--trans-a T --trans-b N` (A K-major, B K-major, D M-major) |
| Leading dims | All ld must be multiples of 16 bytes |
| Min M/N/K | typically 16; FP8 e5m2 needs K ≥ 32 |
| `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` | device fp32 scalar (per-tensor) |
| `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER` | device fp32 scalar |
| `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER` | device fp32 scalar (output rescale) |
| `CUBLASLT_MATMUL_DESC_FAST_ACCUM` | int8 0 or 1; 1 is faster for E5M2 |

`tune_gemm` sets all of these automatically when `--a-dtype` is one of the FP8
or INT8 types, with unit scales (`1.0`).

## Per-tensor vs block-scaled (Blackwell sm_120 caveat)

Blackwell-GeForce (sm_120/sm_121, including GB10/Spark) FP8 mma instructions
are **block-scaled by hardware**. The PTX is:

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
```

Each MMA tile takes a `ue8m0` (8-bit unsigned exponent-only) scale per 32-element
block of A and per 32-element block of B. There is no "unscaled FP8 mma" on
sm_120 — the per-tensor `kind::f8f6f4` exists but most cuBLASLt heuristic algos
don't pick it for sm_120 because the block-scaled hardware path is faster.

Consequence: `cublasLtMatmulAlgoGetHeuristic` returns 0 candidates with status
`INVALID_VALUE` (7) on sm_120 when only per-tensor scales are attached. To make
it work, attach **VEC32_UE8M0** scale tensors:

```c
cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));

// Allocate scale tensors. Each scale is one ue8m0 byte per 32-element block.
//   A is (M, K) → scale tensor A_scale is (M, K/32) of uint8
//   B is (K, N) → scale tensor B_scale is (N, K/32) of uint8
// Initialize to 0x7F to encode 2^0 = 1.0 (= "no scaling").
size_t a_scale_bytes = (size_t)M * ((K + 31) / 32);
size_t b_scale_bytes = (size_t)N * ((K + 31) / 32);
uint8_t* d_a_scale; uint8_t* d_b_scale;
cudaMalloc(&d_a_scale, a_scale_bytes);
cudaMalloc(&d_b_scale, b_scale_bytes);
cudaMemset(d_a_scale, 0x7F, a_scale_bytes);   // 0x7F = ue8m0 exponent 0 = scale 1.0
cudaMemset(d_b_scale, 0x7F, b_scale_bytes);

cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(d_a_scale));
cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(d_b_scale));
```

This matches what `b12x/gemm/fp8_dense_cuda_ext.cu` does manually — its inline
PTX takes packed `0x7F7F7F7F` scale registers, which is the ue8m0 byte 0x7F
("1.0") replicated 4× per uint32.

`tune_gemm` does NOT yet implement this — it currently sets per-tensor scales
only, which works on Ada (sm_89) and Hopper (sm_90) but returns 0 candidates
on Blackwell-GeForce. Adding block-scaled mode is a follow-up; until then, on
sm_120 use the existing `b12x/gemm/fp8_dense_cuda_ext.cu` as the FP8 floor and
benchmark candidate kernels against it directly.

## NVFP4 (block-scaled FP4)

NVFP4 uses the same mode constant family but with `CUDA_R_4F_E2M1` data and
`VEC16_UE4M3` scale mode (16-element block, ue4m3 scale type). Same setup
pattern as block-scaled FP8.

## Scale-mode reference

| Mode | Shape | Block size | Scale dtype |
|---|---|---|---|
| `SCALAR_32F` (0) | scalar | whole tensor | fp32 |
| `VEC16_UE4M3` (1) | block | 16 elem | ue4m3 |
| `VEC32_UE8M0` (2) | block | 32 elem | ue8m0 |
| `OUTER_VEC_32F` (3) | vector | M or N elem | fp32 |
| `VEC128_32F` (4) | block | 128 elem | fp32 |
| `BLK128x128_32F` (5) | block | 128×128 | fp32 |
| `PER_BATCH_SCALAR_32F` (6) | per-batch scalar | whole batch | fp32 |
