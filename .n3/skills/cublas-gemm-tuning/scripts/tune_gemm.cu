// tune_gemm.cu — minimal cuBLASLt GEMM tuner (CLI binary).
//
// Trusts cuBLASLt's built-in heuristic: ask for up to N candidates ranked by
// predicted perf, time each one, emit JSON sorted by measured median_us.
// No custom algo enumeration; the cuBLASLt heuristic already covers the
// full algo×tile×stages×splitK×swizzle space and returns the top picks.
//
// Usage:
//   tune_gemm --M 32 --K 5376 --N 5376 \
//             --a-dtype e4m3 --b-dtype e4m3 --c-dtype bf16 --compute fp32 \
//             --trans-a N --trans-b T \
//             --request-count 32 --warmup 5 --iters 20 --repeats 3
//
// Output: JSON on stdout. See SKILL.md for schema.

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s @ %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
  std::exit(1); } } while(0)
#define LT_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLASLt %s @ %s:%d: status=%d\n", #x, __FILE__, __LINE__, (int)s_); \
  std::exit(1); } } while(0)

// `bits` carries element width (used for sub-byte types like FP4_E2M1=4 bits).
// `mode` selects the random-init kernel branch; -1 means "skip init_random,
// allocations are zeroed via cudaMemset" (used for FP4 — for tuning timing
// the input values don't have to be sensible).
struct DtypeSpec { const char* name; cudaDataType_t cuda; int bits; int mode; };
static DtypeSpec resolve_dtype(const std::string& s) {
  if (s == "fp32" || s == "float")    return {"fp32", CUDA_R_32F,    32, 0};
  if (s == "bf16")                    return {"bf16", CUDA_R_16BF,   16, 1};
  if (s == "fp16" || s == "half")     return {"fp16", CUDA_R_16F,    16, 2};
  if (s == "e4m3" || s == "fp8_e4m3") return {"e4m3", CUDA_R_8F_E4M3, 8, 3};
  if (s == "e5m2" || s == "fp8_e5m2") return {"e5m2", CUDA_R_8F_E5M2, 8, 4};
  if (s == "int8")                    return {"int8", CUDA_R_8I,      8, 5};
  if (s == "fp4_e2m1" || s == "e2m1") return {"fp4_e2m1", CUDA_R_4F_E2M1, 4, -1};
  fprintf(stderr, "unknown dtype: %s\n", s.c_str()); std::exit(2);
}

// Total bytes needed for `elements` of the given type, rounded up. Handles
// sub-byte types correctly (FP4: 2 elements per byte).
static size_t dt_bytes(size_t elements, const DtypeSpec& d) {
  return (elements * (size_t)d.bits + 7) / 8;
}
static cublasComputeType_t resolve_compute(const std::string& s) {
  if (s == "fp32") return CUBLAS_COMPUTE_32F;
  if (s == "fp32_tf32") return CUBLAS_COMPUTE_32F_FAST_TF32;
  if (s == "fp16") return CUBLAS_COMPUTE_16F;
  if (s == "int32") return CUBLAS_COMPUTE_32I;
  fprintf(stderr, "unknown compute: %s\n", s.c_str()); std::exit(2);
}
static cublasOperation_t parse_op(const std::string& s) {
  if (s == "N" || s == "n") return CUBLAS_OP_N;
  if (s == "T" || s == "t") return CUBLAS_OP_T;
  fprintf(stderr, "bad transpose '%s'\n", s.c_str()); std::exit(2);
}

__global__ void init_rand_kernel(uint8_t* dst, size_t n, uint64_t seed, int mode) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t s = seed * 6364136223846793005ULL + (uint64_t)i * 1442695040888963407ULL + 1ULL;
  s ^= s >> 33; s *= 0xff51afd7ed558ccdULL; s ^= s >> 33;
  float f = ((int32_t)(uint32_t)(s >> 32)) / (float)(1u << 31) * 0.125f;
  if      (mode == 0) reinterpret_cast<float*>(dst)[i] = f;
  else if (mode == 1) reinterpret_cast<__nv_bfloat16*>(dst)[i] = __float2bfloat16(f);
  else if (mode == 2) reinterpret_cast<__half*>(dst)[i] = __float2half(f);
  else if (mode == 3) reinterpret_cast<__nv_fp8_e4m3*>(dst)[i] = __nv_fp8_e4m3(f);
  else if (mode == 4) reinterpret_cast<__nv_fp8_e5m2*>(dst)[i] = __nv_fp8_e5m2(f);
  else if (mode == 5) reinterpret_cast<int8_t*>(dst)[i] = (int8_t)max(-127, min(127, (int)(f * 100.0f)));
}
static void init_random(void* d, size_t n, DtypeSpec dt, uint64_t seed) {
  if (dt.mode < 0) {
    // FP4 path — no random init kernel; just zero out the buffer. cuBLAS-Lt
    // doesn't care about input values for tuning purposes, only sizes/strides.
    CUDA_CHECK(cudaMemset(d, 0, dt_bytes(n, dt)));
    return;
  }
  int block = 256, grid = (int)((n + block - 1) / block);
  init_rand_kernel<<<grid, block>>>((uint8_t*)d, n, seed, dt.mode);
  CUDA_CHECK(cudaDeviceSynchronize());
}

struct Args {
  int M=0, K=0, N=0;
  std::string a="bf16", b="bf16", c="bf16", compute="fp32";
  std::string trans_a="N", trans_b="N";
  int warmup=5, iters=20, repeats=3;
  int request_count=32;
  size_t max_workspace=32ull << 20;
  uint64_t seed=12345;
  // Scale mode for FP8/INT8/NVFP4 inputs:
  //   "scalar"             — per-tensor SCALAR_32F (default; works on sm_89/sm_90)
  //   "block_ue8m0_vec32"  — UE8M0, 32-element K-dim blocks (sm_120 FP8 path)
  //   "block_ue4m3_vec16"  — UE4M3, 16-element K-dim blocks (NVFP4 path; not yet wired)
  std::string scale_mode="auto";
};
static void parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto V = [&](const char* tag) { if (i + 1 >= argc) { fprintf(stderr, "missing val for %s\n", tag); std::exit(2); } return std::string(argv[++i]); };
    if      (k == "--M")  a.M = std::atoi(V("--M").c_str());
    else if (k == "--K")  a.K = std::atoi(V("--K").c_str());
    else if (k == "--N")  a.N = std::atoi(V("--N").c_str());
    else if (k == "--a-dtype") a.a = V("--a-dtype");
    else if (k == "--b-dtype") a.b = V("--b-dtype");
    else if (k == "--c-dtype") a.c = V("--c-dtype");
    else if (k == "--compute") a.compute = V("--compute");
    else if (k == "--trans-a") a.trans_a = V("--trans-a");
    else if (k == "--trans-b") a.trans_b = V("--trans-b");
    else if (k == "--warmup")  a.warmup = std::atoi(V("--warmup").c_str());
    else if (k == "--iters")   a.iters  = std::atoi(V("--iters").c_str());
    else if (k == "--repeats") a.repeats = std::atoi(V("--repeats").c_str());
    else if (k == "--request-count") a.request_count = std::atoi(V("--request-count").c_str());
    else if (k == "--max-workspace-bytes") a.max_workspace = (size_t)std::atoll(V("--max-workspace-bytes").c_str());
    else if (k == "--seed") a.seed = (uint64_t)std::atoll(V("--seed").c_str());
    else if (k == "--scale-mode") a.scale_mode = V("--scale-mode");
    else if (k == "--help" || k == "-h") {
      printf("Usage: tune_gemm --M N --K N --N N [--a-dtype ...] [--b-dtype ...] [--c-dtype ...]\n"
             "                 [--compute fp32|fp16|fp32_tf32] [--trans-a N|T] [--trans-b N|T]\n"
             "                 [--warmup 5] [--iters 20] [--repeats 3] [--request-count 32]\n"
             "                 [--max-workspace-bytes %zu]\n"
             "Output: JSON on stdout.\n", a.max_workspace);
      std::exit(0);
    } else { fprintf(stderr, "unknown arg %s\n", k.c_str()); std::exit(2); }
  }
  if (!a.M || !a.K || !a.N) { fprintf(stderr, "--M, --K, --N required\n"); std::exit(2); }
}

int main(int argc, char** argv) {
  Args args; parse_args(argc, argv, args);
  DtypeSpec a_dt = resolve_dtype(args.a), b_dt = resolve_dtype(args.b), c_dt = resolve_dtype(args.c);
  cublasComputeType_t compute = resolve_compute(args.compute);
  cudaDataType_t scale_type = (compute == CUBLAS_COMPUTE_16F) ? CUDA_R_16F : CUDA_R_32F;
  cublasOperation_t opA = parse_op(args.trans_a), opB = parse_op(args.trans_b);

  cublasLtHandle_t h; LT_CHECK(cublasLtCreate(&h));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));

  // Device props (queried early so the scale-mode picker below can read .major).
  cudaDeviceProp dprops; int devid; cudaGetDevice(&devid); cudaGetDeviceProperties(&dprops, devid);

  // Layout dims (cuBLASLt is column-major).
  size_t a_rows = (opA == CUBLAS_OP_N) ? args.M : args.K;
  size_t a_cols = (opA == CUBLAS_OP_N) ? args.K : args.M;
  size_t b_rows = (opB == CUBLAS_OP_N) ? args.K : args.N;
  size_t b_cols = (opB == CUBLAS_OP_N) ? args.N : args.K;

  void *da, *db, *dd; void* dc = nullptr;
  CUDA_CHECK(cudaMalloc(&da, dt_bytes(a_rows * a_cols, a_dt)));
  CUDA_CHECK(cudaMalloc(&db, dt_bytes(b_rows * b_cols, b_dt)));
  CUDA_CHECK(cudaMalloc(&dd, dt_bytes((size_t)args.M * args.N, c_dt)));
  init_random(da, a_rows * a_cols, a_dt, args.seed);
  init_random(db, b_rows * b_cols, b_dt, args.seed + 1);
  CUDA_CHECK(cudaMemset(dd, 0, dt_bytes((size_t)args.M * args.N, c_dt)));

  // L2 flush buffer.
  size_t l2_bytes = std::max((size_t)dprops.l2CacheSize * 2, (size_t)(32ull << 20));
  void* d_l2 = nullptr; CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));

  // Workspace.
  void* d_workspace; CUDA_CHECK(cudaMalloc(&d_workspace, args.max_workspace));

  // Matmul descriptor + layouts.
  cublasLtMatmulDesc_t op_desc;
  LT_CHECK(cublasLtMatmulDescCreate(&op_desc, compute, scale_type));
  LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

  // FP8/INT8/NVFP4 require A/B scale pointers, else heuristic returns
  // NOT_SUPPORTED.
  bool is_fp4   = (a_dt.cuda == CUDA_R_4F_E2M1 || b_dt.cuda == CUDA_R_4F_E2M1);
  bool is_fp8_8 = (a_dt.cuda == CUDA_R_8F_E4M3 || a_dt.cuda == CUDA_R_8F_E5M2 || a_dt.cuda == CUDA_R_8I);
  bool needs_scales = is_fp4 || is_fp8_8;
  void *d_sa = nullptr, *d_sb = nullptr;
  std::string effective_scale_mode = args.scale_mode;
  if (needs_scales && effective_scale_mode == "auto") {
    // Default selection:
    //   FP4 (NVFP4 e2m1)  → block_ue4m3_vec16 (mandatory; no per-tensor path)
    //   FP8/INT8 sm_89/90 → SCALAR_32F per-tensor (legacy Ada/Hopper)
    //   FP8/INT8 sm_120+  → block_ue8m0_vec32 (Blackwell kind::mxf8f6f4 path)
    if      (is_fp4)             effective_scale_mode = "block_ue4m3_vec16";
    else if (dprops.major >= 12) effective_scale_mode = "block_ue8m0_vec32";
    else                         effective_scale_mode = "scalar";
  }
  if (needs_scales && effective_scale_mode == "scalar") {
    float one = 1.0f;
    CUDA_CHECK(cudaMalloc(&d_sa, sizeof(float))); CUDA_CHECK(cudaMemcpy(d_sa, &one, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_sb, sizeof(float))); CUDA_CHECK(cudaMemcpy(d_sb, &one, 4, cudaMemcpyHostToDevice));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sa, sizeof(d_sa)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_sb, sizeof(d_sb)));
    int8_t fast_accum = 1;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum));
  } else if (needs_scales && effective_scale_mode == "block_ue8m0_vec32") {
    // sm_120 FP8 (kind::mxf8f6f4) needs UE8M0 scales — one byte per
    // (output-element, K-block-of-32). Allocate scale tensors sized to the
    // contracting-dim partition and initialize to 0x7F (= 1.0 in UE8M0:
    // exponent encodes 2^(byte - 127), so 0x7F → 2^0 = 1).
    size_t k_blocks = (size_t)((args.K + 31) / 32);
    size_t a_scale_bytes = k_blocks * (size_t)args.M;  // matches A's M-dim
    size_t b_scale_bytes = k_blocks * (size_t)args.N;  // matches B's N-dim
    CUDA_CHECK(cudaMalloc(&d_sa, a_scale_bytes));
    CUDA_CHECK(cudaMalloc(&d_sb, b_scale_bytes));
    CUDA_CHECK(cudaMemset(d_sa, 0x7F, a_scale_bytes));
    CUDA_CHECK(cudaMemset(d_sb, 0x7F, b_scale_bytes));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sa, sizeof(d_sa)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_sb, sizeof(d_sb)));
    int32_t mode_a = (int32_t)CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    int32_t mode_b = mode_a;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode_a, sizeof(mode_a)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode_b, sizeof(mode_b)));
    int8_t fast_accum = 1;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum));
    fprintf(stderr, "[tune_gemm] scale-mode=block_ue8m0_vec32 (a=%zuB, b=%zuB)\n",
            a_scale_bytes, b_scale_bytes);
  } else if (needs_scales && effective_scale_mode == "block_ue4m3_vec16") {
    // NVFP4 (CUDA_R_4F_E2M1) with VEC16_UE4M3 scales.
    //
    // KNOWN LIMITATION (2026-05-04 on cuBLAS 13.2.1 / sm_120 GB10):
    // cublasLtMatmulAlgoGetHeuristic still returns CUBLAS_STATUS_INVALID_VALUE
    // (status=7) for NVFP4 even with all of:
    //   - aScalePointer/bScalePointer set, A/B/C/D/D_OUT_SCALE_MODE set
    //   - cScalePointer/dScalePointer/dOutScalePointer explicitly nullptr
    //   - POINTER_MODE_DEVICE, FAST_ACCUM=1, separate C/D layouts
    //   - PreferenceInit, default col-major layouts
    // The likely missing piece is the **swizzled scale tensor layout** that
    // cuBLAS-Lt expects (the scale tensors are not flat (M, K/16) of UE4M3
    // bytes; they're in a swizzled MX-spec layout that this flat memset
    // doesn't produce). See TRT-LLM cpp/tensorrt_llm/thop/cublasFp4ScaledMM.cpp
    // and the trtllm-gen scale-tensor swizzler for the actual layout. Until
    // that's reverse-engineered, NVFP4 tuning via this binary returns 0
    // candidates; fall back to AutoTuner-driven tactic search through TRT-LLM's
    // CublasLtFP4GemmRunner instead.
    //
    // FP8 (block_ue8m0_vec32) does work correctly — see comment above.
    if (args.K % 16 != 0) {
      fprintf(stderr, "[tune_gemm] WARN: K=%d is not a multiple of 16 — NVFP4 vec16 "
                      "scale layout assumes K%%16==0\n", args.K);
    }
    size_t k_blocks = (size_t)((args.K + 15) / 16);
    size_t a_scale_bytes = k_blocks * (size_t)args.M;
    size_t b_scale_bytes = k_blocks * (size_t)args.N;
    CUDA_CHECK(cudaMalloc(&d_sa, a_scale_bytes));
    CUDA_CHECK(cudaMalloc(&d_sb, b_scale_bytes));
    CUDA_CHECK(cudaMemset(d_sa, 0x38, a_scale_bytes));
    CUDA_CHECK(cudaMemset(d_sb, 0x38, b_scale_bytes));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sa, sizeof(d_sa)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_sb, sizeof(d_sb)));
    int32_t mode_a = (int32_t)CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    int32_t mode_b = mode_a;
    int32_t mode_scalar = (int32_t)CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode_a, sizeof(mode_a)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode_b, sizeof(mode_b)));
    // FP4 heuristic also requires C/D/D_OUT scale modes set explicitly to
    // SCALAR_32F with null pointers (per TRT-LLM cublasMMWrapper FP4 path).
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &mode_scalar, sizeof(mode_scalar)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &mode_scalar, sizeof(mode_scalar)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &mode_scalar, sizeof(mode_scalar)));
    void* null_ptr = nullptr;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &null_ptr, sizeof(null_ptr)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &null_ptr, sizeof(null_ptr)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &null_ptr, sizeof(null_ptr)));
    cublasLtPointerMode_t pmode = CUBLASLT_POINTER_MODE_DEVICE;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pmode, sizeof(pmode)));
    int8_t fast_accum = 1;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum));
    fprintf(stderr, "[tune_gemm] scale-mode=block_ue4m3_vec16 (a=%zuB, b=%zuB)\n",
            a_scale_bytes, b_scale_bytes);
  } else if (needs_scales) {
    fprintf(stderr, "unsupported --scale-mode '%s' for FP8/FP4/INT8\n", effective_scale_mode.c_str());
    std::exit(2);
  }
  // cuBLASLt is column-major natively. ld = rows for the in-memory shape of
  // each descriptor, which makes the "fast" dim contiguous (the K dim for
  // canonical TN FP8: A in memory (K,M), B in memory (K,N), D in memory (M,N)).
  // Setting CUBLASLT_ORDER_ROW here used to work for BF16 but produces an
  // unsupported config for sm_120 block-scaled FP8 (heuristic returns
  // CUBLAS_STATUS_NOT_SUPPORTED). Default col-major matches what
  // torch._scaled_mm emits in production.
  cublasLtMatrixLayout_t la, lb, lc, ld;
  LT_CHECK(cublasLtMatrixLayoutCreate(&la, a_dt.cuda, a_rows, a_cols, a_rows));
  LT_CHECK(cublasLtMatrixLayoutCreate(&lb, b_dt.cuda, b_rows, b_cols, b_rows));
  // C and D are the same matrix in our tune (out-of-place=0 doesn't apply
  // because dd is reused), but FP4 cuBLAS-Lt heuristic on sm_120 requires
  // *distinct* layout descriptor objects for C and D — TRT-LLM's
  // cublasMMWrapper::BlockScaleGemm creates them separately for the same
  // reason. Other dtypes work either way; we always create both.
  LT_CHECK(cublasLtMatrixLayoutCreate(&lc, c_dt.cuda, args.M, args.N, args.M));
  LT_CHECK(cublasLtMatrixLayoutCreate(&ld, c_dt.cuda, args.M, args.N, args.M));

  // Ask cuBLASLt for the top N candidates.
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  // PreferenceInit zero-initializes all fields; matches TRT-LLM's getTactics
  // setup. Some FP4 heuristic paths reject the request when fields aren't
  // explicitly default-initialized.
  LT_CHECK(cublasLtMatmulPreferenceInit(pref));
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &args.max_workspace, sizeof(args.max_workspace)));
  std::vector<cublasLtMatmulHeuristicResult_t> hres(args.request_count);
  int returned = 0;
  cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(h, op_desc, la, lb, lc, ld, pref,
                                                    args.request_count, hres.data(), &returned);
  if (hs != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasLtMatmulAlgoGetHeuristic status=%d\n", (int)hs);
  }
  fprintf(stderr, "[tune_gemm] %d heuristic candidates\n", returned);

  // Time each.
  struct Row {
    int rank_heuristic;
    int algo_id, tile_id, stages_id, splitk_num, swizzling, reduction_scheme;
    size_t workspace_bytes;
    float wave_count;
    double median_us, mean_us, min_us, spread_pct;
    bool launch_ok;
    int launch_status;
  };
  std::vector<Row> rows;
  rows.reserve(returned);

  // alpha/beta: passed by pointer to cublasLtMatmul. Default cuBLASLt mode is
  // POINTER_MODE_HOST so &h_alpha/&h_beta works directly. The FP4 path above
  // sets POINTER_MODE_DEVICE — for that case we mirror the values into device
  // memory and pass those pointers instead.
  float h_alpha = 1.0f, h_beta = 0.0f;
  float *d_alpha = nullptr, *d_beta = nullptr;
  bool device_pointers = (is_fp4 && effective_scale_mode == "block_ue4m3_vec16");
  if (device_pointers) {
    CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta,  sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_alpha, &h_alpha, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  &h_beta,  sizeof(float), cudaMemcpyHostToDevice));
  }
  const float* alpha_p = device_pointers ? (const float*)d_alpha : &h_alpha;
  const float* beta_p  = device_pointers ? (const float*)d_beta  : &h_beta;
  for (int i = 0; i < returned; ++i) {
    Row r{};
    r.rank_heuristic = i;
    r.workspace_bytes = hres[i].workspaceSize;
    r.wave_count = hres[i].wavesCount;
    int v = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&hres[i].algo, CUBLASLT_ALGO_CONFIG_ID, &v, sizeof(v), nullptr); r.algo_id = v;
    cublasLtMatmulAlgoConfigGetAttribute(&hres[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &v, sizeof(v), nullptr); r.tile_id = v;
    cublasLtMatmulAlgoConfigGetAttribute(&hres[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &v, sizeof(v), nullptr); r.stages_id = v;
    cublasLtMatmulAlgoConfigGetAttribute(&hres[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &v, sizeof(v), nullptr); r.splitk_num = v;
    cublasLtMatmulAlgoConfigGetAttribute(&hres[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &v, sizeof(v), nullptr); r.swizzling = v;
    cublasLtMatmulAlgoConfigGetAttribute(&hres[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &v, sizeof(v), nullptr); r.reduction_scheme = v;

    // Launch test (verifies the algo accepts our shape/dtype before benching).
    cublasStatus_t st = cublasLtMatmul(h, op_desc, alpha_p, da, la, db, lb, beta_p, dd, lc, dd, ld,
                                       &hres[i].algo, d_workspace, args.max_workspace, s);
    if (st != CUBLAS_STATUS_SUCCESS) {
      r.launch_ok = false; r.launch_status = (int)st;
      rows.push_back(r); continue;
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    r.launch_ok = true;

    // Time.
    std::vector<double> repeat_medians;
    for (int rep = 0; rep < args.repeats; ++rep) {
      for (int w = 0; w < args.warmup; ++w) {
        cudaMemsetAsync(d_l2, 0, l2_bytes, s);
        cublasLtMatmul(h, op_desc, alpha_p, da, la, db, lb, beta_p, dd, lc, dd, ld,
                       &hres[i].algo, d_workspace, args.max_workspace, s);
      }
      CUDA_CHECK(cudaStreamSynchronize(s));
      std::vector<cudaEvent_t> es(args.iters), ee(args.iters);
      for (int j = 0; j < args.iters; ++j) { cudaEventCreate(&es[j]); cudaEventCreate(&ee[j]); }
      for (int j = 0; j < args.iters; ++j) {
        cudaMemsetAsync(d_l2, 0, l2_bytes, s);
        cudaEventRecord(es[j], s);
        cublasLtMatmul(h, op_desc, alpha_p, da, la, db, lb, beta_p, dd, lc, dd, ld,
                       &hres[i].algo, d_workspace, args.max_workspace, s);
        cudaEventRecord(ee[j], s);
      }
      CUDA_CHECK(cudaStreamSynchronize(s));
      std::vector<double> us; us.reserve(args.iters);
      for (int j = 0; j < args.iters; ++j) {
        float ms; cudaEventElapsedTime(&ms, es[j], ee[j]);
        us.push_back((double)ms * 1000.0);
        cudaEventDestroy(es[j]); cudaEventDestroy(ee[j]);
      }
      std::sort(us.begin(), us.end());
      repeat_medians.push_back(us[args.iters / 2]);
    }
    std::sort(repeat_medians.begin(), repeat_medians.end());
    r.median_us = repeat_medians[repeat_medians.size() / 2];
    r.min_us = repeat_medians.front();
    r.mean_us = 0; for (double v : repeat_medians) r.mean_us += v; r.mean_us /= repeat_medians.size();
    r.spread_pct = (r.median_us > 0) ? 100.0 * (repeat_medians.back() - repeat_medians.front()) / r.median_us : 0.0;
    rows.push_back(r);
  }

  // Capture the heuristic-rank-0 timing as the "baseline" — what cuBLAS-Lt
  // would auto-pick if you didn't sweep. Every other candidate's speedup is
  // reported relative to this. Done BEFORE sorting so we use rank_heuristic
  // (the original index) as the tag.
  double baseline_us = -1.0;
  int    baseline_idx = -1;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (rows[i].rank_heuristic == 0 && rows[i].launch_ok) {
      baseline_us = rows[i].median_us;
      baseline_idx = (int)i;
      break;
    }
  }

  // Sort by measured median ascending; failed launches sink to the bottom.
  std::sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) {
    if (x.launch_ok != y.launch_ok) return x.launch_ok;
    return x.median_us < y.median_us;
  });

  // Emit JSON.
  printf("{\n");
  printf("  \"device\": \"%s\",\n", dprops.name);
  printf("  \"compute_capability\": \"%d.%d\",\n", dprops.major, dprops.minor);
  printf("  \"shape\": {\"M\": %d, \"K\": %d, \"N\": %d},\n", args.M, args.K, args.N);
  printf("  \"dtype\": {\"a\": \"%s\", \"b\": \"%s\", \"c\": \"%s\", \"compute\": \"%s\"},\n",
         a_dt.name, b_dt.name, c_dt.name, args.compute.c_str());
  printf("  \"transpose\": {\"a\": \"%s\", \"b\": \"%s\"},\n", args.trans_a.c_str(), args.trans_b.c_str());
  printf("  \"measurement\": {\"warmup\": %d, \"iters\": %d, \"repeats\": %d},\n",
         args.warmup, args.iters, args.repeats);
  printf("  \"heuristic_returned\": %d,\n", returned);
  printf("  \"candidates\": [\n");
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    printf("    {\"rank_measured\": %zu, \"rank_heuristic\": %d, \"algo_id\": %d, \"tile_id\": %d, "
           "\"stages_id\": %d, \"splitk_num\": %d, \"swizzling\": %d, \"reduction_scheme\": %d, "
           "\"workspace_bytes\": %zu, \"wave_count\": %.4f, ",
           i, r.rank_heuristic, r.algo_id, r.tile_id, r.stages_id, r.splitk_num,
           r.swizzling, r.reduction_scheme, r.workspace_bytes, r.wave_count);
    if (r.launch_ok) {
      double spd = (baseline_us > 0) ? (baseline_us / r.median_us) : 0.0;
      printf("\"median_us\": %.4f, \"mean_us\": %.4f, \"min_us\": %.4f, \"spread_pct\": %.4f, "
             "\"speedup_vs_heuristic_baseline\": %.4f, \"is_heuristic_baseline\": %s, \"launch_ok\": true",
             r.median_us, r.mean_us, r.min_us, r.spread_pct,
             spd, (r.rank_heuristic == 0 ? "true" : "false"));
    } else {
      printf("\"launch_ok\": false, \"launch_status\": %d, \"is_heuristic_baseline\": %s",
             r.launch_status, (r.rank_heuristic == 0 ? "true" : "false"));
    }
    printf("}%s\n", i + 1 < rows.size() ? "," : "");
  }
  printf("  ],\n");

  // Heuristic baseline (rank-0): what cuBLAS-Lt would auto-pick without a sweep.
  if (baseline_idx >= 0) {
    // After sort, baseline_idx is invalid; find the rank_heuristic==0 row again.
    int found = -1;
    for (size_t i = 0; i < rows.size(); ++i) {
      if (rows[i].rank_heuristic == 0) { found = (int)i; break; }
    }
    if (found >= 0 && rows[found].launch_ok) {
      const auto& r = rows[found];
      printf("  \"heuristic_baseline\": {\"rank_heuristic\": 0, \"rank_measured\": %d, "
             "\"algo_id\": %d, \"tile_id\": %d, \"stages_id\": %d, \"splitk_num\": %d, "
             "\"swizzling\": %d, \"median_us\": %.4f},\n",
             found, r.algo_id, r.tile_id, r.stages_id, r.splitk_num, r.swizzling, r.median_us);
    } else {
      printf("  \"heuristic_baseline\": null,\n");
    }
  } else {
    // rank-0 candidate failed to launch; can't compare against it.
    printf("  \"heuristic_baseline\": null,\n");
  }

  if (!rows.empty() && rows.front().launch_ok) {
    const auto& r = rows.front();
    double spd = (baseline_us > 0) ? (baseline_us / r.median_us) : 0.0;
    printf("  \"best\": {\"rank_heuristic\": %d, \"rank_measured\": 0, \"algo_id\": %d, "
           "\"tile_id\": %d, \"stages_id\": %d, \"splitk_num\": %d, \"swizzling\": %d, "
           "\"median_us\": %.4f, \"speedup_vs_heuristic_baseline\": %.4f}\n",
           r.rank_heuristic, r.algo_id, r.tile_id, r.stages_id, r.splitk_num, r.swizzling,
           r.median_us, spd);
  } else {
    printf("  \"best\": null\n");
  }
  printf("}\n");

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc); cublasLtMatrixLayoutDestroy(ld);
  cublasLtMatmulDescDestroy(op_desc);
  cublasLtDestroy(h);
  cudaStreamDestroy(s);
  cudaFree(d_workspace); cudaFree(d_l2);
  cudaFree(da); cudaFree(db); cudaFree(dd);
  if (d_sa) cudaFree(d_sa); if (d_sb) cudaFree(d_sb);
  if (d_alpha) cudaFree(d_alpha); if (d_beta) cudaFree(d_beta);
  return 0;
}
