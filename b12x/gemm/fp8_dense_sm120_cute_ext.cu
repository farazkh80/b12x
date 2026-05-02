#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <cute/arch/mma_sm120.hpp>

namespace b12x_fp8_dense_sm120_cute {
namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 32;
constexpr int kTileK = 32;

__device__ __forceinline__ uint32_t smem_ptr_to_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void st_shared_v4_u32(uint32_t smem_addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" : : "r"(smem_addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3));
}

__device__ __forceinline__ void ldmatrix_left(uint32_t smem_addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, _, %1, _}, [%2];" : "=r"(r0), "=r"(r1) : "r"(smem_addr));
}

__device__ __forceinline__ void ldmatrix_right(uint32_t smem_addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {_, %0, _, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(smem_addr));
}

__device__ __forceinline__ uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t selector) {
  uint32_t out;
  asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(selector));
  return out;
}

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b(uint32_t x) {
  uint32_t tmp = __shfl_xor_sync(0xffffffffu, x, 1);
  x = byte_perm(x, tmp, (threadIdx.x & 0x1) ? 0x3276u : 0x5410u);
  tmp = __shfl_xor_sync(0xffffffffu, x, 2);
  x = byte_perm(x, tmp, (threadIdx.x & 0x2) ? 0x3276u : 0x5410u);
  return x;
}

__device__ __forceinline__ int permuted_offset_128b(int row_idx, int vec_idx, int stride_128b) {
  return row_idx * stride_128b + (vec_idx ^ (row_idx & 7));
}

__device__ __forceinline__ uint32_t smem_addr_from_b128_offset(uint32_t base_addr, int offset_128b) {
  return base_addr + static_cast<uint32_t>(offset_128b * 16);
}

__device__ __forceinline__ void mma_f8f6f4_m16n8k32_f32_e4m3(
    float& d0,
    float& d1,
    float& d2,
    float& d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1) {
  cute::SM120_16x8x32_TN<cute::float_e4m3_t, cute::float_e4m3_t, float>::fma(
      d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
}

__device__ __forceinline__ void store_mma_16x16_tile(
    const float acc[8],
    __nv_bfloat16* __restrict__ c,
    int n,
    int row_base,
    int col_base,
    float scale) {
  int lane = threadIdx.x & 31;
  int lane_group = lane >> 2;
  int lane_pair_base = 2 * (lane & 3);
#pragma unroll
  for (int reg_id = 0; reg_id < 8; ++reg_id) {
    int row_slot = (reg_id & 3) >> 1;
    int row = row_base + lane_group + 8 * row_slot;
    int col = col_base + lane_pair_base + 8 * (reg_id >> 2) + (reg_id & 1);
    c[row * n + col] = __float2bfloat16(acc[reg_id] * scale);
  }
}

template <int TileM, int TileN, int StageK = kTileK>
__global__ __launch_bounds__(128, 2) void fp8_dense_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ a,
    const __nv_fp8_e4m3* __restrict__ b,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    __nv_bfloat16* __restrict__ c,
    int m,
    int n,
    int k) {
  int cta_row_base = blockIdx.x * TileM;
  int cta_col_base = blockIdx.y * TileN;
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  constexpr int col_warps = TileN / kTileN;
  int row_warp = warp_id / col_warps;
  int col_warp = warp_id - row_warp * col_warps;
  int row_base = cta_row_base + row_warp * kTileM;
  int col_base = cta_col_base + col_warp * kTileN;
  constexpr int compute_warps = (TileM / kTileM) * col_warps;
  bool compute_warp = warp_id < compute_warps;

  constexpr int smem_k_stride = StageK < 128 ? 128 : StageK;
  constexpr int smem_k_stride_128b = smem_k_stride / 16;
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* sA = smem;
  unsigned char* sB = smem + TileM * smem_k_stride;
  uint32_t a_base_addr = smem_ptr_to_u32(sA);
  uint32_t b_base_addr = smem_ptr_to_u32(sB);

  float out_acc0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float out_acc1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  const unsigned char* a_u8 = reinterpret_cast<const unsigned char*>(a);
  const unsigned char* b_u8 = reinterpret_cast<const unsigned char*>(b);

  for (int k_stage = 0; k_stage < k; k_stage += StageK) {
    for (int linear = threadIdx.x; linear < (TileM * StageK / 16); linear += blockDim.x) {
      int row = linear / (StageK / 16);
      int vec_idx = linear - row * (StageK / 16);
      int src_col = k_stage + vec_idx * 16;
      uint32_t dst_addr = a_base_addr + row * smem_k_stride + vec_idx * 16;
      const unsigned char* src = a_u8 + (cta_row_base + row) * k + src_col;
      const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
      st_shared_v4_u32(dst_addr, src32[0], src32[1], src32[2], src32[3]);
    }

    for (int linear = threadIdx.x; linear < (TileN * StageK / 16); linear += blockDim.x) {
      int row = linear / (StageK / 16);
      int vec_idx = linear - row * (StageK / 16);
      int src_col = k_stage + vec_idx * 16;
      uint32_t dst_addr = smem_addr_from_b128_offset(b_base_addr, permuted_offset_128b(row, vec_idx, smem_k_stride_128b));
      const unsigned char* src = b_u8 + (cta_col_base + row) * k + src_col;
      const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
      st_shared_v4_u32(dst_addr, src32[0], src32[1], src32[2], src32[3]);
    }
    __syncthreads();

    if (compute_warp) {
#pragma unroll
      for (int k_inner = 0; k_inner < StageK; k_inner += kTileK) {
        int a_row = row_warp * kTileM + (lane >> 2);
        int a_base_col = (lane & 3) * 2;
        uint32_t a_regs[4];
        a_regs[0] = static_cast<uint32_t>(sA[a_row * smem_k_stride + k_inner + a_base_col + 0]) |
                    (static_cast<uint32_t>(sA[a_row * smem_k_stride + k_inner + a_base_col + 1]) << 8) |
                    (static_cast<uint32_t>(sA[a_row * smem_k_stride + k_inner + a_base_col + 8]) << 16) |
                    (static_cast<uint32_t>(sA[a_row * smem_k_stride + k_inner + a_base_col + 9]) << 24);
        a_regs[1] = static_cast<uint32_t>(sA[(a_row + 8) * smem_k_stride + k_inner + a_base_col + 0]) |
                    (static_cast<uint32_t>(sA[(a_row + 8) * smem_k_stride + k_inner + a_base_col + 1]) << 8) |
                    (static_cast<uint32_t>(sA[(a_row + 8) * smem_k_stride + k_inner + a_base_col + 8]) << 16) |
                    (static_cast<uint32_t>(sA[(a_row + 8) * smem_k_stride + k_inner + a_base_col + 9]) << 24);
        a_regs[2] = static_cast<uint32_t>(sA[a_row * smem_k_stride + k_inner + a_base_col + 16]) |
                    (static_cast<uint32_t>(sA[a_row * smem_k_stride + k_inner + a_base_col + 17]) << 8) |
                    (static_cast<uint32_t>(sA[a_row * smem_k_stride + k_inner + a_base_col + 24]) << 16) |
                    (static_cast<uint32_t>(sA[a_row * smem_k_stride + k_inner + a_base_col + 25]) << 24);
        a_regs[3] = static_cast<uint32_t>(sA[(a_row + 8) * smem_k_stride + k_inner + a_base_col + 16]) |
                    (static_cast<uint32_t>(sA[(a_row + 8) * smem_k_stride + k_inner + a_base_col + 17]) << 8) |
                    (static_cast<uint32_t>(sA[(a_row + 8) * smem_k_stride + k_inner + a_base_col + 24]) << 16) |
                    (static_cast<uint32_t>(sA[(a_row + 8) * smem_k_stride + k_inner + a_base_col + 25]) << 24);

        int b_offset = permuted_offset_128b(lane & 7, k_inner / 16 + ((lane & 15) >> 3), smem_k_stride_128b) + smem_k_stride_128b * (lane >> 4) * 8 + col_warp * kTileN * smem_k_stride_128b;
        uint32_t b0_k0, b1_k0, b0_k1, b1_k1;
        ldmatrix_left(smem_addr_from_b128_offset(b_base_addr, b_offset), b0_k0, b1_k0);
        ldmatrix_right(smem_addr_from_b128_offset(b_base_addr, b_offset), b0_k1, b1_k1);
        b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0);
        b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0);
        b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1);
        b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1);

        int b_offset_1 = b_offset + 16 * smem_k_stride_128b;
        uint32_t b2_k0, b3_k0, b2_k1, b3_k1;
        ldmatrix_left(smem_addr_from_b128_offset(b_base_addr, b_offset_1), b2_k0, b3_k0);
        ldmatrix_right(smem_addr_from_b128_offset(b_base_addr, b_offset_1), b2_k1, b3_k1);
        b2_k0 = frag_layout_swizzle_16b_to_8b(b2_k0);
        b3_k0 = frag_layout_swizzle_16b_to_8b(b3_k0);
        b2_k1 = frag_layout_swizzle_16b_to_8b(b2_k1);
        b3_k1 = frag_layout_swizzle_16b_to_8b(b3_k1);

        mma_f8f6f4_m16n8k32_f32_e4m3(out_acc0[0], out_acc0[1], out_acc0[2], out_acc0[3], a_regs[0], a_regs[1], a_regs[2], a_regs[3], b0_k0, b0_k1);
        mma_f8f6f4_m16n8k32_f32_e4m3(out_acc0[4], out_acc0[5], out_acc0[6], out_acc0[7], a_regs[0], a_regs[1], a_regs[2], a_regs[3], b1_k0, b1_k1);
        mma_f8f6f4_m16n8k32_f32_e4m3(out_acc1[0], out_acc1[1], out_acc1[2], out_acc1[3], a_regs[0], a_regs[1], a_regs[2], a_regs[3], b2_k0, b2_k1);
        mma_f8f6f4_m16n8k32_f32_e4m3(out_acc1[4], out_acc1[5], out_acc1[6], out_acc1[7], a_regs[0], a_regs[1], a_regs[2], a_regs[3], b3_k0, b3_k1);
      }
    }
    __syncthreads();
  }

  if (compute_warp) {
    float scale = scale_a[0] * scale_b[0];
    store_mma_16x16_tile(out_acc0, c, n, row_base, col_base, scale);
    store_mma_16x16_tile(out_acc1, c, n, row_base, col_base + 16, scale);
  }
}

void validate_inputs(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scale_a, const torch::Tensor& scale_b, const torch::Tensor& out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && scale_a.is_cuda() && scale_b.is_cuda() && out.is_cuda(), "all tensors must be CUDA tensors");
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn && b.dtype() == torch::kFloat8_e4m3fn, "a and b must be float8_e4m3fn");
  TORCH_CHECK(scale_a.dtype() == torch::kFloat32 && scale_b.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2, "a, b, and out must be rank-2");
  TORCH_CHECK(a.size(1) == b.size(1), "a and b must have the same K dimension");
  TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == b.size(0), "out shape mismatch");
  TORCH_CHECK(scale_a.numel() == 1 && scale_b.numel() == 1, "only scalar scales are supported");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(), "a, b, and out must be contiguous");
  TORCH_CHECK(a.size(0) % kTileM == 0 && b.size(0) % kTileN == 0 && a.size(1) % kTileK == 0, "M must be divisible by 16, N by 32, and K by 32");
}

}  // namespace

void fp8_dense_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor scale_a, torch::Tensor scale_b, torch::Tensor out) {
  validate_inputs(a, b, scale_a, scale_b, out);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  int m = static_cast<int>(a.size(0));
  int k = static_cast<int>(a.size(1));
  int n = static_cast<int>(b.size(0));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.device().index());
  constexpr int tile_m = 16;
  constexpr int tile_n = 32;
  dim3 grid(m / tile_m, n / tile_n, 1);
  dim3 block(32, 1, 1);
  size_t smem_bytes = tile_m * 128 + tile_n * 128;
  fp8_dense_gemm_kernel<tile_m, tile_n><<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3*>(b.data_ptr()),
      scale_a.data_ptr<float>(),
      scale_b.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      m,
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace b12x_fp8_dense_sm120_cute

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_dense_gemm", &b12x_fp8_dense_sm120_cute::fp8_dense_gemm, "FP8 dense GEMM using SM120 CUTLASS/CuTe atom");
}
