"""Debug cp.async in a 2-iteration K-loop. Load 2 K-blocks of W13 sequentially,
write each iteration's SMEM contents back to global, compare to source.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, Int64, Uint8, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import (
    get_ptr_as_int64,
    ld_shared_i32_relaxed,
    shared_ptr_to_u32,
)
from b12x.moe.fused.mxfp4_mxfp8._fused_kernel import _cp_async_16B


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


@cute.jit
def driver(mIn, mOut, stream):
    kernel(mIn, mOut).launch(grid=(1, 1, 1), block=[32, 1, 1], stream=stream)


@cute.kernel
def kernel(mIn, mOut):
    """mIn shape: (FC1_N=64, K_half=32) byte. K_in_blocks=2, fc1_loads_per_thread=2."""
    tidx = cute.arch.thread_idx()[0]
    smem = utils.SmemAllocator()

    @cute.struct
    class Storage:
        sW13_kb: cute.struct.MemRange[Uint8, 64 * 16]   # 1024 bytes

    storage = smem.allocate(Storage)
    sW13_addr = shared_ptr_to_u32(storage.sW13_kb.data_ptr())

    K_half = 32
    fc1_loads_per_thread = 2
    K_in_blocks = 2
    FC1_N = 64

    # K-loop: load K-block, then write SMEM out to global at offset (k * 1024).
    for k in cutlass.range_constexpr(K_in_blocks):
        for i in cutlass.range_constexpr(fc1_loads_per_thread):
            row = Int32(i * 32) + Int32(tidx)
            src_byte_off = row * Int32(K_half) + Int32(k * 16)
            src_addr = get_ptr_as_int64(mIn, src_byte_off)
            dst_addr = sW13_addr + row * Int32(16)
            _cp_async_16B(dst_addr, src_addr)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()

        # Each thread reads its 1 u32 (4 bytes) from SMEM and writes to mOut.
        # Distribution: 32 threads × 4 b32 each = 128 b32 = 512 bytes = half SMEM.
        # Iter 0 writes mOut[k=0, ...]; iter 1 writes mOut[k=1, ...].
        for r in cutlass.range_constexpr(2):
            row_idx = Int32(r * 32) + Int32(tidx)
            for c in cutlass.range_constexpr(4):
                addr = sW13_addr + row_idx * Int32(16) + Int32(c * 4)
                val = ld_shared_i32_relaxed(addr)
                # Write 4 bytes back to mOut at (k, row_idx, c*4..c*4+3).
                mOut[Int32(k), row_idx, Int32(c * 4 + 0)] = Uint8(Uint32(val) & Uint32(0xFF))
                mOut[Int32(k), row_idx, Int32(c * 4 + 1)] = Uint8((Uint32(val) >> Uint32(8)) & Uint32(0xFF))
                mOut[Int32(k), row_idx, Int32(c * 4 + 2)] = Uint8((Uint32(val) >> Uint32(16)) & Uint32(0xFF))
                mOut[Int32(k), row_idx, Int32(c * 4 + 3)] = Uint8((Uint32(val) >> Uint32(24)) & Uint32(0xFF))


def main():
    device = torch.device("cuda")
    # mIn: 64 rows × 32 bytes/row, deterministic.
    src = (torch.arange(64, device=device).unsqueeze(1) * 32 +
           torch.arange(32, device=device).unsqueeze(0)).to(torch.uint8).contiguous()
    # mOut: (2 K-blocks, 64 rows, 16 bytes/row) — what we read from SMEM.
    dst = torch.full((2, 64, 16), 0xCC, dtype=torch.uint8, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(driver, _to_cute(src, cutlass.Uint8), _to_cute(dst, cutlass.Uint8), stream)
    compiled(_to_cute(src, cutlass.Uint8), _to_cute(dst, cutlass.Uint8), stream)
    torch.cuda.synchronize()

    # Expected: dst[k, r, c] = src[r, k*16 + c].
    expected = torch.empty(2, 64, 16, dtype=torch.uint8, device=device)
    for k in range(2):
        expected[k] = src[:, k*16:(k+1)*16]

    print(f"K-block 0 row 0 src   : {src[0, :16].tolist()}")
    print(f"K-block 0 row 0 dst   : {dst[0, 0].tolist()}")
    print(f"K-block 0 max diff: {(dst[0].to(torch.int32) - expected[0].to(torch.int32)).abs().max().item()}")
    print(f"K-block 1 row 0 src   : {src[0, 16:].tolist()}")
    print(f"K-block 1 row 0 dst   : {dst[1, 0].tolist()}")
    print(f"K-block 1 max diff: {(dst[1].to(torch.int32) - expected[1].to(torch.int32)).abs().max().item()}")


if __name__ == "__main__":
    main()
