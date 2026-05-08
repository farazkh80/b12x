"""Debug cp.async: load 32 × 16 bytes from global to SMEM, write SMEM back to global, compare."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x.cute.fp4 import get_ptr_as_int64, shared_ptr_to_u32


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


@dsl_user_op
def cp_async_16B(smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::64B [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def driver(mIn, mOut, stream):
    kernel(mIn, mOut).launch(grid=(1, 1, 1), block=[32, 1, 1], stream=stream)


@cute.kernel
def kernel(mIn, mOut):
    tidx = cute.arch.thread_idx()[0]
    # 32 rows × 16 cols, each thread copies its row.
    for col_idx in cutlass.range_constexpr(16):
        mOut[Int32(tidx), Int32(col_idx)] = mIn[Int32(tidx), Int32(col_idx)]


def main():
    device = torch.device("cuda")
    # Back to (32, 16) to investigate the loop quirk.
    src = (torch.arange(32, device=device).unsqueeze(1) * 16 +
           torch.arange(16, device=device).unsqueeze(0)).to(torch.uint8).contiguous()
    print(f"src strides: {src.stride()}")
    dst = torch.full((32, 16), 99, dtype=torch.uint8, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(driver, _to_cute(src, cutlass.Uint8), _to_cute(dst, cutlass.Uint8), stream)
    compiled(_to_cute(src, cutlass.Uint8), _to_cute(dst, cutlass.Uint8), stream)
    torch.cuda.synchronize()

    print(f"src ptr: {src.data_ptr():x}  dst ptr: {dst.data_ptr():x}")
    print(f"src.shape: {src.shape} dst.shape: {dst.shape}")
    print(f"src[0, :8]: {src[0, :8].tolist()}")
    print(f"dst[0, :8]: {dst[0, :8].tolist()}")
    print(f"src.sum(): {src.sum().item()}  dst.sum(): {dst.sum().item()}")
    eq = torch.equal(src, dst)
    print(f"src == dst: {eq}")
    if not eq:
        diff = (src.to(torch.int32) - dst.to(torch.int32)).abs()
        print(f"max abs diff: {diff.max().item()}")
        print(f"first row src: {src[0].tolist()}")
        print(f"first row dst: {dst[0].tolist()}")


if __name__ == "__main__":
    main()
