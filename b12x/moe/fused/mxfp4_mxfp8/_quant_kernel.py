"""On-device SwiGLU + MXFP8 quantize cute kernel."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x.cute.fp4 import (
    cvt_f32_to_e4m3,
    warp_reduce,
)


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


@dsl_user_op
def floor_f32_to_s32(a: Float32, *, loc=None, ip=None) -> Int32:
    """floor(x) as int32, via PTX `cvt.rmi.s32.f32` (round-to-minus-infinity)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rmi.s32.f32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def imax_s32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.s32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def imin_s32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "min.s32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _make_kernel(is_silu: bool):
    """Compile-time-specialized kernel: builds either silu or relu² version."""

    @cute.kernel
    def kernel(mIn, mOut, mSf):
        row = cute.arch.block_idx()[0]
        kb = cute.arch.block_idx()[1]
        tid = cute.arch.thread_idx()[0]
        j = kb * Int32(32) + Int32(tid)
        half_i = mOut.shape[1]

        # Initialize `val` so cute's flow analysis sees it pre-branch.
        val = Float32(0.0)
        if cutlass.const_expr(is_silu):
            gate = Float32(mIn[row, j])
            up = Float32(mIn[row, j + half_i])
            sigmoid_up = Float32(1.0) / (Float32(1.0) + cute.math.exp(-up, fastmath=True))
            val = gate * up * sigmoid_up
        else:
            x = Float32(mIn[row, j])
            relu_x = cute.arch.fmax(x, Float32(0.0))
            val = relu_x * relu_x

        # |val| via fmax(val, -val)
        abs_val = cute.arch.fmax(val, -val)
        amax = warp_reduce(abs_val, lambda a, b: cute.arch.fmax(a, b), width=32)

        # ceil(log2(x)) = -floor(-log2(x)).
        inv_max = Float32(1.0) / Float32(448.0)
        safe_amax = cute.arch.fmax(amax * inv_max, Float32(1.401298464e-45))
        log_val = cute.math.log2(safe_amax, fastmath=True)
        ceil_log2 = -floor_f32_to_s32(-log_val)

        sf_byte = imin_s32(Int32(254), imax_s32(Int32(0), ceil_log2 + Int32(127)))
        scale = cute.math.exp2(Float32(sf_byte - Int32(127)), fastmath=True)
        inv_scale = Float32(1.0) / scale

        e4m3_byte = cvt_f32_to_e4m3(val * inv_scale)
        mOut[row, j] = e4m3_byte

        if tid == Int32(0):
            mSf[row, kb] = Uint32(sf_byte)

    @cute.jit
    def driver(
        mIn: cute.Tensor,
        mOut: cute.Tensor,
        mSf: cute.Tensor,
        stream: cuda.CUstream,
    ):
        T_dim = mOut.shape[0]
        K_blocks = mSf.shape[1]
        kernel(mIn, mOut, mSf).launch(
            grid=(T_dim, K_blocks, 1), block=[32, 1, 1], stream=stream,
        )

    return driver


_compile_cache = {}


def _compiled_quant(T_dim: int, FC1_N: int, I: int, K_blocks: int, is_silu: bool):
    key = (T_dim, FC1_N, I, K_blocks, is_silu)
    if key in _compile_cache:
        return _compile_cache[key]
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dummy_in = torch.zeros(T_dim, FC1_N, dtype=torch.float32, device=device)
    dummy_out = torch.zeros(T_dim, I, dtype=torch.int32, device=device)
    dummy_sf = torch.zeros(T_dim, K_blocks, dtype=torch.int32, device=device)
    driver = _make_kernel(is_silu)
    compiled = cute.compile(
        driver,
        _to_cute(dummy_in, cutlass.Float32),
        _to_cute(dummy_out, cutlass.Uint32),
        _to_cute(dummy_sf, cutlass.Uint32),
        stream,
    )
    _compile_cache[key] = compiled
    return compiled


def silu_mxfp8_quantize(
    in_f32: torch.Tensor, *, activation: str = "silu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 (T, FC1_N) → MXFP8 (T, I) e4m3 + (T, I/32) ue8m0."""
    T_dim, FC1_N = in_f32.shape
    is_silu = activation == "silu"
    if is_silu:
        assert FC1_N % 2 == 0, "SwiGLU requires even FC1_N"
        I = FC1_N // 2
    elif activation == "relu2":
        I = FC1_N
    else:
        raise ValueError(f"unsupported activation {activation!r}")
    assert I % 32 == 0, f"I={I} must be multiple of 32"
    K_blocks = I // 32

    device = in_f32.device
    out_buf = torch.zeros(T_dim, I, dtype=torch.int32, device=device)
    sf_buf = torch.zeros(T_dim, K_blocks, dtype=torch.int32, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled = _compiled_quant(T_dim, FC1_N, I, K_blocks, is_silu)
    compiled(
        _to_cute(in_f32, cutlass.Float32),
        _to_cute(out_buf, cutlass.Uint32),
        _to_cute(sf_buf, cutlass.Uint32),
        stream,
    )
    torch.cuda.synchronize()
    out_e4m3 = (out_buf & 0xFF).to(torch.uint8)
    sf_ue8m0 = (sf_buf & 0xFF).to(torch.uint8)
    return out_e4m3, sf_ue8m0
