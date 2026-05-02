from __future__ import annotations

import torch

try:
    import nvtx
except ImportError:
    nvtx = None


def make_fp8(shape: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) / 8
    amax = src.abs().max().clamp_min(1e-6).to(torch.float32)
    q_scale = torch.finfo(torch.float8_e4m3fn).max / amax
    q = (src * q_scale).to(torch.float8_e4m3fn)
    return q, q_scale.reciprocal().reshape(1)


def make_case(M: int, K: int, N: int, *, out_dtype: torch.dtype, scale_shape: str, seed: int):
    a, scale_a_scalar = make_fp8((M, K), seed)
    b, scale_b_scalar = make_fp8((N, K), seed + 100000)
    b_t = b.T
    out = torch.empty((M, N), device="cuda", dtype=out_dtype)

    if scale_shape == "scalar1":
        scale_a = scale_a_scalar
        scale_b = scale_b_scalar
    elif scale_shape == "scalar0":
        scale_a = scale_a_scalar[0]
        scale_b = scale_b_scalar[0]
    elif scale_shape == "row_col":
        scale_a = scale_a_scalar.expand(M, 1).contiguous()
        scale_b = scale_b_scalar.expand(1, N).contiguous()
    else:
        raise ValueError(scale_shape)

    def one() -> None:
        torch._scaled_mm(
            a,
            b_t,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
            out=out,
        )

    return one


def run_case(label: str, one, iters: int) -> None:
    for _ in range(4):
        one()
    torch.cuda.synchronize()
    if nvtx is None:
        for _ in range(iters):
            one()
    else:
        with nvtx.annotate(label):
            for _ in range(iters):
                one()
    torch.cuda.synchronize()


def main() -> None:
    torch.cuda.set_device(0)
    shapes = []
    for M in (4, 8, 16):
        for K in (2048, 2096, 2688, 4096, 5376):
            shapes.append((M, K, 4096))
    for M in (1, 2, 4, 8, 16, 32):
        for K in (2688, 4096, 5376):
            shapes.append((M, K, 5376))

    cases = []
    idx = 0
    for M, K, N in shapes:
        for out_dtype_name, out_dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            for scale_shape in ("scalar1", "scalar0", "row_col"):
                label = f"fp8_variant M={M},K={K},N={N},out={out_dtype_name},scale={scale_shape}"
                cases.append((label, make_case(M, K, N, out_dtype=out_dtype, scale_shape=scale_shape, seed=11000 + idx)))
                idx += 1
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    try:
        for label, one in cases:
            run_case(label, one, iters=5)
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
