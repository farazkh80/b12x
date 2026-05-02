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


def make_case(M: int, K: int, N: int, seed: int):
    a, scale_a = make_fp8((M, K), seed)
    b, scale_b = make_fp8((N, K), seed + 100000)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    def one() -> None:
        torch._scaled_mm(
            a,
            b.T,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
            out=out,
        )

    return one


def run_case(label: str, one, iters: int) -> None:
    for _ in range(5):
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
    for M in (24, 32, 40, 48, 56, 64, 80, 96, 112, 128):
        for K in (2048, 2096, 2688, 3072, 3584, 4096, 5376):
            shapes.append((M, K, 5376))

    cases = []
    for idx, (M, K, N) in enumerate(shapes):
        cases.append((f"fp8_n5376_focus M={M},K={K},N={N}", make_case(M, K, N, 21000 + idx)))
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    try:
        for label, one in cases:
            run_case(label, one, iters=8)
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
