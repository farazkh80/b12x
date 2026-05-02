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


def run_shape(M: int, K: int, N: int, iters: int, seed: int) -> None:
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

    for _ in range(4):
        one()
    torch.cuda.synchronize()

    label = f"shape M={M},K={K},N={N}"
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
    for M in (1, 2, 4, 8, 16):
        for K in (1024, 1344, 2048, 2096, 2688, 4096, 5376):
            for N in (1024, 1344, 2048, 2688, 4096, 5376):
                shapes.append((M, K, N))

    torch.cuda.cudart().cudaProfilerStart()
    try:
        for idx, (M, K, N) in enumerate(shapes):
            run_shape(M, K, N, iters=8, seed=1000 + idx)
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
