from __future__ import annotations

import torch

try:
    import nvtx
except ImportError:
    nvtx = None


def profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def make_fp8(shape: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) / 8
    amax = src.abs().max().clamp_min(1e-6).to(torch.float32)
    q_scale = torch.finfo(torch.float8_e4m3fn).max / amax
    q = (src * q_scale).to(torch.float8_e4m3fn)
    deq_scale = q_scale.reciprocal().reshape(1)
    return q, deq_scale


def run_shape(M: int, K: int, N: int, iters: int) -> None:
    a, scale_a = make_fp8((M, K), 1000 + M)
    b, scale_b = make_fp8((N, K), 2000 + M)
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

    for _ in range(5):
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
    shapes = [
        (1, 2688, 4096),
        (2, 2688, 4096),
        (4, 2688, 4096),
        (8, 2688, 4096),
        (16, 2688, 4096),
        (32, 2688, 4096),
        (64, 2688, 4096),
        (128, 2688, 4096),
        (256, 2688, 4096),
        (512, 2688, 4096),
        (1024, 2688, 4096),
    ]
    profiler_start()
    try:
        for shape in shapes:
            run_shape(*shape, iters=20)
    finally:
        profiler_stop()


if __name__ == "__main__":
    main()
