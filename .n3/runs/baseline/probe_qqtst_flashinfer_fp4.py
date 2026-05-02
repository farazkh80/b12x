from __future__ import annotations

import os
import pathlib
import sys

os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
os.environ["FLASHINFER_WORKSPACE_DIR"] = "/tmp/flashinfer-workspace"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

try:
    import nvtx
except ImportError:
    nvtx = None

from b12x.cute.fp4 import quantize_grouped_nvfp4_torch
from b12x.cute.utils import convert_sf_from_mma_layout
from flashinfer.gemm import mm_fp4


def make_quantized_operand(M: int, K: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    source = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16, generator=gen) / 4
    row_counts = torch.full((1,), M, dtype=torch.int32, device="cuda")
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32,
        device="cuda",
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return packed[:, :, 0].contiguous(), convert_sf_from_mma_layout(scales, m=M, k=K, num_groups=1), global_scale


def make_case(M: int, K: int, N: int, seed: int):
    a_fp4, a_sf, a_gs = make_quantized_operand(M, K, seed)
    b_fp4, b_sf, b_gs = make_quantized_operand(N, K, seed + 100000)
    alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    def one() -> None:
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_sf.T,
            alpha,
            torch.bfloat16,
            out,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
            use_nvfp4=True,
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
    for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3071):
        shapes.append((M, 2688, 4096))
    for M in (1, 2, 4, 8, 16, 32, 64, 128):
        shapes.append((M, 4096, 5376))
    for M in (1, 2, 4, 8, 16, 32, 64):
        shapes.append((M, 2688, 5376))

    cases = []
    for idx, (M, K, N) in enumerate(shapes):
        cases.append((f"flashinfer_fp4 shape M={M},K={K},N={N}", make_case(M, K, N, 7000 + idx)))
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    try:
        for label, one in cases:
            run_case(label, one, iters=8)
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
