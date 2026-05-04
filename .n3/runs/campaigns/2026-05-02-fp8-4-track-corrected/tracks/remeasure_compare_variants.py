from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "b12x/gemm/fp8_dense_cuda_ext.cu"
TMP = Path("/tmp/b12x_fp8_variant_compare")
M, K, N = 32, 5376, 5376
WARMUP = 20
ITERS = 100
REPEATS = 9


def build_variant(name: str, tile_n: int):
    TMP.mkdir(parents=True, exist_ok=True)
    source = TMP / f"{name}.cu"
    text = SRC.read_text()
    text = text.replace("constexpr int tile_n = 32;\n    constexpr int stage_k = 256;", f"constexpr int tile_n = {tile_n};\n    constexpr int stage_k = 256;", 1)
    source.write_text(text)
    build_dir = TMP / f"build_{name}"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=name,
        sources=[str(source)],
        build_directory=str(build_dir),
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "-gencode=arch=compute_121a,code=sm_121a",
        ],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def make_fp8(shape: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) / 8
    return src.to(torch.float8_e4m3fn).contiguous(), torch.ones((), device="cuda", dtype=torch.float32)


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def bench(fn) -> dict[str, float | list[float]]:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(REPEATS):
        start.record()
        for _ in range(ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / ITERS)
    return summarize(samples)


def graph_callable(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def main() -> None:
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    parity = build_variant("b12x_fp8_dense_parity_tn32", 32)
    best = build_variant("b12x_fp8_dense_t1best_tn64", 64)

    a, scale_a = make_fp8((M, K), 1000)
    b, scale_b = make_fp8((N, K), 2000)
    baseline_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    parity_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    best_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    def baseline():
        return torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16, out=baseline_out)

    def parity_fn():
        parity.fp8_dense_gemm(a, b, scale_a.reshape(1).contiguous(), scale_b.reshape(1).contiguous(), parity_out)
        return parity_out

    def best_fn():
        best.fp8_dense_gemm(a, b, scale_a.reshape(1).contiguous(), scale_b.reshape(1).contiguous(), best_out)
        return best_out

    baseline(); parity_fn(); best_fn(); torch.cuda.synchronize()
    ref = baseline_out.float()
    correctness = {}
    for label, out in [("parity_tn32", parity_out), ("t1best_tn64", best_out)]:
        diff = (out.float() - ref).abs()
        correctness[label] = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "cos": float(torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()),
        }

    results = {
        "shape": {"M": M, "K": K, "N": N},
        "warmup": WARMUP,
        "iters": ITERS,
        "repeats": REPEATS,
        "variants": {
            "parity_tn32": {"tile_m": 32, "tile_n": 32, "stage_k": 256, "grid_ctas": (M // 32) * (N // 32), "block_threads": 128, "dynamic_smem_bytes": 32 * 256 + 32 * 256},
            "t1best_tn64": {"tile_m": 32, "tile_n": 64, "stage_k": 256, "grid_ctas": (M // 32) * (N // 64), "block_threads": 128, "dynamic_smem_bytes": 32 * 256 + 64 * 256},
        },
        "correctness": correctness,
        "eager": {
            "baseline": bench(baseline),
            "parity_tn32": bench(parity_fn),
            "t1best_tn64": bench(best_fn),
        },
        "graph": {
            "baseline": bench(graph_callable(baseline)),
            "parity_tn32": bench(graph_callable(parity_fn)),
            "t1best_tn64": bench(graph_callable(best_fn)),
        },
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
