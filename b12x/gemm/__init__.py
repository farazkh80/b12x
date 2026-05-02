from .fp8_dense import fp8_dense_gemm, fp8_dense_gemm_mma
from .fp8_dense_cuda import fp8_dense_gemm_cuda

__all__ = [
    "fp8_dense_gemm",
    "fp8_dense_gemm_mma",
    "fp8_dense_gemm_cuda",
]


def __getattr__(name: str):
    if name in {"DenseGemmKernel", "dense_gemm"}:
        from .dense import DenseGemmKernel, dense_gemm

        return {"DenseGemmKernel": DenseGemmKernel, "dense_gemm": dense_gemm}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
