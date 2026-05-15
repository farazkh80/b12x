"""W4A16 dense GEMM kernels for SM120 / SM121.

Consumes bf16 activations and FP4-packed weights directly; no online
activation quantization. See ``.claude_docs/.../W4A16_DENSE_DESIGN.md``.
"""

from .reference import dense_reference_w4a16, quantize_dense_weight_to_fp4

__all__ = [
    "dense_reference_w4a16",
    "quantize_dense_weight_to_fp4",
]
