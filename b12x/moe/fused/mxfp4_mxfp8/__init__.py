"""Fused MoE kernel for MXFP4 weights × MXFP8 activations on SM120.

Targeting OpenAI gpt-oss MoE shapes. UE8M0 (E8M0) per-block-32 scales
on both operands; no per-tensor global scale.
"""
