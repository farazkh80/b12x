<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

# Adaptation Test — Detailed Guide

## The Rule

Two kernels belong to the **same problem** if they share the same optimization
skeleton — the tiling strategy, memory access pattern, synchronization, and
pipeline structure. The math in the inner loop is what varies.

## Decision Matrix

| Change Type | Example | Same Problem? |
|------------|---------|---------------|
| Swap pointwise math | sigmoid → silu | Yes |
| Change normalization formula | softmax → layernorm | Yes (same row-reduction skeleton) |
| Change memory layout | TN gemm → NN gemm | No (fundamentally different access pattern) |
| Change loop structure | single-tensor → multi-tensor Adam | No (new variant) |
| Change dtype | FP16 → BF16 | Yes (same exemplar, agent adapts) |
| Change block/tile size | BLOCK=128 → BLOCK=256 | Yes (tuning parameter) |
| Add fusion | standalone → fused with epilogue | No (structural change) |
| Change reduction axis | row-wise → column-wise | No (different skeleton) |

## Examples of Problem Groupings

| Problem Name | Operations That Share This Skeleton |
|-------------|-------------------------------------|
| `unary-elementwise` | sigmoid, silu, relu, gelu, tanh, exp, rsqrt |
| `binary-elementwise` | add, mul, lerp, fused-add-mul |
| `row-reduction` | softmax, layernorm, rmsnorm |
| `gemm-tn` | any TN-layout matmul |
| `gemm-nn` | any NN-layout matmul |
| `fused-attention` | flash-attention variants |
| `optimizer-update` | adam, lamb, sgd (single-tensor) |

## When in Doubt

If you are unsure whether two kernels share a skeleton, ask:

1. Would the same block size / tile shape work for both?
2. Would the same shared memory allocation work for both?
3. Would the same memory access pattern (coalesced, strided, etc.) work for both?

If all three answers are **yes**, they are the same problem.
