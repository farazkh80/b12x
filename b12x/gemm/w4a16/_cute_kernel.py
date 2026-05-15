"""CuTe-DSL W4A16 dense GEMM kernel for SM120 / SM121 (v2 scaffold).

This is the **scaffold** for the v2 CuTe-DSL kernel.  It is structured
to mirror the bf16-stage / bf16-MMA / FP4-dequant pattern proven by
``b12x.moe.fused.micro.MoEMicroKernelBackend`` (the W4A16 MoE micro
kernel), but stripped to a single matmul — no routing, no expert dim,
no FC2, no intermediate buffer, no scatter.

The Triton kernel in ``_triton_kernel.py`` is the **production v1**.
This file is enabled by ``B12X_GEMM_W4A16_USE_CUTE=1`` and currently
raises ``NotImplementedError`` until the kernel body lands.  The
class scaffolding (imports, signature, launch glue) is in place so
the body-lift can drop in with minimal additional plumbing.

## What needs to be lifted from ``b12x/moe/fused/micro.py``

Every ``cutlass.const_expr(self.w4a16_mode)`` branch in MoE micro is
the W4A16-specific code path — exactly what we need to keep when
dropping routing.

Key sites (read in order):

| Line | Role | Action for dense |
|------|------|------------------|
| 354 | ``w4a16_mode: bool = False`` ctor arg | Remove flag — always true here. |
| 366 | ``self.w4a16_mode = ...`` | Drop. |
| 477-485 | ``is_supported`` W4A16 constraints | Lift; relax K % 512 to K % 128 (see micro.py in this dir). |
| 484 | ``w4a16_rowpair_fc2`` setup | Drop — no FC2. |
| 1040-1044 | bf16-stage A operand (preamble, t=prev_t cache miss) | Lift — this is the A-operand staging into ``smem_xh``. |
| 1096-1100 | bf16-stage A operand (main FC1 loop, per-route) | Lift — same staging, called per K-segment. |
| 1523, 1588 | FC1 dot product with bf16-stage operand | Lift — this is the inner bf16×FP4 MMA call. |
| 1694, 1700 | FC2 path branches | Drop — no FC2. |

Things to **strip** when porting:

- ``cfg.weight_E``, ``cfg.num_topk``, ``topk_ids``, ``topk_weights``,
  ``input_gs``, ``down_input_scale``, ``intermediate``, ``w2_*``,
  ``barrier_count``, ``barrier_epoch``, the scatter-output epilogue
  (replace with a direct ``[m, n]`` store).
- ``cfg.fc2_*``, ``cfg.inter_*``, ``cfg.fc1_chunks_per_block`` (FC2
  pipelining doesn't apply).
- ``self.is_gated`` (always False — dense has no gate).
- ``share_input_across_experts``, ``share_expert_scales``,
  ``dynamic_down_scale`` (MoE-only).
- The ``t = route_idx // num_topk`` token-resolve and the eid lookup.

The minimal cute kernel signature should be (mirroring the staged-bf16
path of MoE micro's ``__call__`` at line 1718):

```
x_ptr, w_ptr, w_sf_ptr, w_alpha_ptr, out_ptr,
m_val, n_val, k_val, grid_x, stream
```

The K loop, smem staging, and FC1 dot product (with W4A16 flag forced
true) are the body.  See the MoE micro lines listed above.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def _cute_backend_enabled() -> bool:
    return os.environ.get("B12X_GEMM_W4A16_USE_CUTE") == "1"


class DenseGemmW4A16CuteKernel:
    """CuTe-DSL W4A16 dense GEMM (scaffold — body lift pending).

    Once the body lands this class becomes the default backend; until
    then it raises ``NotImplementedError`` on first call and the
    public entry falls back to the Triton path.
    """

    @classmethod
    def is_supported(cls, m: int, k: int, n: int) -> bool:
        # Same envelope as the Triton backend; will tighten once the
        # CuTe-DSL kernel's MMA-tile constraints are encoded.
        from .micro import DenseGemmW4A16MicroKernel
        return DenseGemmW4A16MicroKernel.is_supported(m, k, n)

    def __init__(self) -> None:
        self._compiled = None

    def __call__(
        self,
        x: torch.Tensor,
        w_fp4: torch.Tensor,
        w_blockscale: torch.Tensor,
        w_alpha: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "CuTe-DSL W4A16 dense kernel body has not been lifted yet. "
            "See module docstring for the porting plan from "
            "b12x.moe.fused.micro.MoEMicroKernelBackend's w4a16_mode "
            "branches.  Set B12X_GEMM_W4A16_USE_CUTE=0 (the default) "
            "to use the Triton backend in the meantime."
        )


__all__ = ["DenseGemmW4A16CuteKernel", "_cute_backend_enabled"]
