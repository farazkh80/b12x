"""Debug: compare CTA 0 of multi-CTA fused vs single-CTA fused for same data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from b12x.moe.fused.mxfp4_mxfp8._fused_kernel import run_fused_silu_full, run_fused_silu
from b12x.moe.fused.mxfp4_mxfp8.expert_chain import expert_chain_b12x
from b12x.moe.fused.mxfp4_mxfp8.reference import quantize_to_mxfp4, quantize_to_mxfp8


def main():
    torch.manual_seed(7)
    device = torch.device("cuda")
    M, K_in, I, K_out, FC1_N = 32, 32, 32, 32, 64

    x_f32 = torch.randn(M, K_in, dtype=torch.float32, device=device) * 0.5
    w13_f32 = torch.randn(FC1_N, K_in, dtype=torch.float32, device=device) * 0.3
    w2_f32 = torch.randn(K_out, I, dtype=torch.float32, device=device) * 0.3

    x_e4m3, x_sf = quantize_to_mxfp8(x_f32)
    w13_p, w13_sf = quantize_to_mxfp4(w13_f32)
    w2_p, w2_sf = quantize_to_mxfp4(w2_f32)

    # Single-CTA path on rows 0..15 only.
    res0 = expert_chain_b12x(
        x_e4m3[:16], x_sf[:16], w13_p, w13_sf, w2_p, w2_sf, activation="silu",
    ).out_bf16

    res1 = expert_chain_b12x(
        x_e4m3[16:32], x_sf[16:32], w13_p, w13_sf, w2_p, w2_sf, activation="silu",
    ).out_bf16

    # Multi-CTA path covering both rows.
    out_multi = run_fused_silu_full(x_e4m3, x_sf, w13_p, w13_sf, w2_p, w2_sf)
    print("multi[0, :8]:", out_multi[0, :8].tolist())
    print("ref0[0, :8]:", res0[0, :8].tolist())
    print()
    print("multi[16, :8]:", out_multi[16, :8].tolist())
    print("ref1[0, :8]:", res1[0, :8].tolist())
    print()

    err0 = (out_multi[:16].to(torch.float32) - res0.to(torch.float32)).abs().max().item()
    err1 = (out_multi[16:].to(torch.float32) - res1.to(torch.float32)).abs().max().item()
    print(f"CTA 0 (rows 0-15) max abs err: {err0:.4f}")
    print(f"CTA 1 (rows 16-31) max abs err: {err1:.4f}")


if __name__ == "__main__":
    main()
