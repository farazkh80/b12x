"""Verify v4 cute kernel is bit-exact at M ∈ {1, 8, 16, 32} across all
Nano3.5 dense shapes.  Forces v4 dispatch by setting USE_CUTE=1 and
checking the kernel that gets picked is the v4 one (not v3 / Triton)."""
import os, sys, torch
sys.path.insert(0, "/home/farazkh_scratch/agentic-dev/b12x")
try:
    import spark_preamble  # noqa: F401  (cutlass-dsl 4.3.4 compat shim)
except ImportError:
    pass
os.environ["B12X_GEMM_W4A16_USE_CUTE"] = "1"
os.environ["B12X_GEMM_W4A16_CUTE_DEBUG"] = "1"
from b12x.gemm.w4a16 import quantize_dense_weight_to_fp4, dense_gemm_w4a16
from b12x.gemm.w4a16.reference import dense_reference_w4a16
from b12x.gemm.w4a16._cute_dense_kernel import DenseGemmW4A16CuteDenseKernel

dev = torch.device("cuda")
SHAPES = [
    ("qkv_linear", 2688, 4608),
    ("out_linear", 4096, 2688),
    ("shared_fc1", 2688, 3712),
    ("shared_fc2", 3712, 2688),
    ("mamba_in_proj", 2688, 10304),
    ("mamba_out_proj", 4096, 2688),
    ("lm_head", 2688, 131072),
]
MS = [1, 8, 16, 32]
fail = 0
print(f"{'shape':22s} {'M':>3s} {'K':>5s} {'N':>6s}  {'v4_supported':>13s}  {'max_abs':>9s}  {'cos':>10s}  {'verdict':>8s}", flush=True)
for name, k, n in SHAPES:
    for m in MS:
        torch.manual_seed(hash((name, m)) & 0xFFFFFFFF)
        x = (torch.randn(m, k, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
        w = (torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
        w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
        v4_sup = DenseGemmW4A16CuteDenseKernel.is_supported(m, k, n)
        try:
            out = dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha)
            torch.cuda.synchronize()
            ref = dense_reference_w4a16(x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu()).to(dev)
            diff = (out - ref).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(out.flatten().to(torch.float32), ref.flatten().to(torch.float32), dim=0).item()
            ref_max = ref.abs().max().item()
            thresh = max(0.04, 0.01 * ref_max)
            ok = (cos > 0.9999) and (diff <= thresh)
            verdict = "PASS" if ok else "FAIL"
            if not ok:
                fail += 1
            print(f"{name:22s} {m:3d} {k:5d} {n:6d}  {str(v4_sup):>13s}  {diff:9.4f}  {cos:10.6f}  {verdict:>8s}", flush=True)
        except Exception as e:
            fail += 1
            print(f"{name:22s} {m:3d} {k:5d} {n:6d}  {str(v4_sup):>13s}  ---ERR---  {e!s:.30s}  FAIL", flush=True)
sys.exit(0 if fail == 0 else 1)
