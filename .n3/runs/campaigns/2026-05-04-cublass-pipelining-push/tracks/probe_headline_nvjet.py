"""Tiny script that calls torch._scaled_mm at the headline shape so nsys can
capture the cuBLASLt-picked nvjet kernel name."""

import torch
from torch.cuda import cudart

M, K, N = 32, 5376, 5376
a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
sa = torch.ones((), device="cuda", dtype=torch.float32)
sb = torch.ones((), device="cuda", dtype=torch.float32)

# Warm up — first call resolves cuBLASLt heuristic and may launch differently
for _ in range(5):
    torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
torch.cuda.synchronize()

# Profiled region.
cudart().cudaProfilerStart()
for _ in range(20):
    torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
torch.cuda.synchronize()
cudart().cudaProfilerStop()
