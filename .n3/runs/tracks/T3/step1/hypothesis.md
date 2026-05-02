# T3 Step 1 hypothesis

Implement a minimal Triton FP8 GEMM with `tl.dot` and check both correctness and generated PTX. The critical question is whether Triton 3.6.0 on SM121 lowers FP8 `tl.dot` to native `kind::f8f6f4`/`kind::mxf8f6f4` MMA or silently converts to FP16 tensor-core MMA.

Decision rule: if PTX does not contain native FP8 MMA, abort Track 3 per plan because subsequent Triton autotuning would optimize the wrong instruction path.
