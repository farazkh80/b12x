# T3 Step 2 hypothesis

Skipped. Step 1 showed Triton FP8 `tl.dot` does not lower to native `kind::f8f6f4` or `kind::mxf8f6f4` MMA on this SM121 stack. Autotuning `(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)` would tune a fallback FP16-MMA path and is outside the intended Track 3 target.
