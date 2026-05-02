# Headline FP8 dense GEMM shape

Chosen shape: `M=1, K=4096, N=4096` (`decode`).

`torch._scaled_mm` measured `124.768 us` eager and `124.832 us` CUDA graph in `.n3/runs/baseline/fp8_dense_baseline.json`, making this the FP8 shape with baseline eager median closest above 80 us. It is representative of production decode because it corresponds to the Q/O projection family for a single decode token over the Nemotron hidden size; decode latency is the path where a 70-80 us target is meaningful, while prefill shapes are much larger and TP4 decode shapes measured below 80 us already. The existing b12x inline-PTX padded baseline for this shape measured `264.208 us` eager and `264.208 us` graph, so Track 1 should treat cuBLASLt/`torch._scaled_mm` as the primary no-regression floor and the inline PTX as the correctness/PTX-reference floor.
