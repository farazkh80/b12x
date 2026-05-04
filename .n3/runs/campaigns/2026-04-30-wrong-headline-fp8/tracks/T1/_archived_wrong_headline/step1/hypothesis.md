# Track 1 Step 1 hypothesis

Instrument the existing CUDA C inline-PTX kernel without changing `b12x/gemm/fp8_dense_cuda_ext.cu`. The Step 0 baseline shows the headline shape is `M=1,K=4096,N=4096`, while the extension only accepts M multiples of 16; therefore this step uses the verifier's padding path as the parity floor and captures nsys/ncu on the closest native tile shape (`M=16,K=4096,N=4096`) to identify whether the current kernel is dominated by global-to-shared staging, MMA utilization, or launch/occupancy overhead before attempting TMA or scheduler changes.
