# Headline FP8 dense GEMM shape

Chosen shape: **`M=32, K=5376, N=5376`** (`decode`).

## Why this one (overrides the prior "closest to 80 µs" pick)

This is the shape captured in the production nsys profile (`profile_super_nvfp4_disable_b12x.1.nsys-rep`) and is the one the existing `b12x/gemm/fp8_dense_cuda_ext.cu` was hand-tuned for. Latest measured numbers (per `README.md` tuning notes):

- `torch._scaled_mm` / cuBLASLt nvjet: **99.408 µs** median
- b12x inline-PTX kernel:              **102.464 µs** median (correct: max_abs_diff 0.015625, mean_abs_diff 3.27e-7)

That's the race AVO is actually trying to win. The prior pick `(M=1, K=4096, N=4096)` followed the brief's literal "closest to 80 µs" rule, but on that shape the inline-PTX kernel is 264 µs (2× slower than baseline, hardcoded `TileM=32, TileN=32, StageK=256` doesn't fit `M=1`) — there's no useful "parity floor" to optimize against and the shape doesn't appear in the production trace.

## Performance target on this shape

| Reference | Median (eager) |
|---|---|
| **Stretch target** | **70–80 µs** |
| Production cuBLASLt baseline | 99.408 µs |
| b12x inline-PTX (current best) | 102.464 µs |

70–80 µs is 1.25–1.42× faster than cuBLASLt on this shape. Achievable: at FP8 peak (~750 TFLOPs sustained on Spark for a tuned dense GEMM) `(32, 5376, 5376)` is ~1.85 GFLOPs of work → ~2.5 µs compute-bound, far below 80 µs. The bottleneck is memory traffic (128 MB load: A=688 KB, B=29 MB, output=2.2 MB; weights repeatedly reused) and pipeline efficiency. cuBLASLt hits ~99 µs; the headroom is real.

## Validation gates on the headline shape

| Gate | Threshold |
|---|---|
| Correctness vs `torch._scaled_mm` (FP32 reference) | `cos ≥ 0.9995`, `max_abs ≤ 1% × ‖ref‖_∞` |
| **No regression vs `torch._scaled_mm`** | hard requirement (`≤ 99.408 µs`) |
| Parity vs inline-PTX (`fp8_dense_cuda_ext.cu`) | within ±5% (i.e. `≤ 107 µs`) — easy floor |
| Stretch | as close to 70 µs as the hardware allows |

The baseline JSON `.n3/runs/baseline/fp8_dense_baseline.json` does not yet contain a measurement for `(32, 5376, 5376)` — `verify_fp8_dense_gemm_perf.py` has been updated with this shape; re-run it on Spark to fold the headline numbers in.
