# Headline remeasurement summary

Shape: `M=32,K=5376,N=5376`.

Command artifact: `.n3/runs/tracks/remeasure_compare_variants.log`.

Same-run comparison using identical inputs, `warmup=20,iters=100,repeats=9`:

| path | eager median us | graph median us | correctness vs torch |
| --- | ---: | ---: | --- |
| `torch._scaled_mm` | 103.964 | 103.572 | reference |
| current b12x inline-PTX parity floor (`TileN=32`) | 108.996 | 108.744 | exact vs torch output in this run |
| archived T1 best (`TileN=64`) | 115.555 | 115.451 | exact vs torch output in this run |

Result: neither custom inline-PTX variant beats `torch._scaled_mm` in the same-run remeasurement. The current parity floor is closer to torch than the archived T1 best in this cleaner same-run comparison.

## Difference between current parity floor and archived T1 best

Both use the same inline PTX MMA instruction and same `TileM=32,StageK=256,block=128` hot path. The only kernel configuration difference is `TileN`:

| config | `TileM` | `TileN` | `StageK` | CTAs on headline | dynamic shared memory / CTA | effect |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| parity floor | 32 | 32 | 256 | 168 | 16,384 bytes | more CTAs, higher occupancy opportunity |
| T1 best archive | 32 | 64 | 256 | 84 | 24,576 bytes | halves CTA count, stages twice as many B columns |

The T1 archive result looked slightly better in its earlier verifier run because it reduced redundant A staging and that run had a slower simultaneous baseline. In this same-run remeasurement, halving the CTA count and increasing shared memory per CTA loses more than it saves: `TileN=64` is about 6.6 us slower eager and 6.7 us slower graph than the current `TileN=32` parity floor.

Source was restored after the temporary `TileN=64` edit; current `b12x/gemm/fp8_dense_cuda_ext.cu` is back to `TileN=32,StageK=256`.
