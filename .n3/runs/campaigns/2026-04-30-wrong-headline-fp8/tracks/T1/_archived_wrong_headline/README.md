# Archived: Track 1 against the wrong headline shape

These step{1,2,3} directories were produced when AVO picked headline shape
`(M=1, K=4096, N=4096)` by following the "closest above 80 µs" rule. The
inline-PTX kernel `b12x/gemm/fp8_dense_cuda_ext.cu` was hand-tuned for
`(M=32, K=5376, N=5376)` (the production nsys profile shape, per
`b12x/README.md` tuning notes), so the M=1 measurements are uninformative —
the kernel pads to M=16 internally, and AVO's "small-M decode fast path"
direction was a wrong-shape rabbit hole.

Kept as a record of the brief revision; not used for any gating decision.
The fresh Track 1 against `(M=32, K=5376, N=5376)` lives in `T1/step{1,2,3}/`
at the parent level.
