# T2 Step 2 result

Candidate: CUTLASS SM120 atom smoke path with `tile_m=32`, `tile_n=32`, `stage_k=256`, and `block=128`.

Result: rejected before benchmarking. The smoke check failed exact correctness for `M=32,K=32,N=32` with 992/1024 mismatched elements and max absolute difference 47.75. This indicates the Step 1 register/shared-memory layout is only valid for the single-row-warp `tile_m=16` smoke topology and cannot be enlarged by simply adding row warps.

Decision: restore the Step 1 correct `tile_m=16`, `tile_n=32`, `block=32` path before Step 3.
