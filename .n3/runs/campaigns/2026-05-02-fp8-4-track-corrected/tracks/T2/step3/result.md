# T2 Step 3 result

Candidate: CUTLASS SM120 atom smoke path with `tile_m=16`, `tile_n=64`, `stage_k=128`, and `block=64`.

Result: rejected before benchmarking. The smoke check failed exact correctness for `M=16,K=32,N=64` with 1006/1024 mismatched elements and max absolute difference 1880.0. The Step 1 register/shared-memory layout is therefore not valid for a second column warp either.

Decision: restore the only correct CUTLASS atom smoke path (`tile_m=16`, `tile_n=32`, `block=32`) and do not carry Track 2 forward as a production candidate because it is slower than baseline and regresses all larger shapes.
