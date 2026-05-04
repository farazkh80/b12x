# T2 Step 3 hypothesis

Try widening the single-row-warp CUTLASS atom smoke path to `tile_n=64` with two compute warps per CTA. This keeps `tile_m=16`, which was the only known-correct row topology in Step 1, but attempts to halve the number of CTAs along N and reduce overhead.

Expected result: if the B-fragment layout scales to a second column warp, headline latency should improve. If correctness fails, the smoke implementation is limited to a single `16x32` warp tile and the track should stop without carrying this candidate forward.
