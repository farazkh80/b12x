# T2 Step 2 hypothesis

Try to increase useful work per CTA versus the Step 1 CUTLASS atom smoke by launching four warps per CTA with the same 32-column tile and a deeper 256-K stage, matching the Track 1 hot-path CTA geometry.

Expected result: if the CUTLASS atom is layout-compatible for multiple row warps, the larger CTA should reduce grid launch/latency overhead and improve headline time. If correctness fails, the smoke path is only valid for one row warp and should not be used as a production candidate.
