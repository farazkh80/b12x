# Baseline JSON for the AVO autonomous loop

Before launching AVO, capture the baseline that `verify_moe_perf.py` produces
on the unmodified b12x tree. Commit it here so the loop has a stable
reference.

```bash
# Inside the container (after build-sm120.sh or build-sm121.sh):
cd /workspace/b12x
SHA=$(git rev-parse --short HEAD)
python -m benchmarks.verify_moe_perf \
  --activation silu --scale-contract shared \
  --batch-sizes 1 4 32 80 \
  > .n3/baseline/verify_moe_perf.${SHA}.silu.json
python -m benchmarks.verify_moe_perf \
  --activation relu2 --scale-contract shared \
  --batch-sizes 1 4 32 80 \
  > .n3/baseline/verify_moe_perf.${SHA}.relu2.json
git add .n3/baseline/verify_moe_perf.${SHA}.*.json
git commit -m "baseline: verify_moe_perf @ ${SHA}"
```

The TASK.md target ("≥ +5% sustained reduction at every batch size") is
evaluated against `eager_median_us` and `graph_median_us` in these files.
