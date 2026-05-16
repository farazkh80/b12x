#!/usr/bin/env python3
"""Correlate Marlin (W4A16) kernel calls in the Nano3.5 prefill regime
to per-(dlsim_op, M) timings, for benchmarking b12x's v5 prefill
kernel against the production Marlin baseline.

Source trace:
    /home/scratch.kangningl_gpu/dlsim_correlation/results/nano3_5/nsys_trace.sqlite

The trace has 10 NVTX iterations of the form
``execute_context_<id>(<prefill_M>)_generation_0(0)``:

* 5 SINGLE-CHUNK iterations: prefill_M ∈ {579, 1521, 1823, 2001, 2015}.
  Every Marlin call in the iteration operates at this M.  These are
  the cleanest source of per-M baselines.
* 5 CHUNKED iterations: cumulative M ∈ {31920, 34048, 36176}.  Each
  Marlin call operates at the internal chunk size, not the cumulative
  M.  Useful as a sanity cross-check.

Attribution: borrows ``attribute_step`` from
``_attribute_kernels.py`` (in the trace directory) but **patches** two
gaps that decode-era code didn't need:

1. ``_causal_conv1d_fwd_kernel`` (prefill) is recognized as the
   M-block anchor — the decode-only classify matches only
   ``_causal_conv1d_update_kernel``.  Without this fix, M-anchors
   never fire in prefill and qkv attribution leaks into shared_fc.

2. Marlin variant tags are IGNORED for the dense ops — at prefill M
   we see new variants (``<128,4,4,8>``, ``<128,4,8,4>``,
   ``<256,4,16,4>``, ``<128,2,8,4>``, …) that the decode-era table
   doesn't list.  Positional order relative to the anchors is stable
   across regimes, so we rely on order alone.

Outputs (under .claude_docs/marlin-baseline/):

* ``marlin_prefill_per_op.csv`` — one row per Marlin call with its
  attributed dlsim_op, iter, prefill_M, and duration.
* ``marlin_prefill_per_variant.csv`` — per (iter, variant) call count
  + duration percentiles (variant view, independent of attribution).
* ``marlin_prefill_summary.md`` — headline table: per-op p50 µs
  across the single-chunk M ladder, ready to drop into the b12x
  benchmark table as the Marlin baseline column.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


SQLITE = "/home/scratch.kangningl_gpu/dlsim_correlation/results/nano3_5/nsys_trace.sqlite"
ATTRIBUTION_SRC = Path(
    "/home/scratch.kangningl_gpu/dlsim_correlation/results/nano3_5/_attribute_kernels.py"
)
OUT_DIR = Path(__file__).resolve().parent.parent / ".claude_docs" / "marlin-baseline"


# NemotronH per-step (single forward pass) Marlin call counts.
PER_STEP_CALL_COUNT = {
    "mamba_in_proj":         23,
    "mamba_output_proj":     23,
    "shared_fc1":            23,
    "shared_fc2":            23,
    "self_attn_qkv_linear":   6,
    "self_attn_out_linear":   6,
    "proj_linear (lm_head)":  1,
}

# (K, N) shapes per dlsim_op, sourced from
# benchmarks/benchmark_dense_gemm_w4a16.py.
OP_SHAPE = {
    "self_attn_qkv_linear":   (2688,   4608),
    "self_attn_out_linear":   (4096,   2688),
    "shared_fc1":             (2688,   3712),
    "shared_fc2":             (3712,   2688),
    "mamba_in_proj":          (2688,  10304),
    "mamba_output_proj":      (4096,   2688),
    "proj_linear (lm_head)":  (2688, 131072),
}

_VARIANT_RE = re.compile(
    r"(marlin(?:_moe_wna16)?)::Marlin"
    r".*?\(int\)(\d+), \(int\)(\d+), \(int\)(\d+), \(int\)(\d+), \(bool\)"
)
_NVTX_RE = re.compile(r"execute_context_\d+\((\d+)\)_generation_\d+\((\d+)\)")


# Borrow attribute_step's structure from the trace dir's helper, but
# (1) patch classify() to recognize prefill's conv1d_fwd anchor, and
# (2) ignore Marlin variant tags inside the attribution.
sys.path.insert(0, str(ATTRIBUTION_SRC.parent))
import _attribute_kernels as ak  # type: ignore  # noqa: E402


def classify_prefill(name: str):
    """Variant of ak.classify() that also flags prefill anchors."""
    if not name:
        return ("other", "")
    # Prefill's conv1d uses the "fwd" kernel; decode uses "update".
    # The original classify() only matches `update`.
    if "_causal_conv1d_fwd_kernel" in name:
        return ("conv1d_update", "")  # same family token, so attribute_step picks it up
    return ak.classify(name)


# Each main W4A16 GEMM has an "aux" marlin sidekick (typically a small
# pack/unpack kernel, ~10-80us).  Filter those out so we only attribute
# the real compute calls.  Threshold derived from observed bimodality
# at M=579: aux marlins are <80us, main marlins are >150us.
_AUX_MARLIN_THRESH_US = 100.0


def attribute_step_prefill(kernels):
    """Anchor-based attribution patched for the Nano3.5 *prefill* layer order.

    Layer pattern observed (single forward pass):

        Per ME-block (23 of them):
          mamba_in_proj  (main)              <- aux marlin
          conv1d_fwd     (M-anchor)
          [SSM kernels: chunk_state, state_passing, chunk_scan, ...]
          mamba_output_proj  (main)          <- aux marlin
          [optional A-block, 6 per step:
             qkv_linear  (main)              <- aux
             kv_cache_update
             fi_attn     (A-anchor)
             out_linear  (main)              <- aux
          ]
          cublas (route logits)
          shared_fc1  (main)                 <- aux
          shared_fc2  (main)                 <- aux
          moe_topk    (E-anchor)
          moe_align, moe_count_sort
          expert_fc1 (moe_marlin), expert_fc2 (moe_marlin)
          at_reduce

        + 1 proj_linear (lm_head) at the very end of the pass.

    Two key differences from the decode-era attribution:

    1. ``_causal_conv1d_fwd_kernel`` is the M-anchor (decode uses
       ``conv1d_update``).  Handled by ``classify_prefill``.
    2. Shared FCs happen BEFORE moe_topk (not after, as in decode).
       So we search for shared_fc1/fc2 in the BACKWARDS range
       (E-anchor → prev anchor), skipping aux marlins by duration.
    """
    fams = [classify_prefill(k["name"]) for k in kernels]
    n = len(kernels)
    anchors = []
    for i, (fam, _) in enumerate(fams):
        if fam == "conv1d_update":
            anchors.append((i, "M"))
        elif fam == "moe_topk":
            anchors.append((i, "E"))
        elif fam == "fi_attn":
            anchors.append((i, "A"))
    if not anchors:
        return [], [], set(), fams

    claimed = set()
    attributions = []

    def is_main_marlin(jj):
        return fams[jj][0] == "marlin" and kernels[jj]["dur"] / 1000.0 >= _AUX_MARLIN_THRESH_US

    def is_aux_marlin(jj):
        return fams[jj][0] == "marlin" and kernels[jj]["dur"] / 1000.0 < _AUX_MARLIN_THRESH_US

    def is_marlin(jj):
        return fams[jj][0] == "marlin"

    def is_moe_marlin(jj):
        return fams[jj][0] == "moe_marlin"

    def find_next(start, end, pred):
        step = 1 if end > start else -1
        for j in range(start, end, step):
            if j in claimed:
                continue
            if pred(j):
                return j
        return None

    def claim(j, label, atype, bi):
        if j is None or j in claimed:
            return
        claimed.add(j)
        k = kernels[j]
        attributions.append({**k, "dlsim_op": label, "block_type": atype,
                             "block_idx": bi, "kidx": j})

    for bi, (anc, atype) in enumerate(anchors):
        prev_anc = anchors[bi - 1][0] if bi > 0 else -1
        next_anc = anchors[bi + 1][0] if bi + 1 < len(anchors) else n

        if atype == "M":
            claim(anc, "mamba_qkv_conv", "M", bi)
            # mamba_in_proj = main marlin BEFORE anchor (skip aux).
            j = find_next(anc - 1, prev_anc, is_main_marlin)
            claim(j, "mamba_in_proj", "M", bi)
            # mamba_output_proj = first main marlin AFTER anchor.
            j = find_next(anc + 1, next_anc, is_main_marlin)
            claim(j, "mamba_output_proj", "M", bi)

        elif atype == "E":
            claim(anc, "moe_topk", "E", bi)
            # cublas route logits BEFORE anchor.
            j = find_next(anc - 1, prev_anc, lambda jj: fams[jj][0] == "cublas_gemv")
            claim(j, "moe_route", "E", bi)
            # moe_align / count_sort AFTER anchor.
            claim(find_next(anc + 1, next_anc, lambda jj: fams[jj][0] == "moe_align"),
                  "moe_align", "E", bi)
            claim(find_next(anc + 1, next_anc, lambda jj: fams[jj][0] == "moe_count_sort"),
                  "moe_count_sort", "E", bi)
            # Expert FCs (moe_marlin) AFTER anchor.
            fc1_j = find_next(anc + 1, next_anc, is_moe_marlin)
            claim(fc1_j, "expert0_fc1", "E", bi)
            fc2_j = find_next(anc + 1, next_anc, is_moe_marlin)
            claim(fc2_j, "expert0_fc2", "E", bi)
            # shared_fc1, shared_fc2 BEFORE the E-anchor.  Scan
            # BACKWARDS, skipping aux marlins.  shared_fc2 is the
            # CLOSEST main marlin going backwards, shared_fc1 is the
            # next one.  (M-anchor's mamba_out is even further back
            # and would be claimed by the M-block.)
            j2 = find_next(anc - 1, prev_anc, is_main_marlin)
            claim(j2, "shared_fc2", "E", bi)
            j1 = find_next((j2 - 1) if j2 is not None else (anc - 1),
                           prev_anc, is_main_marlin)
            claim(j1, "shared_fc1", "E", bi)
            # at_reduce.
            search_start = fc2_j + 1 if fc2_j is not None else anc + 1
            claim(find_next(search_start, next_anc, lambda jj: fams[jj][0] == "at_reduce"),
                  "moe_reduce", "E", bi)

        elif atype == "A":
            claim(anc, "self_attn_qk_matmul", "A", bi)
            kv_j = find_next(anc - 1, prev_anc, lambda jj: fams[jj][0] == "kv_cache_update")
            claim(kv_j, "kv_cache_update", "A", bi)
            fim_j = find_next(anc + 1, next_anc, lambda jj: fams[jj][0] == "fi_merge")
            claim(fim_j, "self_attn_qk_merge", "A", bi)
            # qkv_linear = main marlin BEFORE kv_cache_update.
            search_end = kv_j if kv_j is not None else anc
            claim(find_next(search_end - 1, prev_anc, is_main_marlin),
                  "self_attn_qkv_linear", "A", bi)
            # out_linear = first main marlin AFTER fi_attn anchor.
            search_start = fim_j + 1 if fim_j is not None else anc + 1
            claim(find_next(search_start, next_anc, is_main_marlin),
                  "self_attn_out_linear", "A", bi)

    # Tail: lm_head = largest remaining unclaimed main marlin.
    unclaimed = [(j, kernels[j]) for j in range(n)
                 if is_main_marlin(j) and j not in claimed]
    if unclaimed:
        lm_j, _ = max(unclaimed, key=lambda x: x[1]["dur"])
        claim(lm_j, "proj_linear (lm_head)", "tail", -1)

    return anchors, attributions, claimed, fams


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1, int(p * len(sorted_vals))))
    return sorted_vals[k]


def parse_variant(name: str):
    m = _VARIANT_RE.search(name or "")
    if not m:
        return None
    return f"{m.group(1)}<{m.group(2)},{m.group(3)},{m.group(4)},{m.group(5)}>"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE)
    cur = conn.cursor()

    cur.execute(
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE text LIKE 'execute_context%' "
        "AND text NOT LIKE '%(0)_generation%' "
        "ORDER BY start"
    )
    iters = []
    for text, start, end in cur.fetchall():
        m = _NVTX_RE.match(text)
        if not m:
            continue
        p_m = int(m.group(1))
        if p_m == 0:
            continue
        iters.append({"text": text, "start": start, "end": end, "prefill_M": p_m})

    print(f"Found {len(iters)} prefill iterations.")
    for it in iters:
        print(f"  {it['text']}  dur={(it['end']-it['start'])/1e6:.2f}ms")

    # Per-call attribution (op view).
    per_op_rows = []
    # Per-(iter, variant) summary (variant view).
    per_variant_rows = []
    # Per-iteration summary (chunks, anchor count).
    per_iter_summary = []

    for it in iters:
        cur.execute(
            """SELECT k.start, k.end - k.start AS dur_ns, ss.value, k.end
               FROM CUPTI_ACTIVITY_KIND_RUNTIME r
               JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON r.correlationId = k.correlationId
               JOIN StringIds ss ON ss.id = k.demangledName
               WHERE r.start >= ? AND r.start <= ?
               ORDER BY r.start""",
            (it["start"], it["end"]),
        )
        rows = cur.fetchall()
        kernels = [{"k_start": r[0], "dur": r[1], "name": r[2] or "",
                    "k_end": r[3]} for r in rows]
        anchors, attributions, _, fams = attribute_step_prefill(kernels)

        per_op_count = Counter(a["dlsim_op"] for a in attributions)
        chunks = max(
            (per_op_count.get(op, 0) // per_step)
            for op, per_step in PER_STEP_CALL_COUNT.items()
            if per_step > 0
        ) or 1
        chunk_M = it["prefill_M"] // chunks if chunks > 0 else it["prefill_M"]

        per_iter_summary.append({
            "iter": it["text"],
            "prefill_M": it["prefill_M"],
            "chunks_inferred": chunks,
            "chunk_M": chunk_M,
            "anchors": Counter(t for _, t in anchors),
            "per_op_count": dict(per_op_count),
        })

        for a in attributions:
            per_op_rows.append({
                "iter": it["text"],
                "prefill_M": it["prefill_M"],
                "chunk_M": chunk_M,
                "dlsim_op": a["dlsim_op"],
                "dur_us": a["dur"] / 1000.0,
                "variant": parse_variant(a["name"]) or "",
            })

        # Per-variant view (independent of attribution).
        by_variant = defaultdict(list)
        for r in rows:
            if not r[2]:
                continue
            v = parse_variant(r[2])
            if v is None:
                continue
            by_variant[v].append(r[1] / 1000.0)
        for v, durs in sorted(by_variant.items(), key=lambda x: -len(x[1])):
            durs.sort()
            per_variant_rows.append({
                "iter": it["text"],
                "prefill_M": it["prefill_M"],
                "variant": v,
                "calls": len(durs),
                "min_us": durs[0],
                "p50_us": percentile(durs, 0.5),
                "p90_us": percentile(durs, 0.9),
                "max_us": durs[-1],
                "mean_us": sum(durs) / len(durs),
            })

    # Write per-op CSV.
    op_csv = OUT_DIR / "marlin_prefill_per_op.csv"
    with open(op_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_op_rows[0].keys()))
        w.writeheader()
        for r in per_op_rows:
            w.writerow(r)
    print(f"Wrote {op_csv} ({len(per_op_rows)} rows)")

    # Write per-variant CSV.
    var_csv = OUT_DIR / "marlin_prefill_per_variant.csv"
    with open(var_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_variant_rows[0].keys()))
        w.writeheader()
        for r in per_variant_rows:
            w.writerow(r)
    print(f"Wrote {var_csv} ({len(per_variant_rows)} rows)")

    # Aggregate per (op, chunk_M) and emit the headline summary.
    grouped = defaultdict(list)
    for r in per_op_rows:
        grouped[(r["dlsim_op"], r["chunk_M"])].append(r["dur_us"])

    single_chunk_Ms = [579, 1521, 1823, 2001, 2015]
    md = ["# Marlin baseline — Nano3.5 prefill\n"]
    md.append(f"Source trace: `{SQLITE}`")
    md.append(f"GPU: NVIDIA GB10 (Spark, SM121).  Memory BW ≈ 273 GB/s.\n")
    md.append("## Per-dlsim_op p50 µs across the single-chunk prefill M ladder\n")
    md.append("Single-chunk iterations (chunk_M = NVTX prefill_M).  Aggregated "
              "from anchor-based attribution.  Variant-agnostic — relies on "
              "kernel order relative to conv1d_fwd / moe_topk / fi_attn anchors.\n")
    md.append("| dlsim_op | K × N | calls/step | M=579 | M=1521 | M=1823 | M=2001 | M=2015 |")
    md.append("|---|---|---|---|---|---|---|---|")

    dense_ops = [
        "mamba_in_proj", "mamba_output_proj",
        "shared_fc1", "shared_fc2",
        "self_attn_qkv_linear", "self_attn_out_linear",
        "proj_linear (lm_head)",
    ]
    op_summary_json = {}
    for op in dense_ops:
        shape = OP_SHAPE.get(op, (None, None))
        per_step = PER_STEP_CALL_COUNT.get(op, "—")
        cells = []
        for M in single_chunk_Ms:
            durs = grouped.get((op, M), [])
            if not durs:
                cells.append("—")
            else:
                durs_s = sorted(durs)
                p50 = durs_s[len(durs_s) // 2]
                cells.append(f"{p50:.0f}µs (n={len(durs)})")
                op_summary_json.setdefault(op, {})[M] = {
                    "p50_us": p50,
                    "p90_us": percentile(durs_s, 0.9),
                    "calls": len(durs),
                    "K": shape[0], "N": shape[1],
                }
        shape_s = f"{shape[0]} × {shape[1]}" if shape[0] else "—"
        md.append(f"| **{op}** | {shape_s} | {per_step} | " + " | ".join(cells) + " |")

    md.append("\n## Caveats\n")
    md.append("**M=1521 looks anomalous** and is likely a multi-chunk iteration "
              "even though it has the canonical 23 M-anchor count.  Its "
              "per-variant call counts are roughly 2× the other single-chunk "
              "iters (150 vs 75 for `<128,4,8,4>`, 69 vs 23 for `<128,4,4,8>`) "
              "and mamba_in_proj p50 (388µs) is lower than M=579 (564µs), "
              "which is impossible on the same shape.  Treat M=1521 numbers "
              "above with skepticism; use the M=579/1823/2001/2015 columns "
              "for the headline comparison against b12x.\n")
    md.append("**Cumulative-M iters (31920, 34048, 36176)** all have 23 "
              "M-anchors → single forward pass per NVTX iter.  The "
              "prefill_M label likely reflects total cumulative context, "
              "not the actual per-call M.  Inferred per-call M ≈ 2048 from "
              "the kernel durations (mamba_in_proj p50 ≈ 860µs, "
              "very close to the M=2015 value of 810µs).\n")
    md.append("**Marlin SOL utilization at these shapes is ~40-50%** of "
              "compute peak — consistent with the design doc's claim that "
              "the prefill regime is the clear b12x v5 opportunity.\n")
    md.append("## Per-iteration anchor sanity check\n")
    md.append("Expected per single-chunk iter: 23 M-anchors, 23 E-anchors, "
              "6 A-anchors.  Deviation flags attribution issues.\n")
    md.append("| iter | prefill_M | chunks (inferred) | M-anchors | E-anchors | A-anchors |")
    md.append("|---|---|---|---|---|---|")
    for it in per_iter_summary:
        a = it["anchors"]
        md.append(f"| `{it['iter']}` | {it['prefill_M']} | {it['chunks_inferred']} | "
                  f"{a.get('M', 0)} | {a.get('E', 0)} | {a.get('A', 0)} |")

    summary_md = OUT_DIR / "marlin_prefill_summary.md"
    summary_md.write_text("\n".join(md))
    print(f"Wrote {summary_md}")

    summary_json = OUT_DIR / "marlin_prefill_summary.json"
    with open(summary_json, "w") as f:
        json.dump({
            "by_op_by_M": op_summary_json,
            "per_iter": [
                {**it, "anchors": dict(it["anchors"])} for it in per_iter_summary
            ],
            "source_trace": SQLITE,
        }, f, indent=2)
    print(f"Wrote {summary_json}")

    # Stdout digest.
    print()
    print("Per-op p50 µs across single-chunk M ladder")
    print("-" * 100)
    hdr = f"{'op':28s} {'K':>5} {'N':>6} {'/step':>6} " + " ".join(f"{f'M={M}':>11}" for M in single_chunk_Ms)
    print(hdr)
    print("-" * len(hdr))
    for op in dense_ops:
        shape = OP_SHAPE.get(op, (None, None))
        per_step = PER_STEP_CALL_COUNT.get(op, 0)
        cells = []
        for M in single_chunk_Ms:
            durs = grouped.get((op, M), [])
            if durs:
                durs_s = sorted(durs)
                p50 = durs_s[len(durs_s) // 2]
                cells.append(f"{p50:>8.1f}us")
            else:
                cells.append("        —")
        k_s = f"{shape[0]:5d}" if shape[0] else "    -"
        n_s = f"{shape[1]:6d}" if shape[1] else "     -"
        print(f"{op:28s} {k_s} {n_s} {per_step:>6} " + " ".join(c.rjust(11) for c in cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
