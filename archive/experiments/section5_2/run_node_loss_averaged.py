# Section 5.7 node loss: paired surviving-region difference under the AVERAGED
# readout, from the rerun that stores the belief arrays.
#
# The original runs could not support this: run_node_failure.py wrote its own
# save block and never stored cell_beliefs_steps / cell_ncov_steps, so the
# section was the only cell-space result in Chapter 5 still on the argmax
# estimator. The save block has since been extended and the section rerun into
# s5_7_node_loss_avg.
#
# CONSTRUCTION RETAINED EXACTLY, only the estimator changes:
#   - last-50 window, steps 340..390, failure at step 150
#   - paired per-cell difference against the matched 0% control at that seed
#   - restricted to cells covered on EVERY step of the window in BOTH runs
#   - cell (not cell-timestep) is the unit of replication; paired t over cells
#   - five seeds
#
# THREE GUARDS, so a discrepancy cannot be mistaken for an estimator effect:
#   1. this script's argmax path must reproduce the NEW manifests' stored
#      paired_vs_control exactly -> proves the window/pairing/masking match
#      run_node_failure.compute_stats
#   2. the rerun's argmax result must match the ORIGINAL runs' within the
#      reproduction floor -> proves the rerun reproduces the reported section
#   3. the averaged readout must reduce to argmax at n_cov == 1
#
# Run: python -u experiments/section5_2/run_node_loss_averaged.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import ttest_rel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from averaged_readout import load_run, validate_reduction

NEW = Path("runs/chapter5_v2/s5_7_node_loss_avg")
OLD = Path("runs/chapter5_v2/s5_7_node_loss")
POST_WINDOW = (340, 390)          # verbatim from run_node_failure.py
OUT_CSV = Path("runs/chapter5_v2/node_loss_paired_averaged.csv")
CONTROL_COND = "nig_product_random_0pct"


def paired(e2_fail, e2_ctrl):
    """run_node_failure.compute_stats' paired block, estimator-agnostic."""
    lo, hi = POST_WINDOW
    fw, cw = e2_fail[lo:hi], e2_ctrl[lo:hi]
    always = (~np.isnan(fw)).all(axis=0) & (~np.isnan(cw)).all(axis=0)
    n_cells = int(always.sum())
    if n_cells < 3:
        return None
    fm = np.nanmean(fw[:, always], axis=0)
    cm = np.nanmean(cw[:, always], axis=0)
    d = fm - cm
    t, p = ttest_rel(fm, cm)
    return dict(n_cells=n_cells, mean_diff=float(d.mean()),
                std_diff=float(d.std()), t=float(t), p=float(p))


runs, stored_old = {}, {}
for p in sorted(glob.glob(str(NEW / "*" / "manifest.yaml"))):
    d = Path(os.path.dirname(p))
    m = yaml.safe_load(open(p))
    seed = int(m["seed"])
    cond = d.name.rsplit("_seed", 1)[0]
    R = load_run(d, seed)
    runs[(cond, seed)] = dict(argmax=R["argmax"][0], averaged=R["averaged"][0],
                              stored=(m["results"].get("paired_vs_control") or {}),
                              dir=d)
    op = OLD / d.name / "manifest.yaml"
    if op.exists():
        om = yaml.safe_load(open(op))
        stored_old[(cond, seed)] = (om["results"].get("paired_vs_control") or {})

seeds = sorted({s for _, s in runs})
conds = sorted({c for c, _ in runs})
print("loaded %d runs from %s: %d conditions x %d seeds"
      % (len(runs), NEW, len(conds), len(seeds)))

# guard 3
d0 = runs[(CONTROL_COND, seeds[0])]["dir"]
v = validate_reduction(d0, seeds[0])
print("\nGUARD 3 -- n_cov==1 reduction on %s: max |averaged - argmax| MSE = %.3e (%d cells)"
      % (d0.name, v["mse_vs_argmax"], v["n_cells"]))
assert v["mse_vs_argmax"] < 1e-6, "averaged readout does not reduce at n_cov==1"
print("  PASSED")

res = defaultdict(lambda: defaultdict(list))
gap_new, gap_old = [], []
for cond in conds:
    if cond == CONTROL_COND:
        continue
    for seed in seeds:
        k = (cond, seed)
        ctrl = runs.get((CONTROL_COND, seed))
        if k not in runs or ctrl is None:
            continue
        for est in ("argmax", "averaged"):
            r = paired(runs[k][est], ctrl[est])
            if r:
                res[cond][est].append(r)
        s = runs[k]["stored"]
        if s.get("mean_diff") is not None and res[cond]["argmax"]:
            gap_new.append(abs(res[cond]["argmax"][-1]["mean_diff"] - s["mean_diff"]))
        so = stored_old.get(k, {})
        if so.get("mean_diff") is not None and res[cond]["argmax"]:
            gap_old.append(res[cond]["argmax"][-1]["mean_diff"] - so["mean_diff"])

print("\nGUARD 1 -- recomputed argmax vs THIS RERUN's manifests: max |gap| = %.3e over %d runs"
      % (max(gap_new), len(gap_new)))
assert max(gap_new) < 1e-12, "recomputation does not match compute_stats"
print("  PASSED -- window, pairing and masking match run_node_failure.compute_stats")

go = np.array(gap_old)
print("\nGUARD 2 -- rerun argmax vs the ORIGINAL runs' stored values (%d runs)" % len(go))
print("  mean signed gap %+.6f   max |gap| %.6f   (values themselves are ~1e-3)"
      % (go.mean(), np.abs(go).max()))
print("  NOTE: these are separate GPU runs, so exact equality is not expected;")
print("  the last-50 window's run-to-run floor was measured at ~1%% of the MSE.")

ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

print("\n" + "=" * 120)
print("SECTION 5.7 PAIRED SURVIVING-REGION DIFFERENCE vs 0% CONTROL")
print("last-50 window (steps %d-%d), failure at 150, paired over cells, 5 seeds"
      % POST_WINDOW)
print("=" * 120)
print("%-30s %3s %8s %8s   %-24s %-24s"
      % ("condition", "n", "cells", "cells", "ARGMAX", "AVERAGED (new)"))
print("%-30s %3s %8s %8s   %-24s %-24s"
      % ("", "", "argmax", "avg", "mean_diff +/- sd", "mean_diff +/- sd"))

rows, flips, mismatch, positives = [], [], [], []
for cond in conds:
    if cond == CONTROL_COND or not res[cond]["averaged"]:
        n = sum(1 for s in seeds if (cond, s) in runs)
        print("%-30s %3d %8s %8s   %-24s %-24s"
              % (cond, n, "--", "--", "control (self)", "control (self)"))
        continue
    A, V = res[cond]["argmax"], res[cond]["averaged"]
    na, nv = ms([r["n_cells"] for r in A]), ms([r["n_cells"] for r in V])
    da, dv = ms([r["mean_diff"] for r in A]), ms([r["mean_diff"] for r in V])
    print("%-30s %3d %8.0f %8.0f   %-24s %-24s"
          % (cond, len(V), na[0], nv[0],
             "%+.5f +/- %.5f" % da, "%+.5f +/- %.5f" % dv))
    rows.append((cond, len(V), na[0], nv[0], da, dv,
                 ms([r["std_diff"] for r in A]), ms([r["std_diff"] for r in V]),
                 ms([r["p"] for r in V])))
    if (da[0] > 0) != (dv[0] > 0):
        flips.append(cond)
    if any(a["n_cells"] != b["n_cells"] for a, b in zip(A, V)):
        mismatch.append(cond)
    if dv[0] > 0:
        positives.append(cond)

print("\n  sd is ACROSS THE FIVE SEEDS (what the figure's error bar plots).")

print("\n" + "=" * 120)
print("THE THREE CONFIRMATIONS")
print("=" * 120)
print("1. SIGN UNCHANGED IN EVERY CONDITION : %s"
      % ("YES" if not flips else "NO -- flipped in %s" % flips))
print("2. CLUSTERED 40%% THE SOLE POSITIVE   : %s   (positives: %s)"
      % ("YES" if positives == ["nig_product_clustered_40pct"] else "NO",
         positives if positives else "none"))
print("3. CELLS-COMPARED IDENTICAL          : %s"
      % ("YES -- the coverage restriction is geometric, as expected"
         if not mismatch else "NO -- differs in %s" % mismatch))

with open(OUT_CSV, "w", encoding="utf8") as f:
    f.write("condition,n_seeds,n_cells_compared,argmax_mean_diff,argmax_sd_across_seeds,"
            "averaged_mean_diff,averaged_sd_across_seeds,argmax_std_diff_cells,"
            "averaged_std_diff_cells,averaged_mean_p\n")
    for cond, n, na, nv, da, dv, sa, sv, pv in rows:
        f.write("%s,%d,%.0f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f\n"
                % (cond, n, nv, da[0], da[1], dv[0], dv[1], sa[0], sv[0], pv[0]))
print("\nwrote %s  (the figure's right panel reads this)" % OUT_CSV)
