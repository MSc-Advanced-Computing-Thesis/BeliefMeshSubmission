# DIAGNOSTIC A -- is the calibration error in the fusion ARITHMETIC or in the
# TRAINING SIGNAL?
#
# Readout only over stored arrays. No training path touched, no default changed.
#
# ACCOUNT UNDER TEST. The head learns how well its predictions matched its
# TARGETS, not how far they are from the TRUTH. In coverage the two coincide,
# and the n_cov=1 control confirms it (hw:RMS ~1.03). Trained on fused beliefs
# the target is another model's output -- smoother and more self-consistent
# than the world -- so the head correctly learns its target is predictable
# while that target is not the truth. Each hop compounds it.
#
# PREDICTION. Calibration error should track HOP DISTANCE (how far the training
# signal has drifted from truth) rather than CONTRIBUTOR COUNT (the fusion
# arithmetic). Both are recorded per cell per step, so they can be separated.
#
# The ARGMAX estimator is used throughout: it is Chapter 5's reported
# convention, and it is the right instrument here because it compares a single
# node's own interval against that same node's own error -- exactly the
# quantity the account is about.
#
# Run: python -u experiments/section5_2/run_diag_a_hop_vs_ncov.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_estimators import LAST

MIN_CELLS = 40          # per-seed bin floor; thinner bins are reported, not used


def load_run(d: Path):
    L = lambda n: np.load(d / ("cell_%s_steps.npy" % n))
    mse, nig, hop, nc = L("mse"), L("nig"), L("hop"), L("ncov")
    T = mse.shape[0]
    sl = slice(T - LAST, T)
    return mse[sl], nig[sl], hop[sl], nc[sl].astype(int)


def cal(mse, nig, mask):
    ok = np.isfinite(mse) & np.isfinite(nig[..., 0]) & (nig[..., 1] > 1.0) & mask
    n = int(ok.sum())
    if n < 8:
        return None
    nu, al, be = (nig[..., i][ok] for i in range(3))
    sc = np.sqrt(be * (1 + nu) / (nu * al))
    hw = sps.t.ppf(0.95, df=2 * al) * sc
    e = np.sqrt(mse[ok])
    m = float(np.mean(mse[ok]))
    return dict(n=n, cov=float((e <= hw).mean()),
                ratio=float(np.mean(hw) * 180 / (np.sqrt(m) * 180)))


def two_way(runs, title):
    """hw:RMS and cov90 for every (hop, n_cov) bin, with cell counts."""
    hops = sorted({int(h) for _, _, hop, _ in runs for h in np.unique(hop[np.isfinite(hop)])})
    ncs = sorted({int(k) for _, _, _, nc in runs for k in np.unique(nc) if k > 0})
    print("\n=== %s ===" % title)
    print("hw:RMS  (cov90)  [cells per step, summed over seeds]")
    head = "%5s" % "hop"
    for k in ncs:
        head += "%22s" % ("n_cov=%d" % k)
    print(head)
    grid = {}
    for h in hops:
        row = "%5d" % h
        for k in ncs:
            vals, cells = [], 0
            for mse, nig, hop, nc in runs:
                m = (hop == h) & (nc == k)
                r = cal(mse, nig, m)
                if r and r["n"] >= MIN_CELLS:
                    vals.append(r)
                    cells += r["n"]
            if vals:
                rr = st.mean([v["ratio"] for v in vals])
                cc = st.mean([v["cov"] for v in vals])
                grid[(h, k)] = (rr, cc, cells)
                row += "%22s" % ("%.2f (%.3f) [%d]" % (rr, cc, cells // LAST))
            else:
                row += "%22s" % "--"
        print(row)
    return grid, hops, ncs


def variation(grid, hops, ncs):
    """How much of the spread in hw:RMS is along hop, and how much along n_cov?

    Marginal ranges are computed on the bins that exist for BOTH factors, so
    the comparison is like-for-like rather than driven by different coverage
    of the two axes.
    """
    by_hop, by_nc = defaultdict(list), defaultdict(list)
    for (h, k), (r, _, _) in grid.items():
        by_hop[h].append(r)
        by_nc[k].append(r)
    # within-hop spread across n_cov, and within-n_cov spread across hop
    w_nc = [max(v) - min(v) for v in by_hop.values() if len(v) > 1]
    w_hop = [max(v) - min(v) for v in by_nc.values() if len(v) > 1]
    print("\n  spread of hw:RMS ACROSS n_cov at fixed hop : mean %.3f  (max %.3f)"
          % (st.mean(w_nc), max(w_nc)) if w_nc else "  n/a")
    print("  spread of hw:RMS ACROSS hop at fixed n_cov : mean %.3f  (max %.3f)"
          % (st.mean(w_hop), max(w_hop)) if w_hop else "  n/a")
    if w_nc and w_hop:
        print("  ratio (hop-driven / n_cov-driven) = %.2f" % (st.mean(w_hop) / st.mean(w_nc)))


def main():
    # ---------------- 5.3.1 ----------------
    runs = [load_run(Path(os.path.dirname(p)))
            for p in sorted(glob.glob("runs/chapter5_v2/s531_b2/nig_product_sampled_seed*/manifest.yaml"))]
    g, hops, ncs = two_way(runs, "5.3.1  nig_product_sampled, 36-node mesh, 5 seeds")
    variation(g, hops, ncs)

    print("\n--- HOP 0 ONLY, by n_cov (nodes trained on GROUND TRUTH) ---")
    print("%8s %10s %10s %10s" % ("n_cov", "hw:RMS", "cov90", "cells/step"))
    for k in ncs:
        vals, cells = [], 0
        for mse, nig, hop, nc in runs:
            r = cal(mse, nig, (hop == 0) & (nc == k))
            if r and r["n"] >= MIN_CELLS:
                vals.append(r)
                cells += r["n"]
        if vals:
            print("%8d %10.2f %10.3f %10d"
                  % (k, st.mean([v["ratio"] for v in vals]),
                     st.mean([v["cov"] for v in vals]), cells // LAST))
    print("  If these hold near 1.0 regardless of n_cov, the fusion arithmetic is")
    print("  NOT implicated independently of the training signal.")

    # ---------------- 5.4.1 density sweep ----------------
    geo = defaultdict(list)
    for p in sorted(glob.glob("runs/chapter5_v2/s5_4_1_overlap/*/manifest.yaml")):
        d = Path(os.path.dirname(p))
        m = yaml.safe_load(open(p))
        geo[(m["mean_cell_coverage"], m["n_nodes"])].append(load_run(d))
    for (mc, nn), rs in sorted(geo.items()):
        g2, h2, k2 = two_way(rs, "5.4.1  mean coverage %.2f  (%d nodes), 5 seeds" % (mc, nn))
        variation(g2, h2, k2)


if __name__ == "__main__":
    main()
