# Chapter 5 readout-convention change: the two carry-throughs that do not
# depend on the batch-4 regeneration.
#
#   1. Estimator comparison, now ARGMAX vs AVERAGED (averaged is the
#      convention; argmax is what it displaces). Does the monotone gain with
#      contributor count survive, and how big is it at n_cov = 9?
#   2. Section 5.4.3 hop and n_cov calibration breakdowns under averaged.
#      Do the two central findings survive --
#        (a) hop dominates contributor count by roughly 1.9x
#        (b) hop-0 cells never fall below hw:RMS 1.0
#
# Readout only over stored beliefs. No training path touched.
#
# Run: python -u experiments/section5_2/run_carrythroughs.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from averaged_readout import load_run

LAST = 50
MIN_CELLS = 40


def stat(R, arm, mask):
    e2, hw = R[arm]
    T = R["T"]
    sl = slice(T - LAST, T)
    ok = np.isfinite(e2[sl]) & np.isfinite(hw[sl]) & mask[sl]
    if ok.sum() < MIN_CELLS:
        return None
    m = float(np.mean(e2[sl][ok]))
    return dict(n=int(ok.sum()), mse=m,
                cov=float((np.sqrt(e2[sl][ok]) <= hw[sl][ok]).mean()),
                ratio=float(np.mean(hw[sl][ok]) * 180 / (np.sqrt(m) * 180)))


def load531():
    out = []
    for p in sorted(glob.glob(
            "runs/chapter5_v2/s531_b2/nig_product_sampled_seed*/manifest.yaml")):
        d = Path(os.path.dirname(p))
        seed = int(yaml.safe_load(open(p))["env_seed"])
        R = load_run(d, seed)
        R["hop"] = np.load(d / "cell_hop_steps.npy")
        out.append(R)
    return out


def carry1(runs):
    print("=" * 98)
    print("CARRY-THROUGH 1 -- ESTIMATOR COMPARISON: argmax vs averaged (5.3.1, 5 seeds)")
    print("=" * 98)
    print("%6s %9s %14s %14s %12s %12s %12s"
          % ("n_cov", "cells", "MSE argmax", "MSE averaged", "gain %",
             "hw:RMS argmax", "hw:RMS avg"))
    ks = sorted({int(k) for R in runs for k in np.unique(R["nc"]) if k > 0})
    gains = []
    for k in ks:
        a, v, cells = [], [], 0
        ra, rv = [], []
        for R in runs:
            m = (R["nc"] == k)
            sa, sv = stat(R, "argmax", m), stat(R, "averaged", m)
            if sa and sv:
                a.append(sa["mse"]); v.append(sv["mse"]); cells += sa["n"]
                ra.append(sa["ratio"]); rv.append(sv["ratio"])
        if not a:
            continue
        ma, mv = st.mean(a), st.mean(v)
        g = 100 * (ma - mv) / ma
        gains.append((k, g))
        print("%6d %9d %14.5f %14.5f %+11.2f%% %12.3f %12.3f"
              % (k, cells // len(runs), ma, mv, g, st.mean(ra), st.mean(rv)))
    mono = all(gains[i][1] <= gains[i + 1][1] + 1e-9 for i in range(len(gains) - 1))
    print("\n  monotone in contributor count: %s" % mono)
    if len(gains) > 1:
        print("  gain at n_cov=1 (control): %+.2f%%   gain at n_cov=%d: %+.2f%%"
              % (gains[0][1], gains[-1][0], gains[-1][1]))
    return gains


def carry2(runs):
    print("\n" + "=" * 98)
    print("CARRY-THROUGH 2 -- 5.4.3 HOP vs n_cov UNDER THE AVERAGED CONVENTION")
    print("=" * 98)
    hops = sorted({int(h) for R in runs for h in np.unique(R["hop"][np.isfinite(R["hop"])])})
    ks = sorted({int(k) for R in runs for k in np.unique(R["nc"]) if k > 0})
    for arm in ("argmax", "averaged"):
        print("\n--- %s ---" % arm.upper())
        print("hw:RMS by (hop, n_cov); '--' = fewer than %d cells" % MIN_CELLS)
        print("%5s" % "hop" + "".join("%12s" % ("n_cov=%d" % k) for k in ks))
        grid = {}
        for h in hops:
            row = "%5d" % h
            for k in ks:
                v = [stat(R, arm, (R["hop"] == h) & (R["nc"] == k)) for R in runs]
                v = [x["ratio"] for x in v if x]
                if v:
                    grid[(h, k)] = st.mean(v)
                    row += "%12.2f" % st.mean(v)
                else:
                    row += "%12s" % "--"
            print(row)
        by_hop, by_nc = defaultdict(list), defaultdict(list)
        for (h, k), r in grid.items():
            by_hop[h].append(r); by_nc[k].append(r)
        w_nc = [max(v) - min(v) for v in by_hop.values() if len(v) > 1]
        w_hop = [max(v) - min(v) for v in by_nc.values() if len(v) > 1]
        if w_nc and w_hop:
            print("  spread across n_cov at fixed hop : mean %.3f" % st.mean(w_nc))
            print("  spread across hop at fixed n_cov : mean %.3f" % st.mean(w_hop))
            print("  RATIO (hop-driven / n_cov-driven): %.2f" % (st.mean(w_hop) / st.mean(w_nc)))
        h0 = [stat(R, arm, R["hop"] == 0) for R in runs]
        h0 = [x["ratio"] for x in h0 if x]
        h0k = {k: st.mean([x["ratio"] for x in
                           [stat(R, arm, (R["hop"] == 0) & (R["nc"] == k)) for R in runs]
                           if x]) for k in ks
               if any(stat(R, arm, (R["hop"] == 0) & (R["nc"] == k)) for R in runs)}
        print("  hop-0 aggregate hw:RMS: %.3f" % st.mean(h0))
        print("  hop-0 by n_cov: " + ", ".join("%d:%.2f" % (k, v) for k, v in sorted(h0k.items())))
        below = [k for k, v in h0k.items() if v < 1.0]
        print("  hop-0 n_cov bins BELOW 1.0: %s"
              % (sorted(below) if below else "none -- finding holds"))


if __name__ == "__main__":
    runs = load531()
    carry1(runs)
    carry2(runs)
