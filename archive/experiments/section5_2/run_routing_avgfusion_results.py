# Matched routing under AVERAGE FUSION in training -- diagnostic readout.
#
# The Product Fusion comparison is RECOMPUTED here from its stored runs rather
# than quoted, so both rules go through one code path and any disagreement with
# the previously reported figures is visible rather than hidden.
#
# Averaged readout throughout. Reference band for 90% coverage: hw:RMS in
# [1.65, 1.87] (the empirical crossing, not the Gaussian 1.645).
#
# Run: python -u experiments/section5_2/run_routing_avgfusion_results.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from averaged_readout import load_run
from calibration_band import BAND

LAST, MIN_CELLS = 50, 200
ROOTS = [("Product fusion", "runs/chapter5_v2/s5_9_routing_matched"),
         ("Average fusion", "runs/chapter5_v2/s5_9_routing_matched_avgfusion")]
ARMS = ["random", "uncertainty_guided"]
LABEL = {"random": "random", "uncertainty_guided": "guided"}


def stat(R, mask=None):
    e2, hw = R["averaged"]
    T = R["T"]
    sl = slice(T - LAST, T)
    ok = np.isfinite(e2[sl]) & np.isfinite(hw[sl])
    if mask is not None:
        ok &= mask[sl]
    if ok.sum() < MIN_CELLS:
        return None
    m = float(np.mean(e2[sl][ok]))
    return dict(n=int(ok.sum()), whole=float(np.nanmean(e2)), last50=m,
                cov=float((np.sqrt(e2[sl][ok]) <= hw[sl][ok]).mean()),
                ratio=float(np.mean(hw[sl][ok]) / np.sqrt(m)))


def load(root, arm):
    out = []
    for p in sorted(glob.glob("%s/%s_seed*/manifest.yaml" % (root, arm))):
        d = Path(os.path.dirname(p))
        R = load_run(d, int(yaml.safe_load(open(p))["env_seed"]))
        R["hop"] = np.load(d / "cell_hop_steps.npy")
        out.append(R)
    return out


ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

D = {}
for rule, root in ROOTS:
    for arm in ARMS:
        runs = load(root, arm)
        if runs:
            D[(rule, arm)] = runs

print("=" * 104)
print("MATCHED ROUTING: PRODUCT vs AVERAGE FUSION IN TRAINING")
print("averaged readout, last-%d window, 5 seeds, mean +/- sd" % LAST)
print("=" * 104)
print("%-16s %-9s %3s %-20s %-20s %-15s %-15s"
      % ("fusion in training", "routing", "n", "whole-run MSE", "last-50 MSE",
         "90% coverage", "hw : RMS"))
AGG = {}
for rule, _ in ROOTS:
    for arm in ARMS:
        runs = D.get((rule, arm))
        if not runs:
            continue
        s = [stat(R) for R in runs]
        f = lambda k: ms([x[k] for x in s])
        AGG[(rule, arm)] = {k: f(k) for k in ("whole", "last50", "cov", "ratio")}
        g = lambda k, p=5: "%.*f +/- %.*f" % (p, f(k)[0], p, f(k)[1])
        print("%-16s %-9s %3d %-20s %-20s %-15s %-15s"
              % (rule, LABEL[arm], len(runs), g("whole"), g("last50"),
                 g("cov", 3), g("ratio", 3)))

print()
print("ROUTING EFFECT (guided minus random), same rule:")
print("%-16s %14s %14s %14s %14s" % ("fusion", "whole-run", "last-50",
                                     "coverage", "hw:RMS"))
for rule, _ in ROOTS:
    if (rule, "random") not in AGG or (rule, "uncertainty_guided") not in AGG:
        continue
    r, g = AGG[(rule, "random")], AGG[(rule, "uncertainty_guided")]
    d = lambda k: g[k][0] - r[k][0]
    print("%-16s %+14.5f %+14.5f %+14.3f %+14.3f"
          % (rule, d("whole"), d("last50"), d("cov"), d("ratio")))
print("  reference band for 90%% coverage: hw:RMS in [%.2f, %.2f]" % BAND)

print()
print("=" * 104)
print("PER-HOP hw:RMS  ('--' = fewer than %d cell-steps)" % MIN_CELLS)
print("=" * 104)
hops = sorted({int(h) for runs in D.values() for R in runs
               for h in np.unique(R["hop"][np.isfinite(R["hop"])])})
print("%-16s %-9s" % ("fusion", "routing")
      + "".join("%11s" % ("hop %d" % h) for h in hops) + "%12s" % "0->max span")
for rule, _ in ROOTS:
    for arm in ARMS:
        runs = D.get((rule, arm))
        if not runs:
            continue
        row, vals = "%-16s %-9s" % (rule, LABEL[arm]), {}
        for h in hops:
            v = [stat(R, R["hop"] == h) for R in runs]
            v = [x["ratio"] for x in v if x]
            if v:
                vals[h] = st.mean(v)
                row += "%11.2f" % vals[h]
            else:
                row += "%11s" % "--"
        row += "%12s" % ("%.2f" % (max(vals.values()) - min(vals.values()))
                         if len(vals) > 1 else "--")
        print(row)
print()
print("  span is the hop gradient: max minus min across the hop bins above.")
print("  Average fusion WITHOUT routing was reported as 3.89 / 3.06 / 2.52.")
