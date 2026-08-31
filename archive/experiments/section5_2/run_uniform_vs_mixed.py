# Section 5.6 -- heterogeneous mesh against the three UNIFORM arms.
#
# Readout only over stored arrays. No runs, no training path touched.
#
# The question: does mixed capacity beat the BEST uniform configuration, or
# only the middling one? All four arms share the aggregation-comparator
# configuration -- nig_product, dynamic offset world, 36 nodes at stride 3,
# lam 5.0, lr 3e-5, exclusions applied, sampled targets, five seeds.
#
# Cell space, argmax estimator (Chapter 5's reported convention).
#
# Run: python -u experiments/section5_2/run_uniform_vs_mixed.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse_estimators import LAST

ARMS = [
    ("heterogeneous", "runs/chapter5_v2/s5_6_het/het_nig_product*/manifest.yaml"),
    ("uniform narrow", "runs/chapter5_v2/s5_6_homog_narrow/**/manifest.yaml"),
    ("uniform baseline", "runs/chapter5_v2/s5_6_homog_baseline/**/manifest.yaml"),
    ("uniform wide", "runs/chapter5_v2/s5_6_homog_wide/**/manifest.yaml"),
]


def metrics(d: Path):
    mse = np.load(d / "cell_mse_steps.npy")
    nigp = d / "cell_nig_steps.npy"
    T = mse.shape[0]
    sl = slice(T - LAST, T)
    out = dict(whole=float(np.nanmean(mse)), last50=float(np.nanmean(mse[sl])))
    if not nigp.exists():
        return out
    nig = np.load(nigp)
    nu, al, be = (nig[sl, ..., i] for i in range(3))
    e = np.sqrt(mse[sl])
    ok = np.isfinite(e) & np.isfinite(nu) & (al > 1.0)
    if ok.sum() < 8:
        return out
    sc = np.sqrt(be[ok] * (1 + nu[ok]) / (nu[ok] * al[ok]))
    hw = sps.t.ppf(0.95, df=2 * al[ok]) * sc
    out["cov"] = float((e[ok] <= hw).mean())
    out["ratio"] = float(np.mean(hw) * 180 / (np.sqrt(out["last50"]) * 180))
    return out


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def main():
    print("=" * 100)
    print("SECTION 5.6 -- MIXED CAPACITY vs UNIFORM CAPACITY")
    print("nig_product, cell space, argmax estimator, 5 seeds, mean +/- sd")
    print("=" * 100)
    rows, ref = [], None
    for name, pat in ARMS:
        ds = sorted({os.path.dirname(p) for p in glob.glob(pat, recursive=True)})
        if not ds:
            rows.append((name, 0, None))
            continue
        R = [metrics(Path(d)) for d in ds]
        agg = {k: ms([r[k] for r in R if k in r])
               for k in ("whole", "last50", "cov", "ratio")
               if any(k in r for r in R)}
        rows.append((name, len(R), agg))
        if name == "heterogeneous":
            ref = agg

    print("%-18s %3s %-21s %-21s %-15s %-15s %11s"
          % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage",
             "hw : RMS", "vs mixed"))
    for name, n, a in rows:
        if a is None:
            print("%-18s %3s %s" % (name, "-", "NOT YET AVAILABLE (run pending)"))
            continue
        f = lambda k, p=5: ("%.*f +/- %.*f" % (p, a[k][0], p, a[k][1])
                            if k in a else "not computable")
        pct = (100 * (a["whole"][0] - ref["whole"][0]) / ref["whole"][0]
               if ref and name != "heterogeneous" else 0.0)
        print("%-18s %3d %-21s %-21s %-15s %-15s %+10.2f%%"
              % (name, n, f("whole"), f("last50"), f("cov", 3), f("ratio", 3), pct))
    print("\n  'vs mixed' is whole-run MSE relative to the heterogeneous arm;")
    print("  positive means the uniform arm is WORSE than the mixture.")
    # A partial arm must NOT drive the verdict: an arm with fewer than all
    # five seeds is a different estimate, not a smaller one.
    NSEEDS = 5
    done = [r for r in rows if r[2] is not None and r[1] >= NSEEDS]
    partial = [r for r in rows if r[2] is not None and r[1] < NSEEDS]
    for name, n, _ in partial:
        print("  (%s has %d/%d seeds -- excluded from the verdict)" % (name, n, NSEEDS))
    if len(done) == len(ARMS):
        best = min((r for r in done if r[0] != "heterogeneous"),
                   key=lambda r: r[2]["whole"][0])
        het = [r for r in done if r[0] == "heterogeneous"][0]
        print("\n  best uniform arm: %s (whole-run %.5f)" % (best[0], best[2]["whole"][0]))
        print("  heterogeneous   : %.5f" % het[2]["whole"][0])
        print("  -> the mixture %s the best uniform configuration"
              % ("BEATS" if het[2]["whole"][0] < best[2]["whole"][0] else "does NOT beat"))
    else:
        print("\n  INCOMPLETE: %d of %d arms available; verdict withheld."
              % (len(done), len(ARMS)))


if __name__ == "__main__":
    main()
