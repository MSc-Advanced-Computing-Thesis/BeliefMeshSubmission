# RECOMPUTATIONS 1 and 2 -- hop composition, and calibration recovery.
# Readout only over stored arrays. No training path touched, no new runs.
#
# Run: python -u experiments/section5_2/run_recomp_hop.py

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

HOPS = [0, 1, 2, 3]


def load_run(d: Path, window=True):
    L = lambda n: np.load(d / ("cell_%s_steps.npy" % n))
    mse, nig, hop = L("mse"), L("nig"), L("hop")
    if window:
        T = mse.shape[0]
        sl = slice(T - LAST, T)
        return mse[sl], nig[sl], hop[sl]
    return mse, nig, hop


def cal(mse, nig, mask):
    ok = np.isfinite(mse) & np.isfinite(nig[..., 0]) & (nig[..., 1] > 1.0) & mask
    if ok.sum() < 8:
        return None
    nu, al, be = (nig[..., i][ok] for i in range(3))
    sc = np.sqrt(be * (1 + nu) / (nu * al))
    hw = sps.t.ppf(0.95, df=2 * al) * sc
    m = float(np.mean(mse[ok]))
    return dict(n=int(ok.sum()), cov=float((np.sqrt(mse[ok]) <= hw).mean()),
                hw=float(np.mean(hw) * 180), rms=float(np.sqrt(m) * 180),
                ratio=float(np.mean(hw) * 180 / (np.sqrt(m) * 180)))


def geometries():
    geo = defaultdict(list)
    for p in sorted(glob.glob("runs/chapter5_v2/s5_4_1_overlap/*/manifest.yaml")):
        m = yaml.safe_load(open(p))
        geo[(m["mean_cell_coverage"], m["n_nodes"])].append(Path(os.path.dirname(p)))
    return dict(sorted(geo.items()))


def recomp1():
    print("=" * 92)
    print("RECOMPUTATION 1 -- HOP COMPOSITION PER GEOMETRY (last 50 steps, 5 seeds)")
    print("=" * 92)
    comp, per_hop, agg = {}, {}, {}
    for (mc, nn), ds in geometries().items():
        fr, ph = defaultdict(list), defaultdict(list)
        ag = []
        for d in ds:
            mse, nig, hop = load_run(d)
            fin = np.isfinite(hop)
            tot = fin.sum()
            for h in HOPS:
                fr[h].append(float((hop == h).sum()) / tot)
                r = cal(mse, nig, hop == h)
                if r and r["n"] >= 40:
                    ph[h].append(r)
            a = cal(mse, nig, np.ones_like(mse, bool))
            ag.append(a)
        comp[(mc, nn)] = {h: st.mean(fr[h]) for h in HOPS}
        per_hop[(mc, nn)] = {h: (st.mean([x["hw"] for x in ph[h]]),
                                 st.mean([x["rms"] for x in ph[h]]),
                                 st.mean([x["ratio"] for x in ph[h]]))
                             for h in HOPS if ph[h]}
        agg[(mc, nn)] = st.mean([x["ratio"] for x in ag])

    print("\nfraction of cells at each hop:")
    print("%9s %6s %9s %9s %9s %9s" % ("mean_cov", "nodes", "hop 0", "hop 1", "hop 2", "hop 3"))
    for (mc, nn), c in comp.items():
        print("%9.2f %6d %9.3f %9.3f %9.3f %9.3f"
              % (mc, nn, c[0], c[1], c[2], c[3]))

    print("\nper-hop hw:RMS within each geometry:")
    print("%9s %9s %9s %9s %9s | %9s" % ("mean_cov", "hop 0", "hop 1", "hop 2", "hop 3", "aggregate"))
    for k, ph in per_hop.items():
        row = "%9.2f" % k[0]
        for h in HOPS:
            row += "%9s" % ("%.2f" % ph[h][2] if h in ph else "--")
        print(row + " | %9.2f" % agg[k])

    # PREDICTION: take the 36-node mesh's per-hop hw and rms, apply each
    # geometry's OWN hop composition, and see whether that reproduces its
    # aggregate ratio. If it does, density and hop are one finding.
    refkey = min(per_hop, key=lambda k: abs(k[0] - 3.64))
    ref = per_hop[refkey]
    print("\nprediction: 36-node per-hop hw/RMS x each geometry's hop composition")
    print("%9s %12s %12s %10s" % ("mean_cov", "predicted", "actual", "error"))
    for k, c in comp.items():
        hs = [h for h in HOPS if h in ref and c[h] > 0]
        w = sum(c[h] for h in hs)
        pred_hw = sum(c[h] * ref[h][0] for h in hs) / w
        pred_rms = sum(c[h] * ref[h][1] for h in hs) / w
        p = pred_hw / pred_rms
        print("%9.2f %12.2f %12.2f %+10.2f" % (k[0], p, agg[k], p - agg[k]))
    print("\n  A small error means the geometry's aggregate calibration is explained")
    print("  by ITS HOP MIX, using per-hop behaviour measured in a single mesh --")
    print("  i.e. density and hop are one finding. A large error means density")
    print("  carries information beyond hop composition.")


def recomp2():
    print("\n" + "=" * 92)
    print("RECOMPUTATION 2 -- DOES CALIBRATION RECOVER AFTER RETURNING TO COVERAGE?")
    print("=" * 92)
    print("SUBSTITUTION: per-NODE hop is not stored -- cell_hop_steps records the hop")
    print("of the node whose belief WON each cell. Recency is therefore computed")
    print("per CELL: timesteps since that cell was last served by a hop-0 node.")
    print("This is the cell-space analogue and matches the reporting convention.\n")

    bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 6)]
    acc = defaultdict(list)
    recency_all = []
    for p in sorted(glob.glob("runs/chapter5_v2/s531_b2/nig_product_sampled_seed*/manifest.yaml")):
        d = Path(os.path.dirname(p))
        mse, nig, hop = load_run(d, window=False)      # need full history for recency
        T, G = hop.shape[0], hop.shape[1]
        rec = np.full(hop.shape, np.nan)
        last = np.full((G, G), -1)
        for t in range(T):
            at0 = (hop[t] == 0)
            last[at0] = t
            seen = last >= 0
            rec[t][seen] = t - last[seen]
        sl = slice(T - LAST, T)
        r, m2, n2 = rec[sl], mse[sl], nig[sl]
        recency_all.append(r[np.isfinite(r)])
        for lo, hi in bins:
            c = cal(m2, n2, np.isfinite(r) & (r >= lo) & (r <= hi))
            if c and c["n"] >= 40:
                acc[(lo, hi)].append(c)

    print("%16s %10s %10s %12s" % ("steps since hop 0", "hw:RMS", "cov90", "cells/step"))
    for lo, hi in bins:
        v = acc.get((lo, hi))
        if not v:
            continue
        lab = "%d" % lo if lo == hi else ("%d-%d" % (lo, hi) if hi < 10 ** 6 else "%d+" % lo)
        print("%16s %10.2f %10.3f %12d"
              % (lab, st.mean([x["ratio"] for x in v]), st.mean([x["cov"] for x in v]),
                 sum(x["n"] for x in v) // (LAST * len(v))))
    allr = np.concatenate(recency_all)
    print("\nrecency distribution over cells (last-50 window, all seeds):")
    for q in (10, 25, 50, 75, 90, 99):
        print("  p%-3d %6.1f steps" % (q, np.percentile(allr, q)))
    print("  mean %.1f, max %.0f" % (allr.mean(), allr.max()))


if __name__ == "__main__":
    recomp1()
    recomp2()
