# Averaged NIG: why the n_cov gradient and the density gradient run opposite
# ways.  Readout only over stored beliefs.  No runs, no training path touched.
#
# The two axes are nearly the same quantity, so they are put on one grid here:
# hw:RMS by (geometry, n_cov).  If density acted only through n_cov, every
# column would be flat.
#
# beta* under the averaged rule is
#     beta* = mean(beta_i)  +  0.5 * ( mean(nu_i x_i^2) - nu* x*^2 )
#             ^ own-belief     ^ SPREAD term, the contributor disagreement
# so both components are reported per geometry and per n_cov, to see which one
# carries which gradient.
#
# Run: python -u experiments/section5_2/run_averaged_two_axes.py

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from analyse_estimators import LAST, truth_for
from run_averaged_nig import averaged, hw_of, wrap_np

NCOVS = [1, 2, 3, 4, 6, 8, 9, 12, 16]


def load(d, seed):
    bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(d / "cell_ncov_steps.npy").astype(int)
    mse = np.load(d / "cell_mse_steps.npy")
    T, G = mse.shape[0], mse.shape[1]
    fin = np.isfinite(bel[..., 0])
    tr = truth_for(seed, T, G)
    gam, nu_s, al_s, be_s = averaged(bel, fin, nc)
    e2 = wrap_np(gam - tr) ** 2
    hw = hw_of(nu_s, al_s, be_s)

    g, nu, al, be = (bel[..., i] for i in range(4))
    # TWO versions of the offsets on purpose: zeros in the unfilled slots for
    # the beta* algebra (0 * nan would poison the sum), nan for the spread
    # statistic so unfilled slots are excluded from nanstd.
    x0 = np.where(fin, wrap_np(g - g[..., 0][..., None]), 0.0)
    x = np.where(fin, wrap_np(g - g[..., 0][..., None]), np.nan)
    N = np.maximum(nc, 1)[..., None].astype(float)
    w = np.where(fin, 1.0 / N, 0.0)
    mean_be = (w * np.where(fin, be, 0.0)).sum(-1)
    sq = (w * np.where(fin, nu, 0.0) * x0 * x0).sum(-1)
    x_s = ((w * np.where(fin, nu, 0.0) * x0).sum(-1)
           / np.maximum(nu_s, 1e-12))
    spread_term = 0.5 * (sq - nu_s * x_s * x_s)
    spread_deg = np.nanstd(x, axis=-1) * 180          # contributor gamma spread
    mean_nu = (w * np.where(fin, nu, 0.0)).sum(-1)

    sl = slice(T - LAST, T)
    return dict(e2=e2[sl], hw=hw[sl], nc=nc[sl], spread=spread_deg[sl],
                mean_be=mean_be[sl], spread_term=spread_term[sl],
                mean_nu=mean_nu[sl], be_s=be_s[sl])


def ratio(R, mask):
    ok = np.isfinite(R["e2"]) & np.isfinite(R["hw"]) & mask
    if ok.sum() < 40:
        return None
    m = float(np.mean(R["e2"][ok]))
    return float(np.mean(R["hw"][ok]) * 180 / (np.sqrt(m) * 180)), int(ok.sum())


def summarise(runs, label):
    allm = np.ones_like(runs[0]["nc"], bool)
    r = [ratio(R, np.ones_like(R["nc"], bool)) for R in runs]
    r = [x[0] for x in r if x]
    mn = st.mean([float(R["nc"].mean()) for R in runs])
    sp = st.mean([float(np.nanmean(R["spread"])) for R in runs])
    mb = st.mean([float(np.nanmean(R["mean_be"])) for R in runs])
    stm = st.mean([float(np.nanmean(R["spread_term"])) for R in runs])
    mnu = st.mean([float(np.nanmean(R["mean_nu"])) for R in runs])
    bs = st.mean([float(np.nanmean(R["be_s"])) for R in runs])
    return dict(label=label, ratio=st.mean(r), mean_ncov=mn, spread=sp,
                mean_be=mb, spread_term=stm, mean_nu=mnu, be_star=bs)


def main():
    groups = []
    for p in sorted(glob.glob("runs/chapter5_v2/s5_4_1_overlap/*/manifest.yaml")):
        m = yaml.safe_load(open(p))
        groups.append(((m["mean_cell_coverage"], m["n_nodes"]),
                       os.path.dirname(p), int(m["env_seed"])))
    geo = defaultdict(list)
    for k, d, s in groups:
        geo[k].append((d, s))
    d531 = [(os.path.dirname(p), yaml.safe_load(open(p))["env_seed"])
            for p in sorted(glob.glob(
                "runs/chapter5_v2/s531_b2/nig_product_sampled_seed*/manifest.yaml"))]

    R = {}
    for k, v in sorted(geo.items()):
        R[k] = [load(Path(d), s) for d, s in v]
    R531 = [load(Path(d), int(s)) for d, s in d531]

    print("=" * 104)
    print("AVERAGED NIG -- THE TWO AXES SIDE BY SIDE  (last-50, 5 seeds)")
    print("=" * 104)
    print("\nDENSITY SWEEP, with the mean n_cov each geometry actually realises:")
    print("%9s %6s %12s %12s" % ("mean_cov", "nodes", "mean n_cov", "hw:RMS"))
    rows = []
    for k in sorted(R):
        s = summarise(R[k], "%.2f" % k[0])
        rows.append((k, s))
        print("%9.2f %6d %12.2f %12.3f" % (k[0], k[1], s["mean_ncov"], s["ratio"]))
    s531 = summarise(R531, "5.3.1")
    print("%9s %6d %12.2f %12.3f  <- the 5.3.1 mesh"
          % ("3.64", 36, s531["mean_ncov"], s531["ratio"]))

    print("\nWITHIN-MESH n_cov BREAKDOWN in the 5.3.1 mesh:")
    print("%9s %12s %12s" % ("n_cov", "cells", "hw:RMS"))
    for k in NCOVS:
        v = [ratio(Rr, Rr["nc"] == k) for Rr in R531]
        v = [x for x in v if x]
        if v:
            print("%9d %12d %12.3f" % (k, sum(x[1] for x in v) // len(v),
                                       st.mean([x[0] for x in v])))
    print("\n  RANGE COMPARISON")
    print("   density sweep spans mean n_cov %.2f to %.2f"
          % (rows[0][1]["mean_ncov"], rows[-1][1]["mean_ncov"]))
    print("   within-mesh breakdown spans n_cov 1 to %d"
          % max(int(Rr["nc"].max()) for Rr in R531))

    print("\n" + "=" * 104)
    print("CROSS-TAB -- hw:RMS by (geometry, n_cov).  If density acted ONLY")
    print("through n_cov, every COLUMN would be flat.")
    print("=" * 104)
    print("%9s %8s" % ("mean_cov", "mean nc") + "".join("%9s" % ("nc=%d" % k) for k in NCOVS))
    for k in sorted(R):
        row = "%9.2f %8.2f" % (k[0], summarise(R[k], "")["mean_ncov"])
        for n in NCOVS:
            v = [ratio(Rr, Rr["nc"] == n) for Rr in R[k]]
            v = [x[0] for x in v if x]
            row += "%9s" % ("%.2f" % st.mean(v) if v else "--")
        print(row)

    print("\n" + "=" * 104)
    print("WHAT DRIVES beta*  --  beta* = mean(beta_i) + spread term")
    print("=" * 104)
    print("%9s %8s %14s %14s %14s %12s %12s"
          % ("mean_cov", "mean nc", "contrib spread", "mean(beta_i)",
             "spread term", "beta*", "mean(nu_i)"))
    for k in sorted(R):
        s = summarise(R[k], "")
        print("%9.2f %8.2f %14.2f %14.6f %14.6f %12.6f %12.4f"
              % (k[0], s["mean_ncov"], s["spread"], s["mean_be"],
                 s["spread_term"], s["be_star"], s["mean_nu"]))
    print("  (contrib spread in degrees; beta terms in normalised units)")

    print("\nSAME DECOMPOSITION BY n_cov WITHIN THE 5.3.1 MESH:")
    print("%9s %14s %14s %14s %12s %12s"
          % ("n_cov", "contrib spread", "mean(beta_i)", "spread term", "beta*", "mean(nu_i)"))
    for n in NCOVS:
        vals = defaultdict(list)
        for Rr in R531:
            m = (Rr["nc"] == n)
            if m.sum() < 40:
                continue
            vals["spread"].append(float(np.nanmean(Rr["spread"][m])))
            vals["mb"].append(float(np.nanmean(Rr["mean_be"][m])))
            vals["stm"].append(float(np.nanmean(Rr["spread_term"][m])))
            vals["bs"].append(float(np.nanmean(Rr["be_s"][m])))
            vals["nu"].append(float(np.nanmean(Rr["mean_nu"][m])))
        if vals["mb"]:
            print("%9d %14.2f %14.6f %14.6f %12.6f %12.4f"
                  % (n, st.mean(vals["spread"]), st.mean(vals["mb"]),
                     st.mean(vals["stm"]), st.mean(vals["bs"]), st.mean(vals["nu"])))


if __name__ == "__main__":
    main()
