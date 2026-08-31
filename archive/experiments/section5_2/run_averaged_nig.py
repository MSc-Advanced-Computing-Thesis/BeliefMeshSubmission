# AVERAGED NIG FUSION AT READOUT -- weights that sum to one instead of
# evidence that sums.  DIAGNOSTIC ONLY: no training path touched, no default
# changed, no new runs.  Recomputes the fused belief from stored contributor
# beliefs.
#
# THE RULE.  fuse_nig_product already accepts per-contributor tempering
# exponents w_i, so the averaging form is that same closed form with
# w_i = 1/N.  Nothing new is derived; every parameter follows from the
# existing algebra with n_eff = sum_i w_i = 1:
#
#     n_eff   = 1
#     nu*     = sum_i (1/N) nu_i          = mean(nu_i)        <- averages
#     gamma*  = sum_i (1/N) nu_i x_i / nu*  = sum nu_i x_i / sum nu_i
#                                                             <- UNCHANGED
#     alpha*  = sum_i (1/N) alpha_i + 1.5 (n_eff - 1)
#             = mean(alpha_i)                                 <- no +1.5(N-1)
#     beta*   = mean(beta_i) + 0.5 ( mean(nu_i x_i^2) - nu* x*^2 )
#
# gamma* is unchanged because the 1/N cancels between numerator and
# denominator.  That is asserted numerically below, not assumed: if it holds,
# MSE is untouched and this is purely an interval change.
#
# SINGLE-CONTRIBUTOR LIMIT.  N=1 gives w=1 and n_eff=1, so nu*=nu_1,
# alpha*=alpha_1, beta*=beta_1, gamma*=gamma_1 -- identical to the product
# rule, which itself reduces to the contributor's own belief.  Checked
# explicitly, and the n_cov=1 column of every table below is the empirical
# control where all estimators must coincide.
#
# WHAT AVERAGING DOES AND DOES NOT DISCARD.  nu* = mean(nu_i) is insensitive
# to whether contributors agree: nine tight agreeing beliefs and nine
# disagreeing ones give the same evidence.  But beta* keeps the spread term
# 0.5(mean(nu_i x_i^2) - nu* x*^2), which is the nu-weighted variance of the
# contributor locations, so disagreement still widens the predictive.  The
# high- vs low-spread comparison at the end measures whether that residual
# sensitivity is enough.
#
# Run: python -u experiments/section5_2/run_averaged_nig.py

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
from analyse_estimators import LAST, truth_for, wrap
from beliefmesh.fusion.nig_product import fuse_nig_product

NU_FLOOR, ALPHA_FLOOR, BETA_FLOOR = 1e-6, 1.0 + 1e-3, 1e-6
ARMS = ["argmax", "product", "averaged"]


def wrap_np(x):
    return x - 2.0 * np.round(x / 2.0)


def averaged(bel, fin, nc):
    """Vectorised w_i = 1/N closed form. Validated against fuse_nig_product."""
    g, nu, al, be = (bel[..., i] for i in range(4))
    N = np.maximum(nc, 1)[..., None].astype(float)
    w = np.where(fin, 1.0 / N, 0.0)
    ref = g[..., 0]                                  # slot 0 always filled
    x = np.where(fin, wrap_np(g - ref[..., None]), 0.0)

    nu_s = np.maximum((w * np.where(fin, nu, 0.0)).sum(-1), NU_FLOOR)
    x_s = (w * np.where(fin, nu, 0.0) * x).sum(-1) / nu_s
    gam = wrap_np(ref + x_s)
    al_s = np.maximum((w * np.where(fin, al, 0.0)).sum(-1), ALPHA_FLOOR)
    sq = (w * np.where(fin, nu, 0.0) * x * x).sum(-1)
    be_s = np.maximum((w * np.where(fin, be, 0.0)).sum(-1)
                      + 0.5 * (sq - nu_s * x_s * x_s), BETA_FLOOR)
    return gam, nu_s, al_s, be_s


def validate(bel, fin, nc, rng, n=400):
    """Assert the vectorised form equals fuse_nig_product with w_i = 1/N,
    and that N=1 reduces to the contributor's own belief."""
    gam, nu_s, al_s, be_s = averaged(bel, fin, nc)
    T, G = bel.shape[0], bel.shape[1]
    worst = 0.0
    checked = 0
    for _ in range(n):
        t, r, c = rng.integers(T), rng.integers(G), rng.integers(G)
        k = int(nc[t, r, c])
        if k < 1:
            continue
        bl = [tuple(float(v) for v in bel[t, r, c, j]) for j in range(k)]
        ref = fuse_nig_product(bl, weights=[1.0 / k] * k)
        got = (gam[t, r, c], nu_s[t, r, c], al_s[t, r, c], be_s[t, r, c])
        d = max(abs(wrap(ref[0] - got[0])),
                *[abs(ref[i] - got[i]) for i in range(1, 4)])
        worst = max(worst, d); checked += 1
    assert checked > 0, "NO CELLS CHECKED -- validation is vacuous"
    assert worst < 1e-9, "vectorised averaged fusion != fuse_nig_product (%.2e)" % worst

    single = np.where(nc == 1)
    ok1 = True
    if len(single[0]):
        i = tuple(a[:200] for a in single)
        ok1 = bool(np.allclose(gam[i], bel[..., 0][i + (0,)], atol=1e-12)
                   and np.allclose(nu_s[i], bel[..., 1][i + (0,)], atol=1e-12)
                   and np.allclose(al_s[i], bel[..., 2][i + (0,)], atol=1e-12)
                   and np.allclose(be_s[i], bel[..., 3][i + (0,)], atol=1e-12))
    return checked, worst, ok1


def hw_of(nu, al, be):
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.sqrt(be * (1 + nu) / (nu * al))
    return sps.t.ppf(0.95, df=2 * al) * sc


def series(d, seed):
    bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(d / "cell_ncov_steps.npy").astype(int)
    fused = np.load(d / "cell_fused_steps.npy").astype(np.float64)
    nig = np.load(d / "cell_nig_steps.npy")
    mse = np.load(d / "cell_mse_steps.npy")
    fin = np.isfinite(bel[..., 0])
    T, G = mse.shape[0], mse.shape[1]
    tr = truth_for(seed, T, G)

    S = {}
    S["argmax"] = (mse, np.where(nig[..., 1] > 1.0,
                                 hw_of(nig[..., 0], nig[..., 1], nig[..., 2]), np.nan))
    S["product"] = (wrap_np(fused[..., 0] - tr) ** 2,
                    hw_of(fused[..., 1], fused[..., 2], fused[..., 3]))
    gam, nu_s, al_s, be_s = averaged(bel, fin, nc)
    S["averaged"] = (wrap_np(gam - tr) ** 2, hw_of(nu_s, al_s, be_s))
    # contributor gamma spread (nu-weighted sd of locations, degrees)
    g = bel[..., 0]
    x = np.where(fin, wrap_np(g - g[..., 0][..., None]), np.nan)
    spread = np.nanstd(x, axis=-1) * 180
    dgam = np.abs(wrap_np(gam - fused[..., 0])) * 180
    return S, nc, spread, dgam


def agg(S, arm, mask):
    e2, hw = S[arm]
    ok = np.isfinite(e2) & np.isfinite(hw) & mask
    if ok.sum() < 8:
        return None
    m = float(np.mean(e2[ok]))
    return dict(n=int(ok.sum()), mse=m,
                cov=float((np.sqrt(e2[ok]) <= hw[ok]).mean()),
                ratio=float(np.mean(hw[ok]) * 180 / (np.sqrt(m) * 180)))


def block(title, dirs, seeds, show_ncov=True):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    per, whole, byn = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    ncells, dg, sp = {}, [], defaultdict(lambda: defaultdict(list))
    rng = np.random.default_rng(0)
    for d, seed in zip(dirs, seeds):
        d = Path(d)
        S, nc, spread, dgam = series(d, seed)
        bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
        chk, worst, ok1 = validate(bel, np.isfinite(bel[..., 0]), nc, rng)
        T = nc.shape[0]
        last = np.zeros(nc.shape, bool); last[T - LAST:] = True
        dg.append(float(np.nanmax(dgam[last])))
        for a in ARMS:
            per[a].append(agg(S, a, last))
            whole[a].append(float(np.nanmean(S[a][0])))
            if show_ncov:
                for k in range(1, int(nc.max()) + 1):
                    r = agg(S, a, last & (nc == k))
                    if r and r["n"] >= 40:
                        byn[k][a].append(r); ncells[k] = r["n"]
        # spread strata (cells with >= 2 contributors)
        finite = np.isfinite(spread) & (nc >= 2) & last
        if finite.sum() > 100:
            lo, hi = np.nanpercentile(spread[finite], [33, 67])
            for lab, msk in (("low", finite & (spread <= lo)),
                             ("high", finite & (spread >= hi))):
                for a in ("product", "averaged"):
                    r = agg(S, a, msk)
                    if r:
                        sp[lab][a].append(r)
    print("  validation: %d sampled cells vs fuse_nig_product, max |diff| %.2e; "
          "n_cov=1 reduces to own belief: %s" % (chk, worst, ok1))
    print("  max |gamma*_averaged - gamma*_product| over last-50 cells: %.3e deg"
          % max(dg))
    print("\n%-10s %-20s %-20s %-15s %-15s" % ("arm", "whole-run MSE", "last-50 MSE",
                                               "90% coverage", "hw : RMS"))
    for a in ARMS:
        w = (st.mean(whole[a]), st.stdev(whole[a]) if len(whole[a]) > 1 else 0)
        f = lambda k, p=5: "%.*f +/- %.*f" % (p, st.mean([x[k] for x in per[a] if x]), p,
                                              st.stdev([x[k] for x in per[a] if x]))
        print("%-10s %-20s %-20s %-15s %-15s"
              % (a, "%.5f +/- %.5f" % w, f("mse"), f("cov", 3), f("ratio", 3)))
    if show_ncov and byn:
        print("\nBY n_cov (last-50).  n_cov=1 is the control: all three must coincide.")
        print("%6s %8s" % ("n_cov", "cells") + "".join("%26s" % a for a in ARMS))
        for k in sorted(byn):
            row = "%6d %8d" % (k, ncells[k])
            base = st.mean([x["mse"] for x in byn[k]["product"]]) if byn[k]["product"] else None
            for a in ARMS:
                v = byn[k][a]
                if not v:
                    row += "%26s" % "--"; continue
                mm = st.mean([x["mse"] for x in v]); rr = st.mean([x["ratio"] for x in v])
                cc = st.mean([x["cov"] for x in v])
                pen = 100 * (mm - base) / base if base else 0.0
                row += "%26s" % ("%.4f/%.2f/%.2f/%+.0f%%" % (mm, cc, rr, pen))
            print(row)
        print("  cell: MSE / coverage / hw:RMS / MSE change vs product")
    if sp:
        print("\nCONTRIBUTOR DISAGREEMENT (cells with n_cov >= 2, last-50):")
        print("%-8s %12s %12s %12s %12s"
              % ("spread", "prod hw:RMS", "avg hw:RMS", "prod cov", "avg cov"))
        for lab in ("low", "high"):
            if lab in sp:
                print("%-8s %12.3f %12.3f %12.3f %12.3f"
                      % (lab, st.mean([x["ratio"] for x in sp[lab]["product"]]),
                         st.mean([x["ratio"] for x in sp[lab]["averaged"]]),
                         st.mean([x["cov"] for x in sp[lab]["product"]]),
                         st.mean([x["cov"] for x in sp[lab]["averaged"]])))
    return per


def main():
    d531 = sorted(os.path.dirname(p) for p in
                  glob.glob("runs/chapter5_v2/s531_b2/nig_product_sampled_seed*/manifest.yaml"))
    s531 = [int(yaml.safe_load(open(Path(d) / "manifest.yaml"))["env_seed"]) for d in d531]
    block("5.3.1  nig_product_sampled, 36-node mesh, 5 seeds", d531, s531)

    geo = defaultdict(list)
    for p in sorted(glob.glob("runs/chapter5_v2/s5_4_1_overlap/*/manifest.yaml")):
        m = yaml.safe_load(open(p))
        geo[(m["mean_cell_coverage"], m["n_nodes"])].append(
            (os.path.dirname(p), int(m["env_seed"])))
    tab = {}
    for k, v in sorted(geo.items()):
        tab[k] = block("5.4.1  mean coverage %.2f (%d nodes), 5 seeds" % k,
                       [x[0] for x in v], [x[1] for x in v], show_ncov=False)

    print("\n" + "=" * 100)
    print("ACROSS THE DENSITY SWEEP -- hw:RMS  (calibrated = 1.00)")
    print("=" * 100)
    print("%9s %6s" % ("mean_cov", "nodes") + "".join("%14s" % a for a in ARMS))
    for k in sorted(tab):
        row = "%9.2f %6d" % k
        for a in ARMS:
            row += "%14.4f" % st.mean([x["ratio"] for x in tab[k][a] if x])
        print(row)
    row = "%9s %6s" % ("", "spread")
    for a in ARMS:
        v = [st.mean([x["ratio"] for x in tab[k][a] if x]) for k in sorted(tab)]
        row += "%14.4f" % (max(v) - min(v))
    print(row + "   <- max-min across geometries")


if __name__ == "__main__":
    main()
