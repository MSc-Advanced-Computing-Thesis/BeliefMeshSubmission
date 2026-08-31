# ALTERNATIVE FUSION RULES AT READOUT -- mixture and covariance intersection.
#
# Readout only over stored covering beliefs. No training path touched, no new
# runs, no default changed. Nothing here can alter a reported number; it
# recomputes the point estimate and interval from cell_beliefs_steps.
#
# WHY. The product rule makes evidence add (nu* = sum nu_i), which is the
# independence assumption. Diagnostic C measured the consequence: the fused
# predictive sd falls 5.5x with density, with hop and recency held fixed. Both
# alternatives below refuse to concentrate that way, by different means.
#
# CONSTRUCTION OF ARM 1 (MIXTURE). The contributor predictive densities are
# Student-t: location gamma_i, df 2*alpha_i, scale s_i = sqrt(beta_i (1+nu_i)
# / (nu_i alpha_i)), variance v_i = s_i^2 * df_i/(df_i - 2) for alpha_i > 1.
# The mixture is the equally weighted average of those densities, w_i = 1/n.
#   POINT ESTIMATE: the mixture's circular mean, arg(sum exp(i pi gamma_i))/pi.
#     For symmetric components this is the mixture's own circular mean, not an
#     approximation.
#   INTERVAL: a mixture of Student-t densities is NOT a Student-t, so the
#     interval requires a choice. Taken here by MOMENT MATCHING via the law of
#     total variance, V = mean(v_i) + mean(d_i^2) where d_i is the circular
#     deviation of component i from the mixture mean, then a Gaussian 90%
#     half-width 1.645*sqrt(V). This is the standard choice and it is
#     CONSERVATIVE IN ONE DIRECTION AND NOT THE OTHER: the second term cannot
#     be negative, so the mixture is at least as wide as the average component
#     and concentration cannot occur by construction; but a Gaussian quantile
#     on a heavy-tailed mixture understates tail mass, so the reported
#     half-width is if anything slightly too NARROW. Both effects are reported
#     rather than corrected.
#
# CONSTRUCTION OF ARM 2 (COVARIANCE INTERSECTION). CI fuses with weights
# summing to one: P_f = sum w_i P_i, P_f x_f = sum w_i P_i x_i, P_i = 1/v_i.
# This is not overconfident for ANY correlation between contributors.
#   WEIGHT RULE, ci_opt: the textbook rule chooses w to minimise the fused
#     covariance. IN ONE DIMENSION THAT PROGRAM IS DEGENERATE -- maximising
#     sum w_i P_i over the simplex puts all weight on the single largest P_i --
#     so CI-optimal reduces exactly to SELECTING THE MOST CERTAIN CONTRIBUTOR.
#     That is reported because the degeneracy is the finding, not a bug.
#   WEIGHT RULE, ci_fast: the standard non-degenerate heuristic, w_i =
#     (1/v_i) / sum_j (1/v_j). Fused precision is then sum P_i^2 / sum P_j,
#     which lies between max P_i and mean P_i -- strictly less concentrated
#     than the product rule, strictly more than the mixture.
#
# Run: python -u experiments/section5_2/run_fusion_alternatives.py

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
from analyse_estimators import LAST, load, truth_for, wrap

Z90 = sps.norm.ppf(0.95)
ARMS = ["argmax", "product", "mixture", "ci_opt", "ci_fast"]


def circ_mean(g, w, ok):
    """Weighted circular mean of gammas (period 2) over the last axis."""
    z = np.where(ok, w * np.exp(1j * np.pi * np.nan_to_num(g)), 0).sum(-1)
    return np.angle(z) / np.pi


def alt_readouts(R):
    """Point estimate and 90% half-width per (t, cell) for each alternative."""
    b = R["beliefs"].astype(np.float64)
    g, nu, al, be = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ok = np.isfinite(g) & np.isfinite(nu) & np.isfinite(al) & (al > 1.0)

    s2 = be * (1 + nu) / (nu * al)              # squared Student-t scale
    df = 2 * al
    with np.errstate(invalid="ignore", divide="ignore"):
        v = s2 * np.where(df > 2, df / (df - 2), np.nan)   # component variance
    ok &= np.isfinite(v) & (v > 0)
    n = ok.sum(-1)

    out = {}

    # ---- ARM 1: equal-weight mixture, moment-matched interval ----
    w = np.where(ok, 1.0 / np.maximum(n, 1)[..., None], 0.0)
    mu = circ_mean(g, w, ok)
    d = wrap(np.where(ok, g, np.nan) - mu[..., None])
    Ev = np.nansum(np.where(ok, v, 0), -1) / np.maximum(n, 1)      # mean(v_i)
    Vd = np.nansum(np.where(ok, d ** 2, 0), -1) / np.maximum(n, 1)  # mean(d_i^2)
    V = Ev + Vd
    out["mixture"] = (np.where(n > 0, mu, np.nan),
                      np.where(n > 0, Z90 * np.sqrt(V), np.nan))

    # ---- ARM 2a: CI-optimal -> selects the minimum-variance contributor ----
    vv = np.where(ok, v, np.inf)
    j = np.argmin(vv, -1)
    take = lambda a: np.take_along_axis(a, j[..., None], -1)[..., 0]
    vmin, gmin = take(vv), take(np.where(ok, g, np.nan))
    out["ci_opt"] = (np.where(n > 0, gmin, np.nan),
                     np.where(n > 0, Z90 * np.sqrt(vmin), np.nan))

    # ---- ARM 2b: CI-fast, w_i proportional to precision ----
    P = np.where(ok, 1.0 / np.where(ok, v, 1.0), 0.0)
    Psum = P.sum(-1)
    wf = np.where(Psum[..., None] > 0, P / np.maximum(Psum, 1e-300)[..., None], 0.0)
    Pf = (wf * P).sum(-1)                       # sum w_i P_i
    mu_ci = circ_mean(g, wf * P, ok)
    out["ci_fast"] = (np.where(Pf > 0, mu_ci, np.nan),
                      np.where(Pf > 0, Z90 / np.sqrt(np.maximum(Pf, 1e-300)), np.nan))
    return out


def arm_series(R, seed):
    """(err^2, half-width) per (t, cell) for all five arms, full history."""
    T, G = R["mse"].shape[0], R["mse"].shape[1]
    tr = truth_for(seed, T, G)
    S = {}

    nu, al, be = (R["nig"][..., i] for i in range(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        hw = sps.t.ppf(0.95, df=2 * al) * np.sqrt(be * (1 + nu) / (nu * al))
    S["argmax"] = (R["mse"], np.where(al > 1.0, hw, np.nan))

    f = R["fused"].astype(np.float64)
    nu, al, be = f[..., 1], f[..., 2], f[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        hw = sps.t.ppf(0.95, df=2 * al) * np.sqrt(be * (1 + nu) / (nu * al))
    S["product"] = (wrap(f[..., 0] - tr) ** 2, np.where(al > 1.0, hw, np.nan))

    for k, (est, h) in alt_readouts(R).items():
        S[k] = (wrap(est - tr) ** 2, h)
    return S


def agg(S, arm, mask):
    e2, hw = S[arm]
    ok = np.isfinite(e2) & np.isfinite(hw) & mask
    if ok.sum() < 8:
        return None
    m = float(np.mean(e2[ok]))
    rms = np.sqrt(m) * 180
    return dict(n=int(ok.sum()), mse=m,
                cov=float((np.sqrt(e2[ok]) <= hw[ok]).mean()),
                ratio=float(np.mean(hw[ok]) * 180 / rms))


def seed_of(d):
    return int(yaml.safe_load(open(Path(d) / "manifest.yaml"))["env_seed"])


def summarise(rows, key):
    v = [r[key] for r in rows if r]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else None


def block(title, dirs):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    per, whole, ncov, ncells = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list)), {}
    for d in dirs:
        d = Path(d)
        R, seed = load(d), seed_of(d)
        S = arm_series(R, seed)
        T = R["mse"].shape[0]
        last = np.zeros(R["mse"].shape, bool)
        last[T - LAST:] = True
        nc = R["ncov"].astype(int)
        for a in ARMS:
            per[a].append(agg(S, a, last))
            whole[a].append(float(np.nanmean(S[a][0])))
            for k in range(1, int(nc.max()) + 1):
                r = agg(S, a, last & (nc == k))
                if r and r["n"] >= 40:
                    ncov[k][a].append(r)
                    ncells[k] = r["n"]
    print("%-10s %-20s %-20s %-16s %-16s"
          % ("arm", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS"))
    for a in ARMS:
        w = (st.mean(whole[a]), st.stdev(whole[a]) if len(whole[a]) > 1 else 0.0)
        l, c, r = (summarise(per[a], k) for k in ("mse", "cov", "ratio"))
        print("%-10s %-20s %-20s %-16s %-16s"
              % (a, "%.5f +/- %.5f" % w, "%.5f +/- %.5f" % l,
                 "%.3f +/- %.3f" % c, "%.3f +/- %.3f" % r))
    if ncov:
        print("\nBY n_cov (last-50).  MSE penalty is vs the product readout at the same n_cov.")
        print("%6s %8s" % ("n_cov", "cells") + "".join("%22s" % a for a in ARMS))
        for k in sorted(ncov):
            row = "%6d %8d" % (k, ncells[k])
            base = summarise(ncov[k]["product"], "mse")
            for a in ARMS:
                m, r = summarise(ncov[k][a], "mse"), summarise(ncov[k][a], "ratio")
                if not m:
                    row += "%22s" % "--"
                    continue
                pen = 100 * (m[0] - base[0]) / base[0] if base and base[0] > 0 else 0.0
                row += "%22s" % ("%.4f/%.2f/%+.0f%%" % (m[0], r[0], pen))
            print(row)
        print("  cell format: MSE / hw:RMS / MSE penalty vs product")
    return per


def main():
    d531 = sorted(os.path.dirname(p) for p in
                  glob.glob("runs/chapter5_v2/s531_b2/nig_product_sampled_seed*/manifest.yaml"))
    block("5.3.1  nig_product_sampled, 36-node mesh, 5 seeds", d531)

    geo = defaultdict(list)
    for p in sorted(glob.glob("runs/chapter5_v2/s5_4_1_overlap/*/manifest.yaml")):
        m = yaml.safe_load(open(p))
        geo[(m["mean_cell_coverage"], m["n_nodes"])].append(os.path.dirname(p))

    print("\n" + "=" * 104)
    print("5.4.1 DENSITY SWEEP -- hw:RMS per arm across the four geometries")
    print("=" * 104)
    tab = {}
    for k, ds in sorted(geo.items()):
        tab[k] = block("5.4.1  mean coverage %.2f (%d nodes), 5 seeds" % k, ds)

    for key, lab, ref in (("ratio", "hw : RMS  (calibrated = 1.00)", 1.0),
                          ("cov", "90% coverage  (target = 0.90)", 0.90),
                          ("mse", "last-50 cell-space MSE", None)):
        print("\n" + "=" * 104)
        print("ACROSS THE DENSITY SWEEP -- %s" % lab)
        print("=" * 104)
        print("%9s %6s" % ("mean_cov", "nodes") + "".join("%14s" % a for a in ARMS))
        for k in sorted(tab):
            row = "%9.2f %6d" % k
            for a in ARMS:
                s = summarise(tab[k][a], key)
                row += "%14s" % ("%.4f" % s[0] if s else "--")
            print(row)
        if ref is not None:
            print("\n%9s %6s" % ("", "spread") + "".join("%14s" % a for a in ARMS))
            row = "%9s %6s" % ("", "max-min")
            for a in ARMS:
                v = [summarise(tab[k][a], key)[0] for k in sorted(tab) if summarise(tab[k][a], key)]
                row += "%14s" % ("%.4f" % (max(v) - min(v)) if v else "--")
            print(row)
            print("  A rule that calibrates AT ONE SETTING needs its value near %.2f in every"
                  % ref)
            print("  row and a small spread. Large spread = the deficit is only relocated.")


if __name__ == "__main__":
    main()
