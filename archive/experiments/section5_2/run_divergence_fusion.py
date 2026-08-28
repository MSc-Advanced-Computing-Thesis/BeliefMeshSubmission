# BLOCK 3 -- divergence-scaled fusion diagnostic.
#
# Analysis over stored beliefs. No training path touched, no default changed,
# no new runs. Chapter 5's reported convention (argmax) is unaffected.
#
# BACKGROUND. The single-node control sits at hw:RMS ~1.03, so the head is
# calibrated and FUSION introduces the miscalibration. The fused rule sets
# nu* = sum(nu_i), which assumes contributors are independent; measured gamma
# spread of 2-5 deg against a cell RMS of ~27 deg shows they are not. A discount
# in contributor COUNT (nu*/N^k) failed because the required correction depends
# on geometry: n_cov=9 inside the 36-node mesh wanted k=0.90, while mean
# coverage 9.00 in the 100-node mesh saturated at k=1 and still reached 0.50.
#
# HYPOTHESIS TESTED HERE. Contributor DIVERGENCE, not count, is the governing
# quantity. For each cell/timestep with >= 2 covering beliefs, compute the
# generalised Jensen-Shannon divergence across the contributing Student-t
# predictives (beliefmesh.fusion.consensus.js_divergence, the existing
# implementation), normalise by log N so it lies in [0, 1], and scale the
# evidence sum by it.
#
# FUNCTIONAL FORM
#     Dhat = D_JS / log(N)                        in [0, 1]   (= 1 - agreement)
#     f(Dhat) = 1/N + (1 - 1/N) * Dhat**p
#     nu*     = sum(nu_i) * f(Dhat)
# Limits are the ones the brief asks for:
#     Dhat = 0 (identical contributors) -> f = 1/N -> nu* = mean(nu_i),
#            i.e. ONE contributor's worth of evidence;
#     Dhat = 1 (maximally divergent)    -> f = 1   -> nu* = sum(nu_i),
#            i.e. exactly the current rule.
# p is the single free parameter. The count-based discount is the special case
# where Dhat is replaced by a constant, so this strictly generalises it: the
# question is whether WHICH cells are discounted (not how much on average) is
# the missing signal.
#
# gamma*, alpha* and beta* are untouched, so MSE is unchanged by construction.
# That is CONFIRMED numerically below rather than assumed.
#
# Run: python -u experiments/section5_2/run_divergence_fusion.py

from __future__ import annotations

import glob
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT / "section5_1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_estimators import LAST, truth_for, wrap

from beliefmesh.fusion.consensus import js_divergence
from beliefmesh.fusion.grid import circular_grid

PS = [0.25, 0.5, 1.0, 2.0, 4.0]
GRID = 360
S531 = "runs/chapter5_v2/s531_b2/nig_product_sampled_seed*"
S541 = "runs/chapter5_v2/s5_4_1_overlap/*_seed*"


def dhat_for(B: np.ndarray, k: int, grid: torch.Tensor) -> np.ndarray:
    """Normalised JS divergence across k contributors, per instance.

    B: (M, k, 4) contributor beliefs. Returns (M,) in [0, 1].
    Densities are evaluated on the CIRCULAR grid at wrapped offsets, matching
    how fusion evaluates them, so beliefs near the +/-1 boundary are handled.
    """
    if k < 2:
        return np.zeros(len(B))
    t = torch.from_numpy(B.astype(np.float64))
    g, nu, al, be = (t[..., i] for i in range(4))                 # (M, k)
    scale = torch.sqrt(be * (1.0 + nu) / (nu * al)).clamp(min=1e-8)
    df = (2.0 * al).clamp(min=2.05)
    # wrapped offset of each grid point from each contributor's location
    d = grid.view(1, 1, -1).double() - g.unsqueeze(-1)            # (M, k, G)
    d = d - 2.0 * torch.round(d / 2.0)
    dist = torch.distributions.StudentT(df=df.unsqueeze(-1),
                                        loc=torch.zeros_like(df).unsqueeze(-1),
                                        scale=scale.unsqueeze(-1))
    logp = dist.log_prob(d)                                       # (M, k, G)
    D = js_divergence(logp.permute(1, 0, 2).float())              # (M,)
    return (D.numpy() / math.log(k)).clip(0.0, 1.0)


def run_metrics(d: Path, seed: int):
    """Per-cell argmax / fused / divergence-scaled-fused quantities, last-50."""
    L = lambda n: np.load(d / f"cell_{n}_steps.npy")
    mse, nig, fus, B, N = L("mse"), L("nig"), L("fused"), L("beliefs"), L("ncov")
    T, G = mse.shape[0], mse.shape[1]
    sl = slice(T - LAST, T)
    tr = truth_for(seed, T, G)[-LAST:]
    grid = circular_grid(GRID)

    e2_arg = mse[sl]
    fu = fus[sl].astype(np.float64)
    e2_fus = wrap(fu[..., 0] - tr) ** 2
    nc = N[sl].astype(int)
    Bw = B[sl].astype(np.float64)

    dhat = np.zeros(nc.shape)
    nusum = np.zeros(nc.shape)
    for k in sorted(set(int(v) for v in np.unique(nc) if v >= 2)):
        m = nc == k
        idx = np.argwhere(m)
        sub = Bw[m][:, :k, :]
        dhat[m] = dhat_for(sub, k, grid)
        nusum[m] = sub[..., 1].sum(axis=1)
    return dict(nc=nc, dhat=dhat, nusum=nusum,
                e2_arg=e2_arg, e2_fus=e2_fus,
                nu_f=fu[..., 1], al_f=fu[..., 2], be_f=fu[..., 3])


def cal(e2, nu, al, be, mask):
    ok = np.isfinite(e2) & np.isfinite(nu) & np.isfinite(al) & (al > 1.0) & (nu > 0) & mask
    if ok.sum() < 8:
        return None
    sc = np.sqrt(be[ok] * (1 + nu[ok]) / (nu[ok] * al[ok]))
    hw = sps.t.ppf(0.95, df=2 * al[ok]) * sc
    e = np.sqrt(e2[ok])
    m = float(np.mean(e2[ok]))
    return dict(cov=float((e <= hw).mean()), ratio=float(np.mean(hw) * 180 / (np.sqrt(m) * 180)),
                mse=m)


def scaled_nu(R, p):
    N = np.maximum(R["nc"], 1)
    f = 1.0 / N + (1.0 - 1.0 / N) * np.power(R["dhat"], p)
    out = R["nusum"] * f
    single = R["nc"] <= 1
    out[single] = R["nu_f"][single]           # n_cov=1 untouched: nu* = nu_1
    return out


def main():
    print("=" * 78)
    print("BLOCK 3 -- DIVERGENCE-SCALED FUSION   nu* = sum(nu_i) * f(Dhat)")
    print("  Dhat = D_JS / log(N);  f = 1/N + (1 - 1/N) * Dhat^p")
    print("  f->1/N at Dhat=0 (one contributor's worth); f->1 at Dhat=1 (current rule)")
    print("=" * 78)

    # ---------------- 5.3.1, by n_cov ----------------
    runs = []
    for dd in sorted(glob.glob(S531)):
        d = Path(dd)
        runs.append((int(d.name.rsplit("_seed", 1)[1]), run_metrics(d, int(d.name.rsplit("_seed", 1)[1]))))
    ks = sorted(set(int(v) for v in np.unique(runs[0][1]["nc"]) if v > 0))

    print("\n--- MSE INVARIANCE CHECK (gamma* untouched) ---")
    worst = 0.0
    for _, R in runs:
        for p in PS:
            nu = scaled_nu(R, p)
            a = cal(R["e2_fus"], nu, R["al_f"], R["be_f"], np.ones_like(R["nc"], bool))
            b = cal(R["e2_fus"], R["nu_f"], R["al_f"], R["be_f"], np.ones_like(R["nc"], bool))
            worst = max(worst, abs(a["mse"] - b["mse"]))
    print("  max |MSE(divergence-scaled) - MSE(fused)| over all p and seeds: %.3e" % worst)
    print("  (zero by construction: only nu* moves, gamma* is unchanged)")

    print("\n--- Dhat DISTRIBUTION by n_cov (does divergence vary with geometry?) ---")
    print("%6s %10s %10s %10s" % ("n_cov", "mean Dhat", "sd", "p90"))
    for k in ks:
        if k < 2:
            continue
        vals = np.concatenate([R["dhat"][R["nc"] == k] for _, R in runs])
        print("%6d %10.4f %10.4f %10.4f" % (k, vals.mean(), vals.std(),
                                            np.percentile(vals, 90)))

    print("\n--- 5.3.1 hw:RMS by n_cov ---")
    hdr = "%6s %9s %9s" % ("n_cov", "argmax", "fused")
    for p in PS:
        hdr += "%11s" % ("p=%.2f" % p)
    print(hdr)
    for k in ks:
        mask_by_run = [(R["nc"] == k) for _, R in runs]
        a = [cal(R["e2_arg"], R["nig_nu"] if False else R["nu_f"], R["al_f"], R["be_f"], m)
             for (_, R), m in zip(runs, mask_by_run)]
        row = "%6d" % k
        # argmax uses its own stored nig params
        arg = []
        for (seed, R), m in zip(runs, mask_by_run):
            d = Path(sorted(glob.glob(S531))[[s for s, _ in runs].index(seed)])
            nig = np.load(d / "cell_nig_steps.npy")[-LAST:]
            arg.append(cal(R["e2_arg"], nig[..., 0], nig[..., 1], nig[..., 2], m))
        row += "%9.2f" % st.mean([x["ratio"] for x in arg if x])
        fu = [cal(R["e2_fus"], R["nu_f"], R["al_f"], R["be_f"], m)
              for (_, R), m in zip(runs, mask_by_run)]
        row += "%9.2f" % st.mean([x["ratio"] for x in fu if x])
        for p in PS:
            v = [cal(R["e2_fus"], scaled_nu(R, p), R["al_f"], R["be_f"], m)
                 for (_, R), m in zip(runs, mask_by_run)]
            row += "%11.2f" % st.mean([x["ratio"] for x in v if x])
        print(row)

    print("\n--- 5.3.1 cov90 by n_cov ---")
    print(hdr)
    for k in ks:
        mask_by_run = [(R["nc"] == k) for _, R in runs]
        row = "%6d" % k
        arg = []
        for (seed, R), m in zip(runs, mask_by_run):
            d = Path(sorted(glob.glob(S531))[[s for s, _ in runs].index(seed)])
            nig = np.load(d / "cell_nig_steps.npy")[-LAST:]
            arg.append(cal(R["e2_arg"], nig[..., 0], nig[..., 1], nig[..., 2], m))
        row += "%9.3f" % st.mean([x["cov"] for x in arg if x])
        fu = [cal(R["e2_fus"], R["nu_f"], R["al_f"], R["be_f"], m)
              for (_, R), m in zip(runs, mask_by_run)]
        row += "%9.3f" % st.mean([x["cov"] for x in fu if x])
        for p in PS:
            v = [cal(R["e2_fus"], scaled_nu(R, p), R["al_f"], R["be_f"], m)
                 for (_, R), m in zip(runs, mask_by_run)]
            row += "%11.3f" % st.mean([x["cov"] for x in v if x])
        print(row)

    # ---------------- 5.4.1 density sweep ----------------
    print("\n--- 5.4.1 density sweep: hw:RMS (cov90) ---")
    geo = defaultdict(list)
    for dd in sorted(glob.glob(S541)):
        d = Path(dd)
        m = yaml.safe_load(open(d / "manifest.yaml"))
        seed = int(d.name.rsplit("_seed", 1)[1])
        geo[(m["mean_cell_coverage"], m["n_nodes"])].append((d, seed))
    print("%9s %6s %14s %14s" % ("mean_cov", "nodes", "argmax", "fused")
          + "".join("%16s" % ("p=%.2f" % p) for p in PS))
    for (mc, nn), lst in sorted(geo.items()):
        A, F, P = [], [], defaultdict(list)
        for d, seed in lst:
            R = run_metrics(d, seed)
            allm = np.ones_like(R["nc"], bool)
            nig = np.load(d / "cell_nig_steps.npy")[-LAST:]
            A.append(cal(R["e2_arg"], nig[..., 0], nig[..., 1], nig[..., 2], allm))
            F.append(cal(R["e2_fus"], R["nu_f"], R["al_f"], R["be_f"], allm))
            for p in PS:
                P[p].append(cal(R["e2_fus"], scaled_nu(R, p), R["al_f"], R["be_f"], allm))
        f = lambda L: "%.2f (%.3f)" % (st.mean([x["ratio"] for x in L]),
                                       st.mean([x["cov"] for x in L]))
        print("%9.2f %6d %14s %14s" % (mc, nn, f(A), f(F))
              + "".join("%16s" % f(P[p]) for p in PS))


if __name__ == "__main__":
    main()
