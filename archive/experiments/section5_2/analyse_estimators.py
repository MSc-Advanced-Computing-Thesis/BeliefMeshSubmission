# Estimator comparison + integrity verification + gamma-spread degeneracy,
# over a directory of regenerated mesh runs. Pure recomputation.
#
# ARGMAX estimator  : the highest-certainty covering node's belief per cell
#                     (the convention used until 2026-08-27).
# FUSED estimator   : the closed-form NIG product of ALL covering nodes'
#                     beliefs, gamma* as the point estimate and the interval
#                     from the fused parameters. This is Sec 3.4's own output
#                     rule -- what a consumer actually does when reading a
#                     cell -- and it is a capability that exists BECAUSE
#                     beliefs are exchanged.
#
# Ground truth is not stored, but is analytically reconstructible: the
# environment precomputes cell_rotations deterministically from rotation_seed
# and the label is _label(rotation, row, col, step). Validated against
# cell_mse_steps to ~1e-7 before use.
#
# INTEGRITY CHECK (2026-08-27): a directory assembled from two concurrent
# executions writing the same tag would hold mutually inconsistent arrays.
# Every run is checked before its numbers are used:
#   - all arrays share the expected step count
#   - certainty reconstructed from cell_beliefs_steps == cell_cert_steps
#   - cell_nig_steps == the argmax belief's (nu, alpha, beta)
#   - cell_fused_steps == fuse_nig_product(stored covering beliefs)
# A run failing any of these is reported and excluded, not silently used.
#
# Run: python -u experiments/section5_2/analyse_estimators.py <root>

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import statistics as st
import yaml
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.nig_product import fuse_nig_product

LAST = 50
EX = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
_truth_cache: dict[int, np.ndarray] = {}


def wrap(x):
    return x - 2.0 * np.round(x / 2.0)


def truth_for(seed: int, T: int, G: int) -> np.ndarray:
    if seed in _truth_cache:
        return _truth_cache[seed]
    env = GridEnvironment(G, np.full((T, G, G), 0.5), rotation_seed=seed,
                          offset_field=build_dynamic_offset_field(G, T),
                          excluded_rotation_ranges=EX, apply_colour_filter=False)
    rot = env.cell_rotations
    t = np.empty((T, G, G))
    for s in range(T):
        for r in range(G):
            for c in range(G):
                t[s, r, c] = env._label(rot[s, r, c], r, c, s).item()
    _truth_cache[seed] = t
    return t


def load(d: Path):
    n = {k: np.load(d / f"cell_{k}_steps.npy") for k in
         ("mse", "cert", "nig", "hop", "beliefs", "ncov", "fused")}
    n["manifest"] = yaml.safe_load(open(d / "manifest.yaml"))
    return n


def integrity(d: Path, R, T_expected: int) -> tuple[bool, list[str]]:
    """Would a directory assembled from two executions pass? No."""
    msgs = []
    ok = True
    for k in ("mse", "cert", "nig", "hop", "beliefs", "ncov", "fused"):
        if R[k].shape[0] != T_expected:
            ok = False
            msgs.append(f"{k} step count {R[k].shape[0]} != {T_expected}")
    B = R["beliefs"].astype(np.float64)
    g, nu, al, be = (B[..., i] for i in range(4))
    epi = be / (np.maximum(nu, 1e-6) * np.maximum(al - 1, 1e-6))
    cert_all = 1.0 / (1.0 + np.minimum(epi, 10.0))
    cert_all[~np.isfinite(g)] = -np.inf
    best = np.argmax(cert_all, axis=-1)
    take = lambda a: np.take_along_axis(a, best[..., None], -1)[..., 0]
    m = np.isfinite(R["cert"])
    d_cert = float(np.max(np.abs(take(cert_all)[m] - R["cert"][m])))
    if d_cert > 1e-9:
        ok = False
    msgs.append(f"cert reconstruction max|d|={d_cert:.2e}")
    d_nig = max(float(np.max(np.abs(take(x)[m] - R["nig"][..., i][m])))
                for i, x in enumerate((nu, al, be)))
    if d_nig > 1e-5:
        ok = False
    msgs.append(f"cell_nig vs argmax belief max|d|={d_nig:.2e}")
    # fused array vs re-fusing the stored contributions (sampled subset)
    rng = np.random.default_rng(0)
    idx = np.argwhere(R["ncov"] > 1)
    pick = idx[rng.choice(len(idx), size=min(300, len(idx)), replace=False)]
    worst = 0.0
    for s, r, c in pick:
        k = int(R["ncov"][s, r, c])
        f = fuse_nig_product([tuple(float(v) for v in B[s, r, c, j]) for j in range(k)])
        worst = max(worst, float(np.max(np.abs(np.asarray(f) - R["fused"][s, r, c]))))
    if worst > 1e-4:
        ok = False
    msgs.append(f"fused vs re-fused max|d|={worst:.2e}")
    return ok, msgs


def metrics(R, seed, mask=None, hop=None):
    """(mse, coverage, hw:RMS) under both estimators over the last-50 window."""
    T, G = R["mse"].shape[0], R["mse"].shape[1]
    tr = truth_for(seed, T, G)[-LAST:]
    sl = slice(T - LAST, T)
    out = {}
    for name, (gam, nu, al, be) in (
        ("argmax", (None, R["nig"][sl, ..., 0], R["nig"][sl, ..., 1], R["nig"][sl, ..., 2])),
        ("fused", (R["fused"][sl, ..., 0].astype(np.float64), R["fused"][sl, ..., 1].astype(np.float64),
                   R["fused"][sl, ..., 2].astype(np.float64), R["fused"][sl, ..., 3].astype(np.float64))),
    ):
        if name == "argmax":
            e2 = R["mse"][sl]
        else:
            e2 = wrap(gam - tr) ** 2
        ok = np.isfinite(e2) & np.isfinite(nu) & np.isfinite(al) & (al > 1.0)
        if mask is not None:
            ok &= np.broadcast_to(mask, ok.shape) if mask.ndim == 2 else mask
        if hop is not None:
            ok &= (R["hop"][sl] == hop)
        if ok.sum() < 8:
            out[name] = None
            continue
        sc = np.sqrt(be[ok] * (1 + nu[ok]) / (nu[ok] * al[ok]))
        hw = sps.t.ppf(0.95, df=2 * al[ok]) * sc
        m = float(np.mean(e2[ok]))
        rms = np.sqrt(m) * 180
        out[name] = dict(mse=m, cov=float((np.sqrt(e2[ok]) <= hw).mean()),
                         hw=float(np.mean(hw) * 180), ratio=float(np.mean(hw) * 180 / rms),
                         n=int(ok.sum()))
    return out


def whole_run_mse(R, seed):
    T, G = R["mse"].shape[0], R["mse"].shape[1]
    tr = truth_for(seed, T, G)
    return (float(np.nanmean(R["mse"])),
            float(np.nanmean(wrap(R["fused"][..., 0].astype(np.float64) - tr) ** 2)))


def random_node_metrics(R, seed, k_mask, rng):
    """Coverage/MSE of a UNIFORMLY RANDOM covering node per cell, as the
    selection control for argmax. argmax picks the most confident node, which
    may also be the best-adapted one; if random selection lands near the fused
    figure rather than the argmax one, the argmax number is a selection
    artefact."""
    T, G = R["mse"].shape[0], R["mse"].shape[1]
    tr = truth_for(seed, T, G)[-LAST:]
    B = R["beliefs"][-LAST:].astype(np.float64)
    N = R["ncov"][-LAST:]
    sel = np.zeros(N.shape, dtype=int)
    nz = N > 0
    sel[nz] = (rng.random(int(nz.sum())) * N[nz]).astype(int)
    g, nu, al, be = (np.take_along_axis(B[..., i], sel[..., None], -1)[..., 0]
                     for i in range(4))
    e2 = wrap(g - tr) ** 2
    ok = np.isfinite(e2) & np.isfinite(nu) & (al > 1.0) & k_mask
    if ok.sum() < 8:
        return None
    sc = np.sqrt(be[ok] * (1 + nu[ok]) / (nu[ok] * al[ok]))
    hw = sps.t.ppf(0.95, df=2 * al[ok]) * sc
    m = float(np.mean(e2[ok])); rms = np.sqrt(m) * 180
    return dict(mse=m, cov=float((np.sqrt(e2[ok]) <= hw).mean()),
                hw=float(np.mean(hw) * 180), ratio=float(np.mean(hw) * 180 / rms))


def gamma_spread(R):
    """Spread of gamma ACROSS covering nodes per cell, by n_cov. The fused rule
    treats contributors as independent; this measures how far apart they
    actually are, wherever that assumption is relied on."""
    B = R["beliefs"][-LAST:].astype(np.float64)
    N = R["ncov"][-LAST:]
    out = {}
    for k in sorted(set(int(v) for v in np.unique(N) if v > 0)):
        m = (N == k)
        if m.sum() < 8:
            continue
        g = B[..., 0][m][:, :k]
        ref = g[:, :1]
        off = wrap(g - ref)
        out[k] = dict(sd_deg=float(np.mean(off.std(axis=1)) * 180),
                      range_deg=float(np.mean(off.max(1) - off.min(1)) * 180),
                      n_cells=int(m.sum()))
    return out


def main(root: Path):
    runs = defaultdict(list)
    for d in sorted(root.iterdir()):
        if not (d / "manifest.yaml").exists():
            continue
        arm = d.name.rsplit("_seed", 1)[0]
        seed = int(d.name.rsplit("_seed", 1)[1])
        runs[arm].append((seed, d))

    print("=== INTEGRITY VERIFICATION ===")
    good = defaultdict(list)
    for arm, lst in runs.items():
        for seed, d in lst:
            R = load(d)
            T = int(R["manifest"]["total_steps"])
            ok, msgs = integrity(d, R, T)
            print(f"  {'PASS' if ok else 'FAIL'}  {d.name:38s} " + "; ".join(msgs))
            if ok:
                good[arm].append((seed, R))

    print("\n=== ESTIMATOR COMPARISON (last %d steps, mean+/-sd over seeds) ===" % LAST)
    print("%-22s %-3s %-21s %-21s %-15s %-15s"
          % ("arm", "n", "whole-run argmax", "whole-run fused", "cov90 argmax", "cov90 fused"))
    for arm, lst in good.items():
        wa, wf, ca, cf = [], [], [], []
        for seed, R in lst:
            a, f = whole_run_mse(R, seed)
            wa.append(a); wf.append(f)
            m = metrics(R, seed)
            ca.append(m["argmax"]["cov"]); cf.append(m["fused"]["cov"])
        f2 = lambda v: "%.5f+/-%.5f" % (st.mean(v), st.stdev(v) if len(v) > 1 else 0)
        f3 = lambda v: "%.3f+/-%.3f" % (st.mean(v), st.stdev(v) if len(v) > 1 else 0)
        print("%-22s %-3d %-21s %-21s %-15s %-15s"
              % (arm, len(lst), f2(wa), f2(wf), f3(ca), f3(cf)))
        d = [100 * (b - a) / a for a, b in zip(wa, wf)]
        print("      fused-vs-argmax whole-run: %+.2f%% +/- %.2f  [floor 0.19%%]"
              % (st.mean(d), st.stdev(d) if len(d) > 1 else 0))

    print("\n=== BY COVERING COUNT n_cov (n_cov=1 is the CONTROL: estimators identical) ===")
    for arm, lst in good.items():
        print(" %s" % arm)
        print("   %5s %8s | %-19s %-19s | %-13s %-13s"
              % ("n_cov", "cells", "MSE argmax", "MSE fused", "cov argmax", "cov fused"))
        ks = sorted(set(int(v) for v in np.unique(lst[0][1]["ncov"]) if v > 0))
        for k in ks:
            am, fm, ac, fc, nn = [], [], [], [], []
            for seed, R in lst:
                mask = (R["ncov"][-LAST:] == k)
                mm = metrics(R, seed, mask=mask)
                if mm["argmax"] and mm["fused"]:
                    am.append(mm["argmax"]["mse"]); fm.append(mm["fused"]["mse"])
                    ac.append(mm["argmax"]["cov"]); fc.append(mm["fused"]["cov"])
                    nn.append(mm["argmax"]["n"])
            if not am:
                continue
            g = lambda v, p=5: "%.*f+/-%.*f" % (p, st.mean(v), p, st.stdev(v) if len(v) > 1 else 0)
            print("   %5d %8.0f | %-19s %-19s | %-13s %-13s"
                  % (k, st.mean(nn) / LAST, g(am), g(fm), g(ac, 3), g(fc, 3)))

    print("\n=== GAMMA SPREAD ACROSS COVERING NODES (degeneracy) ===")
    print("   how far apart are contributors the fused rule treats as independent?")
    print("%-22s %5s %10s %12s %12s" % ("arm", "n_cov", "cells", "sd (deg)", "range (deg)"))
    for arm, lst in good.items():
        acc = defaultdict(lambda: ([], []))
        for seed, R in lst:
            for k, v in gamma_spread(R).items():
                acc[k][0].append(v["sd_deg"]); acc[k][1].append(v["range_deg"])
        for k in sorted(acc):
            sd, rg = acc[k]
            print("%-22s %5d %10s %12.4f %12.4f" % (arm, k, "", st.mean(sd), st.mean(rg)))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
