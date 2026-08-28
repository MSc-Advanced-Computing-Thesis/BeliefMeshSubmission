# Analysis of the sampled-target mesh arms. Pure recomputation from the dumped
# cell_* arrays; no model, no re-evaluation.
#
# Interval coverage is computed from the NEW cell_nig_steps array -- the whole
# reason that array exists. Coverage uses the Student-t predictive of the same
# best-certainty belief cell_mse_steps records: location gamma (implicit in the
# stored squared error), df 2*alpha, scale sqrt(beta*(1+nu)/(nu*alpha)), tested
# on the WRAPPED error, which cell_mse_steps already is.
#
# Run: python -u experiments/section5_2/analyse_mesh_sampled.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats as sps
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
from beliefmesh.node.mesh import fov_cells

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
OUT = Path("runs/section5_2/mesh_sampled")
SEEDS = [42, 1042, 2042, 3042, 4042]
LAST = 50
ARMS = [("nig_product", "unsampled"), ("nig_product", "sampled"),
        ("naive", "unsampled"), ("naive", "sampled")]
HALF_PERIOD = 1.0          # normalised angle domain [-1, 1), period 2


def coverage_count_map(G=22, fov=7):
    k = np.zeros((G, G), dtype=int)
    for cx, cy in np.load(ENV / "node_centres.npy"):
        for r, c in fov_cells(int(cx), int(cy), fov, G):
            k[r, c] += 1
    return k


def load(mode, suffix, seed):
    d = OUT / f"{mode}_{suffix}_seed{seed}"
    if not (d / "cell_nig_steps.npy").exists():
        return None
    return {
        "mse": np.load(d / "cell_mse_steps.npy"),
        "cert": np.load(d / "cell_cert_steps.npy"),
        "nig": np.load(d / "cell_nig_steps.npy"),
        "hop": np.load(d / "cell_hop_steps.npy") if (d / "cell_hop_steps.npy").exists() else None,
    }


def coverage(mse, nig, mask=None, level=0.90, window=LAST):
    """Empirical coverage of the Student-t predictive interval, wrapped."""
    T = mse.shape[0]
    sl = slice(T - window, T)
    e = np.sqrt(mse[sl])                       # |wrapped error|, normalised
    nu, al, be = nig[sl, ..., 0], nig[sl, ..., 1], nig[sl, ..., 2]
    ok = np.isfinite(e) & np.isfinite(nu) & np.isfinite(al) & (al > 1.0)
    if mask is not None:
        ok &= np.broadcast_to(mask, ok.shape)
    if not ok.any():
        return {"empirical": np.nan, "n": 0, "half_width_deg": np.nan,
                "frac_covers_circle": np.nan}
    scale = np.sqrt(be[ok] * (1.0 + nu[ok]) / (nu[ok] * al[ok]))
    half = sps.t.ppf(0.5 + level / 2.0, df=2.0 * al[ok]) * scale
    return {"empirical": float((e[ok] <= half).mean()), "n": int(ok.sum()),
            "half_width_deg": float(np.mean(half) * 180.0),
            "frac_covers_circle": float((half >= HALF_PERIOD).mean())}


def cert_mse_r(mse, cert, mask=None, window=LAST):
    """Per-timestep cross-cell convention: correlate within each step, average."""
    T = mse.shape[0]
    rs = []
    for t in range(T - window, T):
        a, b = mse[t], cert[t]
        ok = np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            ok &= mask
        if ok.sum() >= 8 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            rs.append(pearsonr(b[ok], a[ok])[0])
    return float(np.mean(rs)) if rs else np.nan


def mstat(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def main():
    k = coverage_count_map()
    ks = sorted(int(v) for v in np.unique(k))
    out = {"convention": "cell space; last %d steps; 5 seeds; coverage from "
                         "cell_nig_steps Student-t, wrapped containment" % LAST,
           "arms": {}, "by_k": {}, "by_hop": {}}

    for mode, suf in ARMS:
        name = f"{mode}_{suf}"
        runs = [r for s in SEEDS if (r := load(mode, suf, s))]
        if not runs:
            continue
        out["arms"][name] = {
            "n_seeds": len(runs),
            "whole_run_mse": mstat([float(np.nanmean(r["mse"])) for r in runs]),
            "last50_mse": mstat([float(np.nanmean(r["mse"][-LAST:])) for r in runs]),
            "coverage_90": mstat([coverage(r["mse"], r["nig"])["empirical"] for r in runs]),
            "half_width_deg": mstat([coverage(r["mse"], r["nig"])["half_width_deg"] for r in runs]),
            "frac_covers_circle": mstat([coverage(r["mse"], r["nig"])["frac_covers_circle"] for r in runs]),
            "cert_mse_r": mstat([cert_mse_r(r["mse"], r["cert"]) for r in runs]),
        }
        out["by_k"][name] = {}
        for kv in ks:
            m = (k == kv)
            if m.sum() < 8:
                continue
            out["by_k"][name][kv] = {
                "n_cells": int(m.sum()),
                "mse": mstat([float(np.nanmean(r["mse"][-LAST:][:, m])) for r in runs]),
                "coverage_90": mstat([coverage(r["mse"], r["nig"], m)["empirical"] for r in runs]),
                "half_width_deg": mstat([coverage(r["mse"], r["nig"], m)["half_width_deg"] for r in runs]),
                "cert_mse_r": mstat([cert_mse_r(r["mse"], r["cert"], m) for r in runs]),
            }
        if runs[0]["hop"] is not None:
            out["by_hop"][name] = {}
            for h in range(4):
                per_run_mse, per_run_cov, per_run_hw, ns = [], [], [], []
                for r in runs:
                    hm = (r["hop"][-LAST:] == h)
                    if hm.sum() < 8:
                        continue
                    sub = r["mse"][-LAST:]
                    per_run_mse.append(float(np.nanmean(sub[hm])))
                    e = np.sqrt(sub[hm])
                    nig = r["nig"][-LAST:]
                    nu, al, be = nig[..., 0][hm], nig[..., 1][hm], nig[..., 2][hm]
                    ok = np.isfinite(e) & np.isfinite(al) & (al > 1.0)
                    if ok.sum() >= 8:
                        sc = np.sqrt(be[ok] * (1 + nu[ok]) / (nu[ok] * al[ok]))
                        hw = sps.t.ppf(0.95, df=2 * al[ok]) * sc
                        per_run_cov.append(float((e[ok] <= hw).mean()))
                        per_run_hw.append(float(np.mean(hw) * 180.0))
                    ns.append(int(hm.sum()))
                if per_run_mse:
                    out["by_hop"][name][h] = {
                        "mse": mstat(per_run_mse), "coverage_90": mstat(per_run_cov),
                        "half_width_deg": mstat(per_run_hw),
                        "mean_cells_per_run": float(np.mean(ns))}

    with open(OUT / "analysis.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)

    def row(d, key):
        s = d.get(key) or {}
        return "%.5f+/-%.5f" % (s["mean"], s["sd"]) if s.get("mean") is not None else "     --      "

    print("\n=== ARM SUMMARY (last %d steps, cell space) ===" % LAST)
    print("%-24s %-3s %-20s %-20s %-18s %-10s %-18s"
          % ("arm", "n", "whole-run", "last50", "cov90", "half-w deg", "cert-MSE r"))
    for name, a in out["arms"].items():
        print("%-24s %-3d %-20s %-20s %-18s %-10.2f %-18s"
              % (name, a["n_seeds"], row(a, "whole_run_mse"), row(a, "last50_mse"),
                 row(a, "coverage_90"), a["half_width_deg"]["mean"], row(a, "cert_mse_r")))

    print("\n=== BY COVERAGE COUNT k ===")
    for name, byk in out["by_k"].items():
        print(" %s" % name)
        print("   %3s %6s %-20s %-18s %-10s %-18s"
              % ("k", "cells", "MSE", "cov90", "half-w", "cert-MSE r"))
        for kv, d in byk.items():
            print("   %3d %6d %-20s %-18s %-10.2f %-18s"
                  % (kv, d["n_cells"], row(d, "mse"), row(d, "coverage_90"),
                     d["half_width_deg"]["mean"], row(d, "cert_mse_r")))

    print("\n=== BY HOP DISTANCE ===")
    for name, byh in out["by_hop"].items():
        print(" %s" % name)
        for h, d in byh.items():
            print("   hop %d  cells/run %6.0f  MSE %-20s cov90 %-18s half-w %.2f"
                  % (h, d["mean_cells_per_run"], row(d, "mse"),
                     row(d, "coverage_90"), d["half_width_deg"]["mean"]))
    print("\nwrote %s" % (OUT / "analysis.yaml"))


if __name__ == "__main__":
    main()
