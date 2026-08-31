# Section 5.6 -- why the in-mesh variant ordering inverts the isolated one,
# and whether the fusion weighting actively costs accuracy.
#
# Readout only over stored arrays. No runs, no training path touched.
#
# ATTRIBUTION. cell_beliefs_steps has no node id, but runner.py:336-339 appends
# contributor beliefs while iterating mesh.nodes in id order, so slot k at a
# cell is the k-th LOWEST node id among that cell's geometric coverers. Three
# properties are asserted before anything is reported: ncov equals the
# geometric coverage everywhere, the non-NaN slot count equals ncov, and the
# filled slots are a gap-free prefix. If any fails the script stops.
#
# Run: python -u experiments/section5_2/run_variant_diagnostics.py

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
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from analyse_estimators import truth_for, wrap
from beliefmesh.node.mesh import fov_cells

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
R1D = Path("runs/section5_1/r1d/summary.yaml")
VARIANTS = ["narrow", "baseline", "wide"]
LAST = 50

MECHANISM = """
MECHANISM (for the write-up)
  Under the product rule the fused location is a precision-weighted circular
  mean, gamma* = sum_i (nu_i gamma_i) / sum_j nu_j, so contributor i's share of
  the fused target is exactly

        w_i  =  nu_i / sum_j nu_j

  nu is EVIDENCE THE NODE CLAIMS FOR ITSELF -- self-reported confidence -- not
  measured accuracy. Nothing in the rule consults how close gamma_i actually
  is to the truth. The mesh therefore converges toward whichever variant
  reports the LOWEST uncertainty, which need not be, and here is not, the
  variant that is most accurate. Every node is then trained toward that
  target, so the misallocation compounds rather than averaging out.
"""


def coverage_map(C, G, FOV=7):
    cover = defaultdict(list)
    for i, (cx, cy) in enumerate(C):
        for rc in fov_cells(int(cx), int(cy), FOV, G):
            cover[rc].append(i)                 # ascending node id
    return cover


def isolated():
    """R1d held-out numbers, 20-seed protocol. Read, not hardcoded."""
    d = yaml.safe_load(open(R1D))
    out = {}
    for p in d["points"]:
        if p["label"] in VARIANTS:
            out[p["label"]] = (p["circular_mse_norm_mean"], p["circular_mse_norm_sd"],
                               p["mean_epistemic_mean"])
    return out


def circ_mean(g, w, ok):
    z = np.where(ok, w * np.exp(1j * np.pi * np.nan_to_num(g)), 0).sum(-1)
    return np.angle(z) / np.pi


def main():
    C = np.load(ENV / "node_centres.npy")
    iso = isolated()
    runs = sorted(glob.glob("runs/chapter5_v2/s5_6_het/het_nig_product*/manifest.yaml"))

    acc = {w: defaultdict(lambda: defaultdict(list)) for w in ("whole", "last50")}
    mse_acc = defaultdict(lambda: defaultdict(list))
    # share-binned fused accuracy, and the equal-weight counterfactual
    QS = [(0.0, .25), (.25, .40), (.40, .55), (.55, 1.01)]
    binacc = {v: defaultdict(lambda: defaultdict(list)) for v in VARIANTS}
    overall = defaultdict(list)

    for p in runs:
        d = Path(os.path.dirname(p))
        m = yaml.safe_load(open(p))
        variants = m["node_variants"]
        seed = int(m["env_seed"])
        bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
        nc = np.load(d / "cell_ncov_steps.npy").astype(int)
        fused = np.load(d / "cell_fused_steps.npy").astype(np.float64)
        T, G = bel.shape[0], bel.shape[1]
        cover = coverage_map(C, G)

        geo = np.zeros((G, G), int)
        for rc, ids in cover.items():
            geo[rc] = len(ids)
        fin = np.isfinite(bel[..., 0])
        assert np.all(nc == geo[None]), "ncov != geometric coverage"
        assert np.all(fin.sum(-1) == nc), "filled slots != ncov"
        assert np.all(fin == (np.arange(bel.shape[3])[None, None, None, :]
                              < nc[..., None])), "slots not a gap-free prefix"

        g, nu, al, be = (bel[..., i] for i in range(4))
        with np.errstate(invalid="ignore", divide="ignore"):
            epi = be / (nu * (al - 1.0))
        nusum = np.nansum(np.where(fin, nu, 0.0), axis=-1, keepdims=True)
        wgt = np.where(fin, nu / nusum, np.nan)

        # per-variant share of the fused target at every (t, cell)
        vshare = {v: np.zeros(nc.shape) for v in VARIANTS}
        for rc, ids in cover.items():
            r, c = rc
            for k, node_id in enumerate(ids):
                vshare[variants[node_id]][:, r, c] += np.nan_to_num(wgt[:, r, c, k])

        tr = truth_for(seed, T, G)
        e2_actual = wrap(fused[..., 0] - tr) ** 2
        eq = np.where(fin, 1.0 / np.maximum(nc, 1)[..., None], 0.0)
        e2_equal = wrap(circ_mean(g, eq, fin) - tr) ** 2

        for wname, sl in (("whole", slice(0, T)), ("last50", slice(T - LAST, T))):
            ok = np.isfinite(e2_actual[sl]) & np.isfinite(e2_equal[sl]) & (nc[sl] >= 2)
            overall["actual_" + wname].append(float(np.mean(e2_actual[sl][ok])))
            overall["equal_" + wname].append(float(np.mean(e2_equal[sl][ok])))
            for v in VARIANTS:
                s = vshare[v][sl]
                for lo, hi in QS:
                    msk = ok & (s >= lo) & (s < hi)
                    if msk.sum() >= 200:
                        binacc[v][wname][(lo, hi)].append(
                            (float(np.mean(e2_actual[sl][msk])),
                             float(np.mean(e2_equal[sl][msk])),
                             int(msk.sum())))

        for rc, ids in cover.items():
            r, c = rc
            for k, node_id in enumerate(ids):
                v = variants[node_id]
                for wname, sl in (("whole", slice(0, T)), ("last50", slice(T - LAST, T))):
                    e, w_ = epi[sl, r, c, k], wgt[sl, r, c, k]
                    e, w_ = e[np.isfinite(e)], w_[np.isfinite(w_)]
                    if len(e):
                        acc[wname][v]["epi"].append(float(np.mean(e)))
                    if len(w_):
                        acc[wname][v]["wgt"].append(float(np.mean(w_)))
                    acc[wname][v]["ncov"].append(float(nc[sl, r, c].mean()))
        vm = m["results"]["variant_mse"]
        for v in VARIANTS:
            mse_acc["whole"][v].append(vm[v]["whole_run"])
            mse_acc["last50"][v].append(vm[v]["last50"])

    n = len(runs)
    print("=" * 104)
    print("SECTION 5.6 -- VARIANT ORDERING INSIDE THE MESH vs IN ISOLATION")
    print("nig_product heterogeneous mesh, %d seeds. Attribution asserted exact." % n)
    print("=" * 104)
    print("\nIsolated numbers are R1d held-out, 20-seed protocol (seeds 42-61),")
    print("read from runs/section5_1/r1d/summary.yaml. In-mesh numbers are")
    print("cell-space, from the heterogeneous runs' own manifests and beliefs.")

    for wname, lab in (("whole", "WHOLE RUN"), ("last50", "LAST 50 STEPS")):
        print("\n" + "-" * 104)
        print("%s" % lab)
        print("-" * 104)
        print("%-9s %12s %12s | %12s %12s %10s %12s"
              % ("variant", "isolated", "isolated", "in-mesh", "in-mesh",
                 "fusion", "weight vs"))
        print("%-9s %12s %12s | %12s %12s %10s %12s"
              % ("", "MSE", "epistemic", "MSE", "epistemic", "weight", "equal share"))
        for v in VARIANTS:
            a = acc[wname][v]
            w = st.mean(a["wgt"])
            rel = 100 * (w * st.mean(a["ncov"]) - 1)
            print("%-9s %12.5f %12.5f | %12.5f %12.5f %10.4f %+11.1f%%"
                  % (v, iso[v][0], iso[v][2], st.mean(mse_acc[wname][v]),
                     st.mean(a["epi"]), w, rel))
        rank = lambda d_: " < ".join(sorted(VARIANTS, key=lambda v: d_[v]))
        print("  ranking by isolated MSE : %s"
              % rank({v: iso[v][0] for v in VARIANTS}))
        print("  ranking by in-mesh MSE  : %s"
              % rank({v: st.mean(mse_acc[wname][v]) for v in VARIANTS}))
        print("  ranking by fusion weight: %s"
              % " > ".join(sorted(VARIANTS,
                                  key=lambda v: -st.mean(acc[wname][v]["wgt"]))))

    print(MECHANISM)

    print("=" * 104)
    print("DOES THE WEIGHTING COST ACCURACY?  fused error where each variant")
    print("holds a large vs small share, and the equal-weight counterfactual")
    print("=" * 104)
    print("Cells with n_cov >= 2 only. RMS in degrees. 'equal' re-fuses the SAME")
    print("stored beliefs with w_i = 1/n instead of nu_i/sum(nu), changing only")
    print("the weighting, so the difference isolates what confidence-weighting buys.\n")
    for wname, lab in (("whole", "WHOLE RUN"), ("last50", "LAST 50 STEPS")):
        a = np.sqrt(st.mean(overall["actual_" + wname])) * 180
        e = np.sqrt(st.mean(overall["equal_" + wname])) * 180
        print("%-14s  nu-weighted RMS %6.2f deg   equal-weight RMS %6.2f deg   "
              "confidence weighting is %+.2f deg" % (lab, a, e, a - e))
    for v in VARIANTS:
        print("\n  fused RMS (deg) by %s's share of the fused target:" % v.upper())
        print("  %-14s %10s %12s %12s %10s" % ("share", "cells/seed", "nu-weighted",
                                               "equal-weight", "difference"))
        for lo, hi in QS:
            rows = binacc[v]["last50"].get((lo, hi))
            if not rows:
                continue
            aa = np.sqrt(st.mean([r[0] for r in rows])) * 180
            ee = np.sqrt(st.mean([r[1] for r in rows])) * 180
            print("  %-14s %10d %12.2f %12.2f %+10.2f"
                  % ("%.2f-%.2f" % (lo, min(hi, 1.0)),
                     st.mean([r[2] for r in rows]), aa, ee, aa - ee))
    print("\n  (last-50 window. A POSITIVE difference means nu-weighting is WORSE")
    print("   than weighting every contributor equally in that stratum.)")


if __name__ == "__main__":
    main()
