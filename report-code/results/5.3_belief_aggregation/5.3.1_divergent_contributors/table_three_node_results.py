# Section 5.3 colour three-node V4 results. Readout only.
#
# Reported TWICE: over the student's full 49-cell FOV, and over the 16
# TWO-CONTRIBUTOR cells (C n A n B). The second is the one that tests the
# rules -- a single-contributor cell has nothing to aggregate, so every arm
# returns the same belief there by construction.
#
# Run: python -u experiments/section5_2/run_colour3_v4_results.py

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import glob
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

from _shared.arm_names import name as an
from _shared.averaged_readout import averaged_params, load_run, metrics, wrap_np
from _shared.truth import truth_for_run
from _shared.artefacts import load_array
import experiment_three_node_colour_world as V4
from beliefmesh.node.mesh import fov_cells
from beliefmesh.simulation.cert_mse_metrics import per_timestep_then_averaged

G, LAST = V4.G, 50
a, b, c = V4.regions()
TWO = np.array(sorted(c & a & b))              # 16 cells
SUP = np.array(sorted(c & (a | b)))            # 28 cells
ARMS = V4.ARMS
ROOT = ART + "/5.3_belief_aggregation/5.3.1_divergent_contributors/three_node_colour_world"


def mask_of(cells):
    m = np.zeros((G, G), bool)
    for r, cc in cells:
        m[r, cc] = True
    return m


CMASK, TMASK = mask_of(c), mask_of(map(tuple, TWO))


def slots_for(cell):
    return [i for i, s in enumerate((a, b, c)) if cell in s]


def one(d, seed):
    d = Path(d)
    R = load_run(d, seed)
    T = R["T"]
    bel = load_array(d, "cell_beliefs_steps").astype(np.float64)
    nc = load_array(d, "cell_ncov_steps").astype(int)
    gam, nu_s, al_s, be_s = averaged_params(bel, nc)
    epi = be_s / (np.maximum(nu_s, 1e-6) * np.maximum(al_s - 1.0, 1e-6))
    cert = 1.0 / (1.0 + np.minimum(epi, 10.0))
    tr = truth_for_run(d, seed, T, G)

    out = {}
    for tag, msk in (("fov", CMASK), ("two", TMASK)):
        m = metrics(R, last=LAST, mask=msk)["averaged"]
        # metrics() computes its whole-run figure over ALL cells and ignores the
        # mask, which would make this column identical in both tables. Recompute
        # it restricted to the region.
        e2 = R["averaged"][0]
        m["whole"] = float(np.nanmean(e2[:, msk]))
        r_cert, _, _, _ = per_timestep_then_averaged(
            np.where(msk[None], R["averaged"][0], np.nan),
            np.where(msk[None], cert, np.nan), T - LAST, T)
        out[tag] = dict(m, r_cert=r_cert)

    # r(nu, error) over the ANCHOR contributors on the two-contributor cells
    sub = bel[-LAST:][:, TWO[:, 0], TWO[:, 1], :, :]
    nu, g = sub[..., 1], sub[..., 0]
    err = np.abs(wrap_np(g - tr[-LAST:][:, TWO[:, 0], TWO[:, 1]][..., None])) * 180
    fin = np.isfinite(nu) & np.isfinite(err)
    anch = np.zeros(nu.shape, bool)
    for k, cell in enumerate(map(tuple, TWO)):
        for j, nid in enumerate(slots_for(cell)):
            if nid in (0, 1):
                anch[:, k, j] = True
    ma = fin & anch
    out["two"]["r_anch"] = float(np.corrcoef(nu[ma], err[ma])[0, 1])
    # and over the full supervised region, for continuity with v3
    sub2 = bel[-LAST:][:, SUP[:, 0], SUP[:, 1], :, :]
    nu2, g2 = sub2[..., 1], sub2[..., 0]
    err2 = np.abs(wrap_np(g2 - tr[-LAST:][:, SUP[:, 0], SUP[:, 1]][..., None])) * 180
    fin2 = np.isfinite(nu2) & np.isfinite(err2)
    anch2 = np.zeros(nu2.shape, bool)
    for k, cell in enumerate(map(tuple, SUP)):
        for j, nid in enumerate(slots_for(cell)):
            if nid in (0, 1):
                anch2[:, k, j] = True
    m2 = fin2 & anch2
    out["fov"]["r_anch"] = float(np.corrcoef(nu2[m2], err2[m2])[0, 1])

    # contributor gamma spread on the two-contributor cells, vs RMS there
    gf = np.isfinite(g)
    x = np.where(gf, wrap_np(g - g[..., 0][..., None]), np.nan)
    n = gf.sum(-1)
    ok = n >= 2
    out["gspread"] = float(np.nanmean(
        np.where(ok, np.nanmax(x, -1) - np.nanmin(x, -1), np.nan))) * 180
    out["rms_two"] = float(np.sqrt(out["two"]["last50"])) * 180
    return out




def main(csv_path=None):
    """Print the three-node results; write the two-contributor-cell table
    (report Table 5.1) to csv_path when given."""
    D = {}
    for arm in ARMS:
        V = []
        for p in sorted(glob.glob("%s/%s_seed*/manifest.yaml" % (ROOT, arm))):
            d = Path(os.path.dirname(p))
            V.append(one(d, int(yaml.safe_load(open(p))["env_seed"])))
        if V:
            D[arm] = V

    ms = lambda V, f: (st.mean([f(v) for v in V]),
                       st.stdev([f(v) for v in V]) if len(V) > 1 else 0.0)

    for tag, title, n_cells in (("fov", "STUDENT'S FULL FIELD OF VIEW", 49),
                                ("two", "TWO-CONTRIBUTOR CELLS ONLY", len(TWO))):
        print("=" * 108)
        print("V4 -- %s (%d cells), averaged readout, 5 seeds" % (title, n_cells))
        print("=" * 108)
        print("%-16s %3s %-20s %-20s %-15s %-15s %-16s"
              % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage",
                 "hw : RMS", "r(nu, error)"))
        for arm in ARMS:
            if arm not in D:
                continue
            V = D[arm]
            f = lambda k, p=5: "%.*f +/- %.*f" % (
                p, ms(V, lambda v: v[tag][k])[0], p, ms(V, lambda v: v[tag][k])[1])
            print("%-16s %3d %-20s %-20s %-15s %-15s %-16s"
                  % (an(arm), len(V), f("whole"), f("last50"), f("cov", 3),
                     f("ratio", 3), "%+.3f +/- %.3f" % ms(V, lambda v: v[tag]["r_anch"])))
        print()

    print("=" * 108)
    print("CONTRIBUTOR DISAGREEMENT ON THE TWO-CONTRIBUTOR CELLS")
    print("=" * 108)
    print("%-16s %22s %22s %14s" % ("arm", "gamma spread (deg)", "last-50 RMS (deg)",
                                    "spread / RMS"))
    for arm in ARMS:
        if arm not in D:
            continue
        V = D[arm]
        gs = ms(V, lambda v: v["gspread"])
        rm = ms(V, lambda v: v["rms_two"])
        print("%-16s %22s %22s %14.2f"
              % (an(arm), "%.2f +/- %.2f" % gs, "%.2f +/- %.2f" % rm,
                 gs[0] / rm[0] if rm[0] else float("nan")))
    print()
    print("  v3 reference: gamma spread 3.90 deg against a last-50 RMS of ~4 deg")

    if csv_path is not None:
        import csv
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "n_seeds", "whole_run_mse_mean", "whole_run_mse_sd",
                        "last50_mse_mean", "last50_mse_sd", "coverage90_mean", "coverage90_sd",
                        "hw_rms_mean", "hw_rms_sd", "r_nu_error_mean", "r_nu_error_sd"])
            for arm in ARMS:
                if arm not in D:
                    continue
                V = D[arm]
                row = [an(arm), len(V)]
                for k in ("whole", "last50", "cov", "ratio"):
                    row += list(ms(V, lambda v, k=k: v["two"][k]))
                row += list(ms(V, lambda v: v["two"]["r_anch"]))
                w.writerow(row)
        print("  wrote %s" % csv_path)


if __name__ == "__main__":
    main()
