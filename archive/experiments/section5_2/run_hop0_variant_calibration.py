# Section 5.6 -- are the three width variants on a COMMON uncertainty scale?
#
# Readout only over stored arrays. No runs, no training path touched. This is
# the last calibration diagnostic.
#
# WHY HOP 0. Hop-0 nodes train on ground truth, so their reported uncertainty
# is not contaminated by another model's belief. If all three variants sit near
# hw:RMS 1.0 there, they are individually calibrated, nu is comparable across
# models, and the fusion weighting is tracking reliability. If they sit at
# materially different ratios, nu means different things in different models
# and the weighting is comparing different units.
#
# INSTRUMENT. Each node's OWN contributed belief is scored against truth --
# not the per-cell argmax winner. The question is about the individual models'
# self-reports, so the per-node quantity is the right one.
#
# TWO SUBSTITUTIONS, both checked below:
#   1. Contributor -> node id comes from runner.py:336-339 appending in node-id
#      order. Asserted exact (ncov == geometric coverage, filled slots == ncov,
#      slots are a gap-free prefix).
#   2. Per-node hop is not stored per step. The manifest's final_hop_distance
#      was tried first and REJECTED: it agrees with the stored per-step hop on
#      only 0.587 of n_cov==1 cells, so hop is not stable over the window and
#      the final snapshot selects the wrong nodes.
#      Instead hop 0 is reconstructed exactly from its definition --
#      Mesh.in_coverage (mesh.py:712-716) makes a node an anchor at step t iff
#      its fov_set contains a wearable cell at t -- using the stored
#      realised_wearable_paths. Validated below against cell_hop_steps on
#      n_cov==1 cells, where the winning node IS the unique coverer.
#
# Run: python -u experiments/section5_2/run_hop0_variant_calibration.py

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
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from analyse_estimators import truth_for, wrap
from beliefmesh.node.mesh import fov_cells

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
VARIANTS = ["narrow", "baseline", "wide"]
LAST = 50


def coverage_map(C, G, FOV=7):
    cover = defaultdict(list)
    for i, (cx, cy) in enumerate(C):
        for rc in fov_cells(int(cx), int(cy), FOV, G):
            cover[rc].append(i)
    return cover


def main():
    C = np.load(ENV / "node_centres.npy")
    runs = sorted(glob.glob("runs/chapter5_v2/s5_6_het/het_nig_product*/manifest.yaml"))
    per = {w: {v: defaultdict(list) for v in VARIANTS} for w in ("whole", "last50")}
    nodecount = defaultdict(list)
    hopcheck = []

    for p in runs:
        d = Path(os.path.dirname(p))
        m = yaml.safe_load(open(p))
        variants = m["node_variants"]
        seed = int(m["env_seed"])

        paths = np.load(d / "realised_wearable_paths.npy")
        bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
        nc = np.load(d / "cell_ncov_steps.npy").astype(int)
        cell_hop = np.load(d / "cell_hop_steps.npy")
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

        # --- reconstruct hop 0 per (node, step) from the wearable positions ---
        fov_set = {i: set(fov_cells(int(cx), int(cy), 7, G))
                   for i, (cx, cy) in enumerate(C)}
        is_anchor = np.zeros((T, len(C)), bool)
        for t in range(T):
            wcells = {(int(paths[w, t, 0]), int(paths[w, t, 1]))
                      for w in range(paths.shape[0])}
            for i in range(len(C)):
                if fov_set[i] & wcells:
                    is_anchor[t, i] = True

        # --- validate against the stored per-step hop at n_cov == 1 cells ---
        agree = tot = 0
        for rc, ids in cover.items():
            if len(ids) != 1:
                continue
            r, c = rc
            h = cell_hop[:, r, c]
            ok = np.isfinite(h)
            tot += int(ok.sum())
            agree += int(((h[ok] == 0) == is_anchor[ok, ids[0]]).sum())
        if tot:
            hopcheck.append(agree / tot)

        tr = truth_for(seed, T, G)
        g, nu, al, be = (bel[..., i] for i in range(4))

        n0 = defaultdict(int)
        for rc, ids in cover.items():
            r, c = rc
            for k, node_id in enumerate(ids):
                v = variants[node_id]
                for wname, sl in (("whole", slice(0, T)), ("last50", slice(T - LAST, T))):
                    gg, nn, aa, bb = (x[sl, r, c, k] for x in (g, nu, al, be))
                    # hop 0 is per (node, STEP), so it masks steps, not nodes
                    ok = (np.isfinite(gg) & np.isfinite(nn) & (aa > 1.0)
                          & is_anchor[sl, node_id])
                    if not ok.any():
                        continue
                    e = np.abs(wrap(gg[ok] - tr[sl, r, c][ok]))
                    sc = np.sqrt(bb[ok] * (1 + nn[ok]) / (nn[ok] * aa[ok]))
                    hw = sps.t.ppf(0.95, df=2 * aa[ok]) * sc
                    per[wname][v]["e2"].append(float(np.mean(e ** 2)))
                    per[wname][v]["hw"].append(float(np.mean(hw)))
                    per[wname][v]["cov"].append(float((e <= hw).mean()))
                    per[wname][v]["n"].append(int(ok.sum()))
        for i in range(len(C)):
            n0[variants[i]] += float(is_anchor[:, i].mean())
        for v in VARIANTS:
            nodecount[v].append(n0[v])

    print("=" * 96)
    print("SECTION 5.6 -- HOP-0 CALIBRATION PER WIDTH VARIANT")
    print("Heterogeneous nig_product mesh, %d seeds. Each node scored on its OWN" % len(runs))
    print("contributed belief against truth, over the cells it covers.")
    print("=" * 96)
    print("\nreconstruction check: is_anchor(node,step) == (stored cell hop == 0)")
    print("  at n_cov==1 cells, all steps: agreement %.4f (mean over seeds)"
          % st.mean(hopcheck))
    print("  (the manifest's final_hop_distance was tried first and rejected at 0.587)")
    print("\nfraction of steps at hop 0, summed over each variant's 12 nodes: " + ", ".join(
        "%s %.2f" % (v, st.mean(nodecount[v])) for v in VARIANTS))

    for wname, lab in (("whole", "WHOLE RUN"), ("last50", "LAST 50 STEPS")):
        print("\n" + "-" * 96)
        print("%s -- hop-0 nodes only" % lab)
        print("-" * 96)
        print("%-9s %9s %12s %12s %12s %14s"
              % ("variant", "node-cells", "RMS (deg)", "half-width",
                 "90% coverage", "hw : RMS"))
        vals = {}
        for v in VARIANTS:
            a = per[wname][v]
            if not a["e2"]:
                print("%-9s %9s" % (v, "--"))
                continue
            rms = np.sqrt(st.mean(a["e2"])) * 180
            hw = st.mean(a["hw"]) * 180
            cov = st.mean(a["cov"])
            vals[v] = (hw / rms, cov)
            print("%-9s %9d %12.2f %12.2f %12.3f %14.3f"
                  % (v, sum(a["n"]), rms, hw, cov, hw / rms))
        if len(vals) == 3:
            r = [vals[v][0] for v in VARIANTS]
            c = [vals[v][1] for v in VARIANTS]
            print("  hw:RMS spread across variants  : %.3f  (min %.3f, max %.3f)"
                  % (max(r) - min(r), min(r), max(r)))
            print("  coverage spread across variants: %.3f  (min %.3f, max %.3f)"
                  % (max(c) - min(c), min(c), max(c)))
            print("  ratio of largest to smallest hw:RMS: %.2fx" % (max(r) / min(r)))


if __name__ == "__main__":
    main()
