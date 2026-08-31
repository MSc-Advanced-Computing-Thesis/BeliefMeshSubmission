# Which argmax-era diagnostics are convention-dependent?
#
# Readout only. No training path touched.
#
# RECOMPUTED HERE (both conventions, side by side):
#   1. recency breakdown -- hw:RMS and cov90 by timesteps since last hop-0
#   2. per-hop hw:RMS across the four 5.4.1 density geometries
#
# NOTE ON THE RECENCY / HOP VARIABLE ITSELF. cell_hop_steps records the hop of
# the node whose belief WON each cell under argmax. Under the averaged
# convention no single node wins, so that variable is a legacy of the old rule.
# Both definitions are therefore reported:
#     winner-hop  : cell_hop_steps == 0   (comparable to the published tables)
#     anchor-cov  : the cell is covered by >= 1 anchor node at that step,
#                   reconstructed from realised_wearable_paths via
#                   Mesh.in_coverage's definition -- convention-free, and
#                   validated elsewhere at 1.0000 agreement on n_cov==1 cells.
#
# Run: python -u experiments/section5_2/run_convention_recheck.py

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
sys.path.insert(0, str(HERE.parents[1] / "src"))
from averaged_readout import load_run
from beliefmesh.node.mesh import fov_cells

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
LAST = 50
RBINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 6)]
HOPS = [0, 1, 2, 3]
MIN_CELLS = 40


def rlab(lo, hi):
    return "%d" % lo if lo == hi else ("%d-%d" % (lo, hi) if hi < 10 ** 6 else "%d+" % lo)


def stat(R, arm, mask):
    e2, hw = R[arm]
    T = R["T"]
    sl = slice(T - LAST, T)
    ok = np.isfinite(e2[sl]) & np.isfinite(hw[sl]) & mask[sl]
    if ok.sum() < MIN_CELLS:
        return None
    m = float(np.mean(e2[sl][ok]))
    return dict(n=int(ok.sum()),
                cov=float((np.sqrt(e2[sl][ok]) <= hw[sl][ok]).mean()),
                ratio=float(np.mean(hw[sl][ok]) * 180 / (np.sqrt(m) * 180)))


def recency_from(mask0):
    """timesteps since mask0 was last true, per cell."""
    T, G = mask0.shape[0], mask0.shape[1]
    rec = np.full(mask0.shape, np.nan)
    last = np.full((G, G), -1)
    for t in range(T):
        last[mask0[t]] = t
        seen = last >= 0
        rec[t][seen] = t - last[seen]
    return rec


def anchor_cover(d, T, G):
    """Cell is covered by >= 1 anchor (hop-0) node at step t."""
    C = np.load(ENV / "node_centres.npy")
    paths = np.load(Path(d) / "realised_wearable_paths.npy")
    fov = {i: fov_cells(int(cx), int(cy), 7, G) for i, (cx, cy) in enumerate(C)}
    fset = {i: set(v) for i, v in fov.items()}
    out = np.zeros((T, G, G), bool)
    for t in range(T):
        wc = {(int(paths[w, t, 0]), int(paths[w, t, 1])) for w in range(paths.shape[0])}
        for i in range(len(C)):
            if fset[i] & wc:
                for (r, c) in fov[i]:
                    out[t, r, c] = True
    return out


def load(pat):
    out = []
    for p in sorted(glob.glob(pat)):
        d = Path(os.path.dirname(p))
        m = yaml.safe_load(open(p))
        R = load_run(d, int(m["env_seed"]))
        R["hop"] = np.load(d / "cell_hop_steps.npy")
        R["dir"] = d
        R["meta"] = m
        out.append(R)
    return out


def item1(runs):
    print("=" * 100)
    print("ITEM 1 -- RECENCY BREAKDOWN, both conventions (5.3.1 mesh, 5 seeds)")
    print("=" * 100)
    for defn in ("winner-hop", "anchor-cov"):
        print("\n--- recency defined by %s ---" % defn)
        print("%18s %14s %12s %14s %12s"
              % ("steps since hop 0", "argmax hw:RMS", "argmax cov",
                 "averaged hw:RMS", "averaged cov"))
        acc = defaultdict(lambda: defaultdict(list))
        for R in runs:
            T, G = R["T"], R["hop"].shape[1]
            base = (R["hop"] == 0) if defn == "winner-hop" else anchor_cover(R["dir"], T, G)
            rec = recency_from(base)
            for b in RBINS:
                m = np.isfinite(rec) & (rec >= b[0]) & (rec <= b[1])
                for arm in ("argmax", "averaged"):
                    s = stat(R, arm, m)
                    if s:
                        acc[b][arm].append(s)
        for b in RBINS:
            if not acc[b].get("argmax"):
                continue
            print("%18s %14.2f %12.3f %14.2f %12.3f"
                  % (rlab(*b),
                     st.mean([x["ratio"] for x in acc[b]["argmax"]]),
                     st.mean([x["cov"] for x in acc[b]["argmax"]]),
                     st.mean([x["ratio"] for x in acc[b]["averaged"]]),
                     st.mean([x["cov"] for x in acc[b]["averaged"]])))


def item2():
    print("\n" + "=" * 100)
    print("ITEM 2 -- PER-HOP hw:RMS ACROSS THE FOUR 5.4.1 GEOMETRIES, both conventions")
    print("=" * 100)
    geo = defaultdict(list)
    for p in sorted(glob.glob("runs/chapter5_v2/s5_4_1_overlap/*/manifest.yaml")):
        m = yaml.safe_load(open(p))
        geo[(m["mean_cell_coverage"], m["n_nodes"])].append(p)
    for arm in ("argmax", "averaged"):
        print("\n--- %s ---" % arm.upper())
        print("%9s %6s" % ("mean_cov", "nodes")
              + "".join("%10s" % ("hop %d" % h) for h in HOPS) + "%12s" % "aggregate")
        for k in sorted(geo):
            rs = []
            for p in geo[k]:
                d = Path(os.path.dirname(p))
                mm = yaml.safe_load(open(p))
                R = load_run(d, int(mm["env_seed"]))
                R["hop"] = np.load(d / "cell_hop_steps.npy")
                rs.append(R)
            row = "%9.2f %6d" % k
            for h in HOPS:
                v = [stat(R, arm, R["hop"] == h) for R in rs]
                v = [x["ratio"] for x in v if x]
                row += "%10s" % ("%.2f" % st.mean(v) if v else "--")
            agg = [stat(R, arm, np.ones_like(R["hop"], bool)) for R in rs]
            agg = [x["ratio"] for x in agg if x]
            print(row + "%12.2f" % st.mean(agg))


if __name__ == "__main__":
    item1(load("runs/chapter5_v2/s531_b2/nig_product_sampled_seed*/manifest.yaml"))
    item2()
