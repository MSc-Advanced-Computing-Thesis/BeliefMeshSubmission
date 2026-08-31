# Section 5.6 -- spatial layout of the width variants, and whether position
# rather than capacity could carry the per-variant MSE ordering.
#
# Readout only over stored arrays and manifests. No runs, no training path.
#
# node_centres.npy is COLUMN-major: index i -> column 3+3*(i//6),
# row 3+3*(i%6). The node grid is therefore 6x6 with grid coords
# (gcol, grow) = (i//6, i%6).
#
# Run: python -u experiments/section5_2/run_variant_layout.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field
from beliefmesh.node.mesh import fov_cells

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
VARIANTS = ["narrow", "baseline", "wide"]
SYM = {"narrow": "n", "baseline": "b", "wide": "W"}
GD = 6
FOV = 7


def grid(variants):
    """6x6 array of variant symbols, rows = grid row, cols = grid column."""
    a = np.empty((GD, GD), dtype="<U1")
    for i, v in enumerate(variants):
        a[i % GD, i // GD] = SYM[v]
    return a


def same_neighbour_rate(variants):
    """Fraction of 4-connected node-grid pairs sharing a variant."""
    same = tot = 0
    for i, v in enumerate(variants):
        gc, gr = i // GD, i % GD
        for dc, dr in ((1, 0), (0, 1)):
            nc_, nr_ = gc + dc, gr + dr
            if nc_ < GD and nr_ < GD:
                j = nc_ * GD + nr_
                tot += 1
                same += int(variants[j] == v)
    return same / tot


def main():
    C = np.load(ENV / "node_centres.npy")
    runs = sorted(glob.glob("runs/chapter5_v2/s5_6_het/het_nig_product*/manifest.yaml"))
    layouts, seeds = [], []
    for p in runs:
        m = yaml.safe_load(open(p))
        layouts.append(m["node_variants"])
        seeds.append(int(m["env_seed"]))

    print("=" * 92)
    print("SECTION 5.6 -- VARIANT SPATIAL ASSIGNMENT  (%d seeds)" % len(runs))
    print("=" * 92)

    identical = all(l == layouts[0] for l in layouts)
    print("\nassignment identical across all seeds? %s" % identical)
    pair = [(i, j) for i in range(len(layouts)) for j in range(i + 1, len(layouts))]
    ov = [sum(a == b for a, b in zip(layouts[i], layouts[j])) / len(layouts[i])
          for i, j in pair]
    print("pairwise agreement between seed layouts: mean %.3f, min %.3f, max %.3f"
          % (st.mean(ov), min(ov), max(ov)))
    print("  (chance agreement for 3 equal classes is 1/3 = 0.333)")

    print("\nnode grid per seed (rows = grid row, cols = grid column;")
    print("n=narrow, b=baseline, W=wide):")
    for s, l in zip(seeds, layouts):
        g = grid(l)
        print("  seed %-5d  " % s + "   ".join("".join(g[r]) for r in range(GD)))

    rates = [same_neighbour_rate(l) for l in layouts]
    rng = np.random.default_rng(0)
    null = []
    for _ in range(4000):
        x = list(layouts[0])
        rng.shuffle(x)
        null.append(same_neighbour_rate(x))
    lo, hi = np.percentile(null, [2.5, 97.5])
    print("\nBLOCKED vs INTERLEAVED vs RANDOM")
    print("  same-variant 4-neighbour rate: observed mean %.3f (per seed %s)"
          % (st.mean(rates), ", ".join("%.2f" % r for r in rates)))
    print("  permutation null: mean %.3f, 95%% interval [%.3f, %.3f]"
          % (st.mean(null), lo, hi))
    verdict = ("BLOCKED (clustered)" if st.mean(rates) > hi else
               "INTERLEAVED (anti-clustered)" if st.mean(rates) < lo else
               "consistent with RANDOM")
    print("  -> %s" % verdict)

    # ---------------- contributor composition per cell ----------------
    G = 22
    cover = defaultdict(list)
    for i, (cx, cy) in enumerate(C):
        for rc in fov_cells(int(cx), int(cy), FOV, G):
            cover[rc].append(i)

    comp = defaultdict(list)
    same_ctr, ncov_all, dom = [], [], []
    domcount = Counter()
    for l in layouts:
        for rc, ids in cover.items():
            vs = [l[i] for i in ids]
            c = Counter(vs)
            n = len(ids)
            ncov_all.append(n)
            for v in VARIANTS:
                comp[v].append(c[v])
            # expected same-variant partners for a randomly chosen contributor
            same_ctr.append(sum(c[v] * (c[v] - 1) for v in VARIANTS) / n)
            top, topn = c.most_common(1)[0]
            dom.append(topn / n)
            domcount[top] += 1

    print("\n" + "=" * 92)
    print("CONTRIBUTOR COMPOSITION PER CELL  (all cells, %d seeds pooled)" % len(layouts))
    print("=" * 92)
    print("  mean contributors per cell (n_cov)      : %.2f" % st.mean(ncov_all))
    print("  mean contributors of each variant per cell:")
    for v in VARIANTS:
        print("      %-9s %.2f   (%.1f%% of the contributor set)"
              % (v, st.mean(comp[v]), 100 * st.mean(comp[v]) / st.mean(ncov_all)))
    print("  mean SAME-VARIANT partners for a contributor: %.2f of %.2f others"
          % (st.mean(same_ctr), st.mean(ncov_all) - 1))
    print("  mean share held by the cell's most common variant: %.3f" % st.mean(dom))
    frac_mixed = np.mean([d < 0.5 for d in dom])
    frac_single = np.mean([d >= 0.75 for d in dom])
    print("  cells where NO variant holds a majority      : %.1f%%" % (100 * frac_mixed))
    print("  cells where one variant holds >= 75%%         : %.1f%%" % (100 * frac_single))
    print("  -> a typical cell is fused from %s"
          % ("a MIXTURE" if st.mean(dom) < 0.55 else "MOSTLY ONE VARIANT"))

    # ---------------- positional exposure per variant ----------------
    print("\n" + "=" * 92)
    print("POSITIONAL EXPOSURE PER VARIANT  (does position confound capacity?)")
    print("=" * 92)
    field = np.abs(np.asarray(build_dynamic_offset_field(G, 390)))
    fovs = {i: fov_cells(int(cx), int(cy), FOV, G) for i, (cx, cy) in enumerate(C)}
    ncov_map = np.zeros((G, G), int)
    for rc, ids in cover.items():
        ncov_map[rc] = len(ids)

    exposure = {v: defaultdict(list) for v in VARIANTS}
    for p, l in zip(runs, layouts):
        d = Path(os.path.dirname(p))
        paths = np.load(d / "realised_wearable_paths.npy")
        visits = np.zeros((G, G), int)
        for w in range(paths.shape[0]):
            for t in range(paths.shape[1]):
                visits[int(paths[w, t, 0]), int(paths[w, t, 1])] += 1
        for i, v in enumerate(l):
            cells = fovs[i]
            rs = [c[0] for c in cells]; cs = [c[1] for c in cells]
            exposure[v]["ncov"].append(float(np.mean(ncov_map[rs, cs])))
            exposure[v]["visits"].append(float(visits[rs, cs].sum()))
            exposure[v]["offset"].append(float(field[:, rs, cs].mean()))
            exposure[v]["edge"].append(float(np.mean(
                [min(r, G - 1 - r, c, G - 1 - c) for r, c in cells])))
    print("%-9s %14s %16s %16s %14s"
          % ("variant", "mean n_cov", "wearable visits", "mean |offset|", "edge distance"))
    for v in VARIANTS:
        e = exposure[v]
        print("%-9s %14.3f %16.1f %16.2f %14.3f"
              % (v, st.mean(e["ncov"]), st.mean(e["visits"]),
                 st.mean(e["offset"]), st.mean(e["edge"])))
    for key, lab in (("ncov", "n_cov"), ("visits", "wearable visits"),
                     ("offset", "mean |offset|"), ("edge", "edge distance")):
        vals = [st.mean(exposure[v][key]) for v in VARIANTS]
        sds = [st.stdev(exposure[v][key]) for v in VARIANTS]
        spread = max(vals) - min(vals)
        print("  %-16s spread across variants %8.3f   vs within-variant sd %8.3f  -> %s"
              % (lab, spread, st.mean(sds),
                 "MATERIAL" if spread > 0.5 * st.mean(sds) else "negligible"))


if __name__ == "__main__":
    main()
