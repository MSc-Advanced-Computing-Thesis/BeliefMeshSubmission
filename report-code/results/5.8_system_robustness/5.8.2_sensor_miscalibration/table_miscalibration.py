# Miscalibrated wearable: results against the matched five-seed baseline.
#
# Averaged NIG readout, cell space, last-50 window. Baseline is
# s531_b2/nig_product_sampled_seed*, which shares every setting except the
# bias -- same seeds, same wearable paths, same field, same exclusions.
#
# THE THREE ITEMS:
#   1. headline metrics vs baseline
#   2. per-cell last-50 MSE difference binned by distance from the corrupted
#      wearable's realised path -- does the damage stay local or propagate?
#   3. certainty in the affected region vs baseline -- does the mesh report
#      lower confidence where it was fed bad data, or absorb it confidently?
#
# Item 3 decides whether the uncertainty is usable as a corruption signal.
#
# Run: python -u experiments/section5_2/run_miscalibrated_results.py

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

from _shared.averaged_readout import averaged_params, load_run, metrics

LAST, G = 50, 22
SEEDS = [42, 1042, 2042, 3042, 4042]
MISCAL = Path(ART + "/5.8_system_robustness/5.8.2_sensor_miscalibration/miscalibrated_wearable")
BASE = Path(ART + "/5.3_belief_aggregation/5.3.2_mesh_scale/aggregation_arms")
# distance bins in cells, from the corrupted wearable's realised path
BINS = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 99)]


def dirs(seed):
    return (MISCAL / ("miscal_seed%d" % seed),
            BASE / ("nig_product_sampled_seed%d" % seed))


def certainty(d, seed):
    """Averaged-readout epistemic -> certainty, same form the pipeline uses."""
    bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(d / "cell_ncov_steps.npy").astype(int)
    _, nu, al, be = averaged_params(bel, nc)
    epi = be / (np.maximum(nu, 1e-6) * np.maximum(al - 1.0, 1e-6))
    return 1.0 / (1.0 + np.minimum(epi, 10.0))


def dist_map(path):
    """Per-cell distance to the nearest cell the corrupted wearable visited."""
    pts = np.unique(np.asarray(path).reshape(-1, 2), axis=0).astype(float)
    rr, cc = np.mgrid[0:G, 0:G]
    allc = np.stack([rr.ravel(), cc.ravel()], 1).astype(float)
    d = np.linalg.norm(allc[:, None, :] - pts[None, :, :], axis=2).min(1)
    return d.reshape(G, G)


rows_head, rows_dist, rows_cert = [], [], []
for seed in SEEDS:
    dm, db = dirs(seed)
    if not (dm / "manifest.yaml").exists() or not (db / "manifest.yaml").exists():
        print("missing seed %d" % seed)
        continue
    Rm, Rb = load_run(dm, seed), load_run(db, seed)
    rows_head.append((metrics(Rm, last=LAST)["averaged"],
                      metrics(Rb, last=LAST)["averaged"]))

    e2m, e2b = Rm["averaged"][0][-LAST:], Rb["averaged"][0][-LAST:]
    D = dist_map(np.load(dm / "bad_wearable_path.npy"))
    cm, cb = certainty(dm, seed)[-LAST:], certainty(db, seed)[-LAST:]

    per_bin, per_bin_c = {}, {}
    for lo, hi in BINS:
        m = (D >= lo) & (D < hi)
        ok = m[None, :, :] & np.isfinite(e2m) & np.isfinite(e2b)
        if ok.sum() < 100:
            continue
        per_bin[(lo, hi)] = (float(np.mean(e2m[ok])) - float(np.mean(e2b[ok])),
                             int(ok.sum()), int(m.sum()))
        okc = m[None, :, :] & np.isfinite(cm) & np.isfinite(cb)
        per_bin_c[(lo, hi)] = (float(np.mean(cm[okc])), float(np.mean(cb[okc])))
    rows_dist.append(per_bin)
    rows_cert.append(per_bin_c)

ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

print("=" * 104)
print("MISCALIBRATED WEARABLE (wearable 2, constant +20 deg, certainty 1.0)")
print("averaged readout, cell space, last-%d window, %d seeds, mean +/- sd"
      % (LAST, len(rows_head)))
print("=" * 104)
print("%-14s %-21s %-21s %-16s %-16s"
      % ("arm", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS"))
for lab, i in (("baseline", 1), ("miscalibrated", 0)):
    f = lambda k, p=5: "%.*f +/- %.*f" % (p, ms([r[i][k] for r in rows_head])[0],
                                          p, ms([r[i][k] for r in rows_head])[1])
    print("%-14s %-21s %-21s %-16s %-16s"
          % (lab, f("whole"), f("last50"), f("cov", 3), f("ratio", 3)))
for k, lab, p in (("whole", "whole-run MSE", 5), ("last50", "last-50 MSE", 5),
                  ("cov", "coverage", 3), ("ratio", "hw:RMS", 3)):
    a, b = ms([r[1][k] for r in rows_head])[0], ms([r[0][k] for r in rows_head])[0]
    print("  %-14s %+.*f  (%+.1f%%)" % (lab, p, b - a, 100 * (b - a) / a))

print("\n" + "=" * 104)
print("ITEM 2 -- SPATIAL DISTRIBUTION OF THE DAMAGE")
print("per-cell last-50 MSE difference (miscalibrated minus baseline),")
print("binned by distance from the corrupted wearable's realised path")
print("=" * 104)
print("%-16s %8s %26s" % ("distance (cells)", "cells", "MSE difference"))
keys = [k for k in BINS if any(k in r for r in rows_dist)]
for k in keys:
    v = [r[k][0] for r in rows_dist if k in r]
    nc = st.mean([r[k][2] for r in rows_dist if k in r])
    print("%-16s %8.0f %26s"
          % ("%d - %d" % k if k[1] < 99 else "%d+" % k[0], nc,
             "%+.5f +/- %.5f" % ms(v)))
near = [r[keys[0]][0] for r in rows_dist if keys[0] in r]
far = [r[keys[-1]][0] for r in rows_dist if keys[-1] in r]
print("\n  nearest bin %+.5f vs farthest bin %+.5f" % (st.mean(near), st.mean(far)))
print("  -> damage is %s"
      % ("LOCAL to the corrupted wearable" if st.mean(near) > 2 * abs(st.mean(far))
         else "NOT concentrated near the wearable -- it propagates"))

print("\n" + "=" * 104)
print("ITEM 3 -- CERTAINTY IN THE AFFECTED REGION  (the one that matters)")
print("=" * 104)
print("%-16s %-20s %-20s %-14s" % ("distance (cells)", "certainty MISCAL",
                                   "certainty BASELINE", "change"))
for k in keys:
    m = ms([r[k][0] for r in rows_cert if k in r])
    b = ms([r[k][1] for r in rows_cert if k in r])
    print("%-16s %-20s %-20s %+14.5f"
          % ("%d - %d" % k if k[1] < 99 else "%d+" % k[0],
             "%.5f +/- %.5f" % m, "%.5f +/- %.5f" % b, m[0] - b[0]))
dn = st.mean([r[keys[0]][0] - r[keys[0]][1] for r in rows_cert if keys[0] in r])
df = st.mean([r[keys[-1]][0] - r[keys[-1]][1] for r in rows_cert if keys[-1] in r])
print("\n  certainty change nearest the corrupted wearable : %+.5f" % dn)
print("  certainty change farthest from it               : %+.5f" % df)
print("\n  VERDICT:")
if dn < 0 and abs(dn) > abs(df) * 2:
    print("  Certainty FALLS where the corrupted wearable travelled, and falls")
    print("  more there than elsewhere -> the uncertainty is a usable corruption")
    print("  signal.")
else:
    print("  Certainty does NOT fall selectively where the corrupted wearable")
    print("  travelled. The mesh absorbs the corruption CONFIDENTLY: it reports")
    print("  a confident error, which is the more serious finding.")
