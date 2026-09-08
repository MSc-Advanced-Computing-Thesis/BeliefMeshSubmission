# Miscalibrated wearable: decomposition of the hw:RMS change.
#
# hw:RMS is a ratio, so a rise can come from wider intervals, from errors
# growing more slowly than the intervals, or both. This splits it:
#
#   half-width  = mean of the averaged-readout 90% Student-t half-width (deg)
#   RMS         = sqrt(mean cell-space squared error) (deg)
#   ratio       = half-width / RMS   -- exactly as metrics() computes it
#
# Aggregate first, then the same three quantities by distance from the
# miscalibrated wearable's realised path, using the bins the damage and
# certainty breakdowns already use, so the three tables are on the same cells.
#
# Averaged readout, last-50 window, 5 seeds, paired against the matched
# baseline seed by seed.
#
# Run: python -u experiments/section5_2/run_miscal_hwrms_decomp.py

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

from _shared.averaged_readout import load_run

LAST, G, DEG = 50, 22, 180.0
SEEDS = [42, 1042, 2042, 3042, 4042]
MIS = Path(ART + "/5.8_system_robustness/5.8.2_sensor_miscalibration/miscalibrated_wearable")
BASE = Path(ART + "/5.3_belief_aggregation/5.3.2_mesh_scale/aggregation_arms")
BAD_W = 2
BINS = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 99)]


def parts(R, mask=None):
    """(mean half-width deg, RMS deg, ratio) over the last-50 window."""
    e2, hw = R["averaged"]
    e2, hw = e2[-LAST:], hw[-LAST:]
    ok = np.isfinite(e2) & np.isfinite(hw)
    if mask is not None:
        ok &= mask[None, :, :]
    if ok.sum() < 100:
        return None
    rms = float(np.sqrt(np.mean(e2[ok]))) * DEG
    h = float(np.mean(hw[ok])) * DEG
    return h, rms, h / rms


def dist_map(path):
    pts = np.unique(np.asarray(path).reshape(-1, 2), axis=0).astype(float)
    rr, cc = np.mgrid[0:G, 0:G]
    a = np.stack([rr.ravel(), cc.ravel()], 1).astype(float)
    return np.linalg.norm(a[:, None, :] - pts[None, :, :], axis=2).min(1).reshape(G, G)


agg = {"baseline": [], "miscal": []}
per_bin = {b: {"baseline": [], "miscal": []} for b in BINS}
for s in SEEDS:
    dm, db = MIS / ("miscal_seed%d" % s), BASE / ("nig_product_sampled_seed%d" % s)
    Rm, Rb = load_run(dm, s), load_run(db, s)
    agg["miscal"].append(parts(Rm))
    agg["baseline"].append(parts(Rb))
    D = dist_map(np.load(dm / "bad_wearable_path.npy"))
    for b in BINS:
        m = (D >= b[0]) & (D < b[1])
        pm, pb = parts(Rm, m), parts(Rb, m)
        if pm and pb:
            per_bin[b]["miscal"].append(pm)
            per_bin[b]["baseline"].append(pb)

ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
col = lambda rows, i: ms([r[i] for r in rows])

print("=" * 100)
print("hw:RMS DECOMPOSITION -- miscalibrated wearable vs baseline")
print("averaged readout, last-%d window, %d seeds, mean +/- sd" % (LAST, len(SEEDS)))
print("=" * 100)
print("%-14s %-22s %-22s %-18s" % ("arm", "half-width (deg)", "RMS error (deg)",
                                   "hw : RMS"))
for lab, key in (("baseline", "baseline"), ("miscalibrated", "miscal")):
    h, r, q = (col(agg[key], i) for i in range(3))
    print("%-14s %-22s %-22s %-18s"
          % (lab, "%.3f +/- %.3f" % h, "%.3f +/- %.3f" % r, "%.3f +/- %.3f" % q))
hb, rb, qb = (col(agg["baseline"], i)[0] for i in range(3))
hm, rm, qm = (col(agg["miscal"], i)[0] for i in range(3))
print()
print("  half-width  %.3f -> %.3f deg   %+.1f%%" % (hb, hm, 100 * (hm - hb) / hb))
print("  RMS error   %.3f -> %.3f deg   %+.1f%%" % (rb, rm, 100 * (rm - rb) / rb))
print("  hw : RMS    %.3f -> %.3f       %+.1f%%" % (qb, qm, 100 * (qm - qb) / qb))
print()
print("  WHICH COMPONENT MOVES:")
if abs(100 * (hm - hb) / hb) > 2 * abs(100 * (rm - rb) / rb):
    print("  the INTERVALS widen; the error barely moves")
elif abs(100 * (rm - rb) / rb) > 2 * abs(100 * (hm - hb) / hb):
    print("  the ERROR moves; the intervals barely widen")
else:
    print("  BOTH move; the ratio rises because the half-width grows the faster")

print("\n" + "=" * 100)
print("BY DISTANCE FROM THE MISCALIBRATED WEARABLE'S PATH")
print("=" * 100)
print("%-10s %-19s %-19s   %-19s %-19s   %-14s"
      % ("distance", "half-width base", "half-width miscal",
         "RMS base", "RMS miscal", "hw:RMS b -> m"))
for b in BINS:
    if not per_bin[b]["miscal"]:
        continue
    hb_, rb_, qb_ = (col(per_bin[b]["baseline"], i) for i in range(3))
    hm_, rm_, qm_ = (col(per_bin[b]["miscal"], i) for i in range(3))
    lab = "%d-%d" % b if b[1] < 99 else "%d+" % b[0]
    print("%-10s %-19s %-19s   %-19s %-19s   %.3f -> %.3f"
          % (lab, "%.2f +/- %.2f" % hb_, "%.2f +/- %.2f" % hm_,
             "%.2f +/- %.2f" % rb_, "%.2f +/- %.2f" % rm_, qb_[0], qm_[0]))

print("\n%-10s %12s %12s %12s" % ("distance", "d half-width", "d RMS", "d ratio"))
for b in BINS:
    if not per_bin[b]["miscal"]:
        continue
    hb_, rb_, qb_ = (col(per_bin[b]["baseline"], i)[0] for i in range(3))
    hm_, rm_, qm_ = (col(per_bin[b]["miscal"], i)[0] for i in range(3))
    lab = "%d-%d" % b if b[1] < 99 else "%d+" % b[0]
    print("%-10s %11.1f%% %11.1f%% %11.1f%%"
          % (lab, 100 * (hm_ - hb_) / hb_, 100 * (rm_ - rb_) / rb_,
             100 * (qm_ - qb_) / qb_))
print("\n  positive d half-width = intervals widened there")
