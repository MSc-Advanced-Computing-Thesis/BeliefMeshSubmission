# Section 5.7: coverage and hw:RMS under node loss.
#
# Not computable before the rerun. The original runs stored only cell_mse_steps
# and cell_cert_steps, and certainty collapses (nu, alpha, beta) into one
# non-invertible scalar, so no Student-t interval could be recovered -- the
# section's own PROVENANCE.yaml says exactly this. The rerun stores the
# contributor beliefs, so both quantities exist for the first time.
#
# CONSTRUCTION matches the paired difference exactly, so the three numbers
# describe the same cells:
#   - averaged NIG readout (w_i = 1/N)
#   - last-50 window, steps 340..390
#   - restricted to cells covered on EVERY step of the window in BOTH the
#     failed run and its matched 0% control at that seed
#
# Reported for the FAILED run and for its CONTROL over the SAME cells, so the
# change under failure is visible rather than only the level. The control's
# value differs between rows because each row's cell set differs.
#
# Run: python -u experiments/section5_2/run_node_loss_calibration.py

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
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from _shared.averaged_readout import load_run
from _shared.calibration_band import BAND

ROOT = Path(ART + "/5.8_system_robustness/5.8.1_node_failure/node_loss")
POST_WINDOW = (340, 390)
CONTROL_COND = "nig_product_random_0pct"
OUT_CSV = Path(ART + "/5.8_system_robustness/5.8.1_node_failure/node_loss_calibration.csv")


def always_covered(e2_fail, e2_ctrl):
    """The paired construction's cell set, verbatim."""
    lo, hi = POST_WINDOW
    return ((~np.isnan(e2_fail[lo:hi])).all(axis=0)
            & (~np.isnan(e2_ctrl[lo:hi])).all(axis=0))


def calib(R, mask):
    """coverage and hw:RMS over the window, restricted to mask."""
    lo, hi = POST_WINDOW
    e2, hw = R["averaged"]
    e2w, hww = e2[lo:hi], hw[lo:hi]
    ok = np.isfinite(e2w) & np.isfinite(hww) & mask[None, :, :]
    if ok.sum() < 200:
        return None
    m = float(np.mean(e2w[ok]))
    return dict(n=int(ok.sum()),
                cov=float((np.sqrt(e2w[ok]) <= hww[ok]).mean()),
                ratio=float(np.mean(hww[ok]) / np.sqrt(m)))


runs = {}
for p in sorted(glob.glob(str(ROOT / "*" / "manifest.yaml"))):
    d = Path(os.path.dirname(p))
    m = yaml.safe_load(open(p))
    runs[(d.name.rsplit("_seed", 1)[0], int(m["seed"]))] = load_run(d, int(m["seed"]))

seeds = sorted({s for _, s in runs})
conds = [c for c in sorted({c for c, _ in runs}) if c != CONTROL_COND]
print("loaded %d runs: %d failed conditions + control, %d seeds"
      % (len(runs), len(conds), len(seeds)))

res = defaultdict(lambda: defaultdict(list))
for cond in conds:
    for seed in seeds:
        F, C = runs.get((cond, seed)), runs.get((CONTROL_COND, seed))
        if F is None or C is None:
            continue
        mask = always_covered(F["averaged"][0], C["averaged"][0])
        f, c = calib(F, mask), calib(C, mask)
        if f and c:
            res[cond]["failed"].append(f)
            res[cond]["control"].append(c)

ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

print("\n" + "=" * 118)
print("SECTION 5.7 CALIBRATION UNDER NODE LOSS -- averaged readout, last-50")
print("window (steps %d-%d), on the cells covered throughout in BOTH the failed"
      % POST_WINDOW)
print("run and its 0% control. 5 seeds, mean +/- sd.")
print("=" * 118)
print("%-30s %3s %-16s %-16s %-9s   %-16s %-16s %-9s"
      % ("condition", "n", "coverage FAIL", "coverage CTRL", "change",
         "hw:RMS FAIL", "hw:RMS CTRL", "change"))
rows = []
for cond in conds:
    if not res[cond]["failed"]:
        continue
    F, C = res[cond]["failed"], res[cond]["control"]
    cf, cc = ms([x["cov"] for x in F]), ms([x["cov"] for x in C])
    rf, rc = ms([x["ratio"] for x in F]), ms([x["ratio"] for x in C])
    print("%-30s %3d %-16s %-16s %+9.3f   %-16s %-16s %+9.3f"
          % (cond, len(F), "%.3f +/- %.3f" % cf, "%.3f +/- %.3f" % cc,
             cf[0] - cc[0], "%.3f +/- %.3f" % rf, "%.3f +/- %.3f" % rc,
             rf[0] - rc[0]))
    rows.append((cond, len(F), cf, cc, rf, rc))

print("\n  reference band for 90%% coverage: hw:RMS in [%.2f, %.2f]" % BAND)

# is the pattern worth a figure panel, or flat?
print("\n" + "=" * 118)
print("IS THERE A PATTERN ACROSS CONDITIONS?")
print("=" * 118)
for lab, i_f, i_c in (("coverage", 2, 3), ("hw:RMS", 4, 5)):
    fv = [r[i_f][0] for r in rows]
    dv = [r[i_f][0] - r[i_c][0] for r in rows]
    sd_within = st.mean([r[i_f][1] for r in rows])
    print("  %-9s failed-run level : range %.3f to %.3f (spread %.3f), "
          "mean within-condition sd %.3f"
          % (lab, min(fv), max(fv), max(fv) - min(fv), sd_within))
    print("  %-9s change vs control: range %+.3f to %+.3f (spread %.3f)"
          % (lab, min(dv), max(dv), max(dv) - min(dv)))
    print("            -> spread %s the within-condition sd"
          % ("EXCEEDS" if (max(fv) - min(fv)) > sd_within else "is BELOW"))

with open(OUT_CSV, "w", encoding="utf8") as f:
    f.write("condition,n_seeds,coverage_failed,coverage_failed_sd,coverage_control,"
            "coverage_control_sd,hw_rms_failed,hw_rms_failed_sd,hw_rms_control,"
            "hw_rms_control_sd\n")
    for cond, n, cf, cc, rf, rc in rows:
        f.write("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n"
                % (cond, n, cf[0], cf[1], cc[0], cc[1], rf[0], rf[1], rc[0], rc[1]))
print("\nwrote %s" % OUT_CSV)
