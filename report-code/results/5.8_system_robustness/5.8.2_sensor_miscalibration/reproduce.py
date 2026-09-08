"""Section 5.8.2 -- Sensor Miscalibration.

Regenerates, from artefacts/5.8_system_robustness/5.8.2_sensor_miscalibration/
miscalibrated_wearable/ (one of three wearables biased by +20 degrees, 5 seeds)
against the matched baseline (the Section 5.3.2 Product fusion arm):

    Table 5.6   table_5_6_miscalibration.csv (+ .txt)
    Table 5.7   table_5_7_miscalibration_by_distance.csv (+ .txt)
                MSE difference, hw:RMS and half-width change by distance from
                the miscalibrated wearable's path

--rerun re-runs the five miscalibrated experiments (GPU, ~1 h) through
experiment_miscalibrated_wearable.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1]
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, TABLES, SEEDS  # noqa: E402
from _shared.artefacts import load_array  # noqa: E402
from _shared.averaged_readout import load_run, metrics  # noqa: E402
from _shared.style import ms, fmt  # noqa: E402

SECTION_ART = ARTEFACTS / "5.8_system_robustness" / "5.8.2_sensor_miscalibration" / "miscalibrated_wearable"
BASELINE_ART = ARTEFACTS / "5.3_belief_aggregation" / "5.3.2_mesh_scale" / "aggregation_arms"
LAST, G, DEG = 50, 22, 180.0
BINS = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 99)]


def dist_map(path):
    """Per-cell distance to the nearest cell the miscalibrated wearable visited."""
    pts = np.unique(np.asarray(path).reshape(-1, 2), axis=0).astype(float)
    rr, cc = np.mgrid[0:G, 0:G]
    a = np.stack([rr.ravel(), cc.ravel()], 1).astype(float)
    return np.linalg.norm(a[:, None, :] - pts[None, :, :], axis=2).min(1).reshape(G, G)


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


def analyse(mis: Path, base: Path, seeds):
    head, per_bin = [], {b: [] for b in BINS}
    agg = {"baseline": [], "miscal": []}
    for s in seeds:
        dm, db = mis / ("miscal_seed%d" % s), base / ("nig_product_sampled_seed%d" % s)
        if not (dm / "manifest.yaml").exists() or not (db / "manifest.yaml").exists():
            print("  missing seed %d" % s)
            continue
        Rm, Rb = load_run(dm, s), load_run(db, s)
        head.append((metrics(Rm, last=LAST)["averaged"], metrics(Rb, last=LAST)["averaged"]))
        agg["miscal"].append(parts(Rm)); agg["baseline"].append(parts(Rb))
        D = dist_map(load_array(dm, "bad_wearable_path"))
        e2m, e2b = Rm["averaged"][0][-LAST:], Rb["averaged"][0][-LAST:]
        for b in BINS:
            m = (D >= b[0]) & (D < b[1])
            ok = m[None] & np.isfinite(e2m) & np.isfinite(e2b)
            pm, pb = parts(Rm, m), parts(Rb, m)
            if ok.sum() >= 100 and pm and pb:
                per_bin[b].append((float(np.mean(e2m[ok]) - np.mean(e2b[ok])), int(m.sum()), pm, pb))
    return head, agg, per_bin


def write(head, agg, per_bin, tab_dir: Path):
    tab_dir.mkdir(parents=True, exist_ok=True)
    col = lambda rows, i: ms([r[i] for r in rows])
    lines = ["TABLE 5.6 -- ONE OF THREE WEARABLES MISCALIBRATED BY +20 deg (averaged-NIG readout, last-50 window, %d seeds)" % len(head),
             "%-14s %-21s %-21s %-15s %-15s" % ("condition", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS")]
    rows = []
    for lab, i in (("Baseline", 1), ("Miscalibrated", 0)):
        r = {k: ms([h[i][k] for h in head]) for k in ("whole", "last50", "cov", "ratio")}
        rows.append((lab, r))
        lines.append("%-14s %-21s %-21s %-15s %-15s" % (lab, fmt(r["whole"]), fmt(r["last50"]), fmt(r["cov"], 3), fmt(r["ratio"], 3)))
    b, m = rows[0][1], rows[1][1]
    lines.append("  whole-run %+.1f%%, last-50 %+.1f%%" % (100 * (m["whole"][0] - b["whole"][0]) / b["whole"][0],
                                                          100 * (m["last50"][0] - b["last50"][0]) / b["last50"][0]))
    hb, rb, qb = (col(agg["baseline"], i)[0] for i in range(3))
    hm, rm, qm = (col(agg["miscal"], i)[0] for i in range(3))
    lines.append("  half-width %.3f -> %.3f deg (%+.1f%%), RMS error %.3f -> %.3f deg (%+.1f%%), hw:RMS %.3f -> %.3f"
                 % (hb, hm, 100 * (hm - hb) / hb, rb, rm, 100 * (rm - rb) / rb, qb, qm))
    (tab_dir / "table_5_6_miscalibration.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    with open(tab_dir / "table_5_6_miscalibration.csv", "w", encoding="utf8") as f:
        f.write("condition,n_seeds,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,last50_mse_sd,coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd\n")
        for lab, r in rows:
            f.write("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (lab, len(head), *r["whole"], *r["last50"], *r["cov"], *r["ratio"]))
    print("\n".join(lines))

    lines = ["TABLE 5.7 -- IMPACT BY DISTANCE FROM THE MISCALIBRATED WEARABLE'S PATH (last-50 window, %d seeds)" % len(head),
             "%-10s %6s %-22s %-16s %-16s" % ("distance", "cells", "MSE difference", "hw:RMS base->mis", "half-width change")]
    csv = ["distance_cells,cells,mse_difference_mean,mse_difference_sd,hw_rms_baseline,hw_rms_miscalibrated,half_width_change_pct"]
    for bn in BINS:
        v = per_bin[bn]
        if not v:
            continue
        d = ms([x[0] for x in v]); n = np.mean([x[1] for x in v])
        hb_ = np.mean([x[3][0] for x in v]); hm_ = np.mean([x[2][0] for x in v])
        qb_ = np.mean([x[3][2] for x in v]); qm_ = np.mean([x[2][2] for x in v])
        lab = "%d-%d" % bn if bn[1] < 99 else "%d+" % bn[0]
        lines.append("%-10s %6.0f %-22s %-16s %+15.1f%%" % (lab, n, "%+.5f +/- %.5f" % d, "%.3f -> %.3f" % (qb_, qm_), 100 * (hm_ - hb_) / hb_))
        csv.append("%s,%.0f,%.6f,%.6f,%.4f,%.4f,%.2f" % (lab, n, d[0], d[1], qb_, qm_, 100 * (hm_ - hb_) / hb_))
    (tab_dir / "table_5_7_miscalibration_by_distance.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    (tab_dir / "table_5_7_miscalibration_by_distance.csv").write_text("\n".join(csv) + "\n", encoding="utf8")
    print("\n".join(lines))
    print("  wrote %s" % (tab_dir / "table_5_7_miscalibration_by_distance.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--baseline", default=str(BASELINE_ART), help="Section 5.3.2 Product fusion runs (the matched baseline)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rerun", action="store_true", help="re-run the five experiments first (GPU, ~1 h)")
    a = ap.parse_args()
    if a.rerun:
        subprocess.run([sys.executable, "-u", str(HERE / "experiment_miscalibrated_wearable.py"), "--root", a.artefacts],
                       check=True, cwd=str(HERE))
    head, agg, per_bin = analyse(Path(a.artefacts), Path(a.baseline), [int(s) for s in a.seeds.split(",")])
    write(head, agg, per_bin, Path(a.tables))
