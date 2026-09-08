# F -- error tail by true angle. Read-only: consumes the existing R1a
# per_sample_baseline.csv, evaluates nothing.
#
# R1a's max absolute error was 171.5 deg, which is close enough to a half turn
# to suggest rotational self-ambiguity of the digit rather than a degraded
# prediction. This bins by TRUE angle to establish whether the tail
# concentrates, and whether epistemic uncertainty is elevated where it does --
# i.e. whether the head's uncertainty is semantically meaningful or merely
# correlated with error magnitude.
#
# The mesh runs exclude 120-150 and +/-165-180 from rotation sampling
# (GridEnvironment.excluded_rotation_ranges). Whether those are the same bins
# is checked here, not assumed.

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import sys
from pathlib import Path

import numpy as np
import yaml

from analyse import read_dump, write_rows_csv

SRC = Path(ART + "/5.1_baseline_model_and_calibration/r1a_held_out_evaluation/per_sample_baseline.csv")
OUT = Path(ART + "/5.1_baseline_model_and_calibration/f_angle_bins")
BIN_DEG = 15.0
MESH_EXCLUDED = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]


def in_excluded(lo: float, hi: float) -> bool:
    """True if the bin overlaps any mesh-excluded range."""
    return any(lo < ehi and hi > elo for elo, ehi in MESH_EXCLUDED)


def main():
    d = read_dump(SRC)
    ang = d["true_angle_deg"]
    abs_err = np.abs(d["circular_error_deg"])
    epi = d["epistemic"]

    edges = np.arange(-180.0, 180.0 + BIN_DEG, BIN_DEG)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (ang >= lo) & (ang < hi)
        if not m.any():
            continue
        rows.append({
            "bin_lo_deg": float(lo), "bin_hi_deg": float(hi),
            "bin_centre_deg": float((lo + hi) / 2),
            "n": int(m.sum()),
            "mean_abs_error_deg": float(np.mean(abs_err[m])),
            "median_abs_error_deg": float(np.median(abs_err[m])),
            "p90_abs_error_deg": float(np.percentile(abs_err[m], 90)),
            "max_abs_error_deg": float(np.max(abs_err[m])),
            "mean_epistemic": float(np.mean(epi[m])),
            "median_epistemic": float(np.median(epi[m])),
            "n_err_over_90deg": int((abs_err[m] > 90).sum()),
            "mesh_excluded_overlap": in_excluded(lo, hi),
        })
    write_rows_csv(rows, OUT / "error_by_true_angle.csv")

    exc = [r for r in rows if r["mesh_excluded_overlap"]]
    inc = [r for r in rows if not r["mesh_excluded_overlap"]]
    w = lambda rs, k: (float(np.average([r[k] for r in rs], weights=[r["n"] for r in rs]))
                       if rs else float("nan"))

    # which bins actually carry the tail, independent of the mesh ranges
    ranked = sorted(rows, key=lambda r: -r["mean_abs_error_deg"])
    big = [r for r in rows if r["n_err_over_90deg"] > 0]

    summary = {
        "source": str(SRC), "bin_width_deg": BIN_DEG, "n_bins": len(rows),
        "n_samples": int(len(ang)),
        "mesh_excluded_ranges": [list(t) for t in MESH_EXCLUDED],
        "bins_overlapping_mesh_excluded": [
            [r["bin_lo_deg"], r["bin_hi_deg"]] for r in exc],
        "weighted_mean_abs_error_deg": {
            "in_mesh_excluded_bins": w(exc, "mean_abs_error_deg"),
            "outside": w(inc, "mean_abs_error_deg"),
        },
        "weighted_mean_epistemic": {
            "in_mesh_excluded_bins": w(exc, "mean_epistemic"),
            "outside": w(inc, "mean_epistemic"),
        },
        "n_err_over_90deg": {
            "total": int(sum(r["n_err_over_90deg"] for r in rows)),
            "in_mesh_excluded_bins": int(sum(r["n_err_over_90deg"] for r in exc)),
            "outside": int(sum(r["n_err_over_90deg"] for r in inc)),
            "n_samples_in_excluded": int(sum(r["n"] for r in exc)),
            "n_samples_outside": int(sum(r["n"] for r in inc)),
        },
        "top5_bins_by_mean_abs_error": [
            {"bin": [r["bin_lo_deg"], r["bin_hi_deg"]], "n": r["n"],
             "mean_abs_error_deg": r["mean_abs_error_deg"],
             "mean_epistemic": r["mean_epistemic"],
             "mesh_excluded_overlap": r["mesh_excluded_overlap"]}
            for r in ranked[:5]],
        "bins_containing_any_error_over_90deg": [
            {"bin": [r["bin_lo_deg"], r["bin_hi_deg"]], "n": r["n"],
             "n_err_over_90deg": r["n_err_over_90deg"],
             "mean_epistemic": r["mean_epistemic"],
             "mesh_excluded_overlap": r["mesh_excluded_overlap"]}
            for r in big],
    }
    with open(OUT / "summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)

    print(f"{'bin':>16}  {'n':>4}  {'meanAE':>7}  {'p90':>7}  {'max':>7}  "
          f"{'epi':>9}  {'>90d':>4}  excl")
    for r in rows:
        print(f"[{r['bin_lo_deg']:+6.0f},{r['bin_hi_deg']:+6.0f})  {r['n']:4d}  "
              f"{r['mean_abs_error_deg']:7.2f}  {r['p90_abs_error_deg']:7.2f}  "
              f"{r['max_abs_error_deg']:7.2f}  {r['mean_epistemic']:9.6f}  "
              f"{r['n_err_over_90deg']:4d}  {'YES' if r['mesh_excluded_overlap'] else ''}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Error by true-angle bin from the R1a dump (no GPU)")
    _p.parse_args()
    main()
