# R1a -- baseline characterisation on the full digit-7 held-out partition.
#
# Two products:
#   1. the canonical pinned-angle per-sample dump (pass 0), which every other
#      R1 sweep point is comparable against, plus its derived summary;
#   2. a 20-pass reference comparison, mean +/- sd, against the values recorded
#      in runs/stage0/baseline/manifest.yaml. The 0.2% reproduction floor does
#      not apply here -- it addresses mesh GPU non-determinism, not the
#      per-pass rotation-angle re-roll that dominates this measurement.
#
# Run: python -u experiments/section5_1/run_r1a.py

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from analyse import (read_dump, rms_by_uncertainty_decile, summarise,
                     write_rows_csv)
from dump import dump_pass, held_out_indices, load_model, write_csv

OUT = Path(ART + "/5.1_baseline_model_and_calibration/r1a_held_out_evaluation")
N_REFERENCE_PASSES = 20
MANIFEST = CHECKPOINTS["baseline"].parent / "training_manifest.yaml"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_idx = held_out_indices()
    model = load_model("baseline", device)
    print(f"device={device}  held-out n={len(eval_idx)}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # ── canonical pinned-angle pass ───────────────────────────────────────
    rows = dump_pass(model, eval_idx, "neutral", 1.0, device, pass_index=0)
    dump_path = write_csv(rows, OUT / "per_sample_baseline.csv")
    print(f"wrote {dump_path} ({len(rows)} rows)")

    d = read_dump(dump_path)
    summary = summarise(d)
    deciles = rms_by_uncertainty_decile(d)
    write_rows_csv(deciles, OUT / "rms_by_uncertainty_decile.csv")
    write_rows_csv(summary["coverage_detail"], OUT / "interval_coverage.csv")

    # ── 20-pass reference comparison ──────────────────────────────────────
    mses, maes, rs, rms_deg, rman = [], [], [], [], []
    for p in range(N_REFERENCE_PASSES):
        rp = dump_pass(model, eval_idx, "neutral", 1.0, device, pass_index=p)
        dp = {k: np.array([r[k] for r in rp], dtype=np.float64) for k in rp[0]}
        dp["alpha_floor_bound"] = dp["alpha_floor_bound"].astype(np.int64)
        s = summarise(dp)
        mses.append(s["circular_mse_norm"])
        maes.append(s["mean_abs_error_deg"])
        rs.append(s["uncertainty_error_pearson_r_total"])
        rms_deg.append(s["rms_error_deg"])
        rman.append(s["manifest_convention_pearson_r"])
        print(f"  pass {p:02d}: mse={mses[-1]:.5f} mae={maes[-1]:.3f}deg "
              f"r={rs[-1]:+.4f}")

    ref = yaml.safe_load(open(MANIFEST))["results"]
    passes = {
        "n_passes": N_REFERENCE_PASSES,
        "circular_mse_norm": {"mean": float(np.mean(mses)), "sd": float(np.std(mses, ddof=1)),
                              "min": float(np.min(mses)), "max": float(np.max(mses)),
                              "values": [float(v) for v in mses]},
        "mean_abs_error_deg": {"mean": float(np.mean(maes)), "sd": float(np.std(maes, ddof=1)),
                               "values": [float(v) for v in maes]},
        "rms_error_deg": {"mean": float(np.mean(rms_deg)), "sd": float(np.std(rms_deg, ddof=1)),
                          "values": [float(v) for v in rms_deg]},
        "manifest_convention_pearson_r": {
            "mean": float(np.mean(rman)), "sd": float(np.std(rman, ddof=1)),
            "values": [float(v) for v in rman]},
        "uncertainty_error_pearson_r_total": {
            "mean": float(np.mean(rs)), "sd": float(np.std(rs, ddof=1)),
            "values": [float(v) for v in rs]},
        "manifest_reference": {
            "final_holdout_mse_normalised": ref["final_holdout_mse_normalised"],
            "mean_error_degrees": ref["mean_error_degrees"],
            "uncertainty_error_pearson": ref["uncertainty_error_pearson"],
            "note": "manifest mean_error_degrees is sqrt(MSE)*180 (an RMS error), "
                    "not the mean absolute angular error; compare it to rms_error_deg. "
                    "manifest uncertainty_error_pearson correlates predictive_uncertainty "
                    "(the EPISTEMIC term) against |error|, not total uncertainty against "
                    "squared error -- both variants are reported below.",
        },
    }

    result = {"pinned_pass": summary, "reference_passes": passes,
              "params": sum(p.numel() for p in model.parameters()),
              "n_held_out": int(len(eval_idx))}
    with open(OUT / "summary.yaml", "w") as f:
        yaml.safe_dump(result, f, sort_keys=False, default_flow_style=False)
    print("\n" + json.dumps({k: v for k, v in summary.items()
                             if k != "coverage_detail"}, indent=2))
    print(f"\nwrote {OUT/'summary.yaml'}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="R1a held-out evaluation of the pretrained baseline (GPU, ~5 min)")
    _p.parse_args()
    main()
