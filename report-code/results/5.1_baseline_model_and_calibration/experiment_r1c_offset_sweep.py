# R1c -- offset shift sweep. Frozen baseline checkpoint, neutral render, NO
# colour filter. A constant offset is added to the TARGET angle; the input
# image is untouched, so the model sees exactly what it saw at offset 0 and
# only the correct answer moves underneath it.
#
# The question is whether the uncertainty terms respond to the displacement at
# all. Interval coverage is the sharper test: if the model cannot detect the
# shift, coverage should collapse as the offset grows while the uncertainty
# terms stay flat -- confidently wrong rather than honestly uncertain.
#
# Run: python -u experiments/section5_1/run_r1c.py

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

import torch
import yaml

from analyse import write_rows_csv
from dump import held_out_indices, load_model
from protocol import N_PASSES, SEEDS, evaluate_point

OUT = Path(ART + "/5.1_baseline_model_and_calibration/r1c_offset_sweep")
OFFSETS = [float(10 * i) for i in range(12)]   # 0..110 deg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_idx = held_out_indices()
    model = load_model("baseline", device)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"sweep: {len(OFFSETS)} points x {N_PASSES} seeds")
    points = []
    for off in OFFSETS:
        label = f"offset{off:.0f}"
        agg, per_pass = evaluate_point(
            model, eval_idx, device, filter_type="neutral", strength=1.0,
            target_offset_deg=off, label=label, dump_dir=OUT / "per_sample")
        agg["offset_deg"] = off
        points.append(agg)
        write_rows_csv(per_pass, OUT / "per_pass" / f"{label}.csv")
        print(f"  {label:10s} mse={agg['circular_mse_norm_mean']:.5f}"
              f"+/-{agg['circular_mse_norm_sd']:.5f}  "
              f"epi={agg['mean_epistemic_mean']:.6f}  "
              f"ale={agg['mean_aleatoric_mean']:.6f}  "
              f"r={agg['manifest_convention_pearson_r_mean']:+.4f}  "
              f"cov90={agg['coverage_90_mean']:.4f}  "
              f"afloor={agg['alpha_floor_total']}")

    cols = (["offset_deg", "n_passes", "alpha_floor_total"]
            + [k for k in points[0] if k.endswith(("_mean", "_sd"))])
    write_rows_csv([{c: p[c] for c in cols} for p in points], OUT / "sweep.csv")
    with open(OUT / "summary.yaml", "w") as f:
        yaml.safe_dump({"seeds": list(SEEDS), "n_passes": N_PASSES,
                        "offsets_deg": OFFSETS, "points": points},
                       f, sort_keys=False, default_flow_style=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description=__doc__ or "R1c target-offset sweep of the pretrained baseline (GPU, ~15 min)")
    _p.parse_args()
    main()
