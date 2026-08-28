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

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import write_rows_csv
from dump import held_out_indices, load_model
from protocol import N_PASSES, SEEDS, evaluate_point

OUT = Path("runs/section5_1/r1c")
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
    main()
