# R1d -- width variants. R1a repeated in full for narrow and wide under the
# 20-seed protocol, plus a re-emission of baseline under the same protocol so
# all three rows of the table come from one code path and one seed list.
#
# Parameter counts are the 3-CHANNEL figures (389,444 / 172,324 / 961,924):
# these checkpoints are the pretrained Stage 0 models, not the 5-channel
# CoordConv mesh variant. The mesh's extra 18*w1 conv1 weights are
# zero-initialised at load (MeshNode.load_checkpoint) and stay zero in any
# frozen arm, so they carry no information here.
#
# Run: python -u experiments/section5_1/run_r1d.py

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import write_rows_csv
from dump import held_out_indices, load_model
from protocol import N_PASSES, SEEDS, evaluate_point

OUT = Path("runs/section5_1/r1d")
VARIANTS = ["narrow", "baseline", "wide"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_idx = held_out_indices()
    OUT.mkdir(parents=True, exist_ok=True)

    points = []
    for v in VARIANTS:
        model = load_model(v, device)
        n_params = sum(p.numel() for p in model.parameters())
        agg, per_pass = evaluate_point(
            model, eval_idx, device, filter_type="neutral", strength=1.0,
            label=v, dump_dir=OUT / "per_sample")
        agg["variant"], agg["n_params"] = v, int(n_params)
        points.append(agg)
        write_rows_csv(per_pass, OUT / "per_pass" / f"{v}.csv")
        print(f"  {v:9s} params={n_params:>9,}  "
              f"mse={agg['circular_mse_norm_mean']:.5f}+/-{agg['circular_mse_norm_sd']:.5f}  "
              f"rms={agg['rms_error_deg_mean']:.2f}deg  "
              f"mae={agg['mean_abs_error_deg_mean']:.2f}deg  "
              f"r={agg['manifest_convention_pearson_r_mean']:+.4f}  "
              f"cov90={agg['coverage_90_mean']:.4f}  "
              f"afloor={agg['alpha_floor_total']}")

    cols = (["variant", "n_params", "n_passes", "alpha_floor_total"]
            + [k for k in points[0] if k.endswith(("_mean", "_sd"))])
    write_rows_csv([{c: p[c] for c in cols} for p in points], OUT / "width_table.csv")
    with open(OUT / "summary.yaml", "w") as f:
        yaml.safe_dump({"seeds": list(SEEDS), "n_passes": N_PASSES,
                        "variants": VARIANTS, "points": points},
                       f, sort_keys=False, default_flow_style=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
