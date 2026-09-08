# R1b -- colour shift sweep. Frozen baseline checkpoint, full held-out
# partition, 20 angle seeds per point.
#
# NO rendering change. beliefmesh.data.filters.apply_colour_filter already
# takes a `strength` in [0,1] that interpolates each channel multiplier
# between 1.0 (no effect) and the filter's full value, which IS the blend
# weight w between the neutral green render (w=0) and the fully filtered
# render (w=1). RotatedDigitDataset already threads it through as
# filter_strength. The w=0 byte-identity against filter_type="neutral" is
# verified below rather than assumed.
#
# Run: python -u experiments/section5_1/run_r1b.py

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from analyse import write_rows_csv
from dump import held_out_indices, load_model
from protocol import N_PASSES, SEEDS, evaluate_point, paired_delta

from beliefmesh.data.digits import RotatedDigitDataset

OUT = Path(ART + "/5.1_baseline_model_and_calibration/r1b_colour_sweep")
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]
FILTERS = ["red", "blue"]
IDENTITY_N = 64     # fixed sample set for the w=0 byte-identity check


def verify_w0_identity(eval_idx) -> dict:
    """Byte-identity of the w=0 render against filter_type='neutral' on a
    fixed sample set. Same angle seed on both sides, so any difference is the
    render and not the draw."""
    out = {"n_samples": IDENTITY_N, "seed": SEEDS[0]}
    for ft in FILTERS:
        neutral = RotatedDigitDataset(filter_type="neutral", subset_indices=eval_idx)
        filt = RotatedDigitDataset(filter_type=ft, filter_strength=0.0,
                                   subset_indices=eval_idx)
        random.seed(SEEDS[0])
        a = torch.stack([neutral[i][0] for i in range(IDENTITY_N)])
        random.seed(SEEDS[0])
        b = torch.stack([filt[i][0] for i in range(IDENTITY_N)])
        identical = bool(torch.equal(a, b))
        out[ft] = {
            "bitwise_identical": identical,
            "max_abs_diff": float((a - b).abs().max()),
            "n_differing_elements": int((a != b).sum()),
            "n_elements": int(a.numel()),
        }
        print(f"  w=0 {ft:4s} vs neutral: bitwise_identical={identical} "
              f"max|diff|={out[ft]['max_abs_diff']:.3e} "
              f"differing={out[ft]['n_differing_elements']}/{a.numel()}")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_idx = held_out_indices()
    model = load_model("baseline", device)
    OUT.mkdir(parents=True, exist_ok=True)

    print("w=0 byte-identity check:")
    identity = verify_w0_identity(eval_idx)

    print(f"\nsweep: {len(FILTERS)}x{len(WEIGHTS)} = "
          f"{len(FILTERS)*len(WEIGHTS)} points x {N_PASSES} seeds")
    points, per_pass_store = [], {}
    for ft in FILTERS:
        for w in WEIGHTS:
            label = f"{ft}_w{w:.1f}"
            agg, per_pass = evaluate_point(
                model, eval_idx, device, filter_type=ft, strength=w,
                label=label, dump_dir=OUT / "per_sample")
            agg["filter_type"], agg["w"] = ft, w
            points.append(agg)
            per_pass_store[label] = per_pass
            write_rows_csv(per_pass, OUT / "per_pass" / f"{label}.csv")
            print(f"  {label:10s} mse={agg['circular_mse_norm_mean']:.5f}"
                  f"+/-{agg['circular_mse_norm_sd']:.5f}  "
                  f"epi={agg['mean_epistemic_mean']:.6f}  "
                  f"ale={agg['mean_aleatoric_mean']:.6f}  "
                  f"r={agg['manifest_convention_pearson_r_mean']:+.4f}  "
                  f"cov90={agg['coverage_90_mean']:.4f}  "
                  f"afloor={agg['alpha_floor_total']}")

    cols = (["filter_type", "w", "n_passes", "alpha_floor_total"]
            + [k for k in points[0] if k.endswith(("_mean", "_sd"))])
    write_rows_csv([{c: p[c] for c in cols} for p in points], OUT / "sweep.csv")

    # red vs blue at matched w: paired across the shared seed list
    symmetry = []
    for w in WEIGHTS:
        d = paired_delta(per_pass_store[f"red_w{w:.1f}"],
                         per_pass_store[f"blue_w{w:.1f}"], "circular_mse_norm")
        d["w"] = w
        symmetry.append(d)
    write_rows_csv(symmetry, OUT / "red_vs_blue_paired.csv")

    # w=0 must reproduce the R1a 20-seed headline exactly: same condition,
    # same seeds, same partition.
    r1a = yaml.safe_load(open(ART + "/5.1_baseline_model_and_calibration/r1a_held_out_evaluation/summary.yaml"))["reference_passes"]
    checks = {}
    for ft in FILTERS:
        p0 = next(p for p in points if p["filter_type"] == ft and p["w"] == 0.0)
        checks[ft] = {
            "w0_mse_mean": p0["circular_mse_norm_mean"],
            "w0_mse_sd": p0["circular_mse_norm_sd"],
            "r1a_mse_mean": r1a["circular_mse_norm"]["mean"],
            "r1a_mse_sd": r1a["circular_mse_norm"]["sd"],
            "abs_diff": abs(p0["circular_mse_norm_mean"] - r1a["circular_mse_norm"]["mean"]),
            "matches": bool(np.isclose(p0["circular_mse_norm_mean"],
                                       r1a["circular_mse_norm"]["mean"], rtol=1e-9)),
        }
        print(f"\nw=0 {ft} vs R1a headline: {p0['circular_mse_norm_mean']:.8f} vs "
              f"{r1a['circular_mse_norm']['mean']:.8f} -> matches={checks[ft]['matches']}")

    with open(OUT / "summary.yaml", "w") as f:
        yaml.safe_dump({"seeds": list(SEEDS), "n_passes": N_PASSES,
                        "weights": WEIGHTS, "filters": FILTERS,
                        "w0_byte_identity": identity,
                        "w0_matches_r1a": checks,
                        "points": points,
                        "red_vs_blue_paired": symmetry},
                       f, sort_keys=False, default_flow_style=False)
    print(f"\nwrote {OUT}")
    if not all(c["matches"] for c in checks.values()):
        print("!! w=0 does NOT match the R1a headline -- stopping as specified")
        sys.exit(1)


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description=__doc__ or "R1b colour-shift sweep of the pretrained baseline (GPU, ~30 min)")
    _p.parse_args()
    main()
