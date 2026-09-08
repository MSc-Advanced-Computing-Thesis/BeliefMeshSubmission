# Section 5.1 evaluation protocol (20-seed).
#
# R1a measured the per-pass rotation-angle re-roll at sd 0.00219 on an MSE mean
# of 0.01366 -- a single pass is not a stable measurement. Every evaluation
# point from R1b onward is therefore run over the SAME fixed list of 20 angle
# seeds and reported as mean +/- sd.
#
# The seed list is identical at every sweep point and identical to the R1a
# 20-pass reference (dump.ANGLE_SEED + pass_index for pass_index 0..19), so
# every point sees the same 1,253 images at the same 20 sets of angles. Every
# comparison across the sweep is therefore PAIRED: differences between points
# cannot come from the angle draw.

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

from analyse import (interval_coverage, manifest_convention_pearson,
                     crosssample_uncertainty_error_pearson, summarise)
from dump import ANGLE_SEED, COLUMNS, dump_pass, write_csv

N_PASSES = 20
SEEDS = [ANGLE_SEED + p for p in range(N_PASSES)]   # 42..61 inclusive

# The per-point quantities carried through every sweep. Each is aggregated as
# mean +/- sd over SEEDS.
TRACKED = [
    "circular_mse_norm",
    "rms_error_deg",
    "mean_abs_error_deg",
    "mean_epistemic",
    "mean_aleatoric",
    "mean_total",
    "manifest_convention_pearson_r",       # PRIMARY: epistemic vs |error|
    "uncertainty_error_pearson_r_total",
    "uncertainty_error_pearson_r_epistemic",
    "uncertainty_error_pearson_r_aleatoric",
    "coverage_90",
    "coverage_50",
    "coverage_80",
    "coverage_95",
    "n_alpha_floor_bound",
]


def rows_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    d = {}
    for c in COLUMNS:
        vals = [r[c] for r in rows]
        d[c] = (np.array(vals, dtype=np.int64) if c in ("sample_index", "alpha_floor_bound")
                else np.array(vals, dtype=np.float64))
    return d


def _flatten(summary: dict) -> dict:
    out = {k: summary[k] for k in TRACKED if k in summary}
    for lvl, val in summary["coverage"].items():
        out[f"coverage_{int(round(float(lvl) * 100))}"] = val
    return out


def evaluate_point(model, eval_idx, device, *, filter_type="neutral", strength=1.0,
                   target_offset_deg=0.0, label="", dump_dir: Path | None = None,
                   dump_seed_index: int = 0):
    """Run one sweep point over all 20 angle seeds.

    Returns (aggregate, per_pass) where aggregate carries mean/sd/values for
    every TRACKED quantity and per_pass is the list of per-seed summaries.

    dump_dir: if given, the per-sample CSV for seed SEEDS[dump_seed_index] is
    written there -- the same artefact shape R1a produced, one row per sample,
    so the per-point record is inspectable. The other 19 passes are retained as
    per-pass summary rows rather than 20x1,253 rows per point.
    """
    per_pass = []
    for p in range(N_PASSES):
        rows = dump_pass(model, eval_idx, filter_type, strength, device,
                         pass_index=p, target_offset_deg=target_offset_deg)
        if dump_dir is not None and p == dump_seed_index:
            write_csv(rows, dump_dir / f"per_sample_{label}_seed{SEEDS[p]}.csv")
        s = summarise(rows_to_arrays(rows))
        s["_seed"] = SEEDS[p]
        per_pass.append(_flatten(s) | {"seed": SEEDS[p]})

    agg = {"label": label, "n_passes": N_PASSES, "seeds": list(SEEDS)}
    for k in per_pass[0]:
        if k == "seed":
            continue
        v = np.array([pp[k] for pp in per_pass], dtype=np.float64)
        agg[f"{k}_mean"] = float(v.mean())
        agg[f"{k}_sd"] = float(v.std(ddof=1))
    agg["alpha_floor_total"] = int(sum(pp["n_alpha_floor_bound"] for pp in per_pass))
    return agg, per_pass


def paired_delta(a_per_pass, b_per_pass, key="circular_mse_norm"):
    """Paired difference b - a across the shared seed list: effect size first
    (mean difference and Cohen's dz), p-value second. Paired because both
    points saw identical angle draws seed for seed."""
    from scipy import stats
    a = np.array([p[key] for p in a_per_pass], dtype=np.float64)
    b = np.array([p[key] for p in b_per_pass], dtype=np.float64)
    d = b - a
    sd = d.std(ddof=1)
    t, p = stats.ttest_rel(b, a)
    return {"mean_diff": float(d.mean()), "sd_diff": float(sd),
            "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
            "n_pairs": int(len(d)), "p_value": float(p)}
