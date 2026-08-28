# Section 5.1 derived quantities. Reads the per-sample CSVs written by dump.py
# and computes every summary statistic. Nothing here touches a model or a
# dataset -- R1a, R1b, R1c and R1d all reduce to this one code path.
#
# numpy/csv only: pandas is not a dependency of this project and this section
# must not change the environment that produced the already-reported numbers.

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)
HALF_PERIOD = 1.0   # normalised angle domain is [-1, 1), period 2


def read_dump(path: str | Path) -> dict[str, np.ndarray]:
    """Per-sample CSV -> dict of float arrays (alpha_floor_bound stays int)."""
    with open(path, newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    out = {}
    for c in cols:
        vals = [r[c] for r in rows]
        out[c] = (np.array(vals, dtype=np.int64) if c in ("sample_index", "alpha_floor_bound")
                  else np.array(vals, dtype=np.float64))
    return out


def crosssample_uncertainty_error_pearson(d, column: str = "total"):
    """ONE correlation across the whole held-out set, between predictive
    uncertainty and SQUARED circular error.

    Named to keep it distinct from the mesh convention
    (cert_mse_metrics.per_timestep_then_averaged), which correlates certainty
    against MSE across CELLS WITHIN a timestep and then averages over
    timesteps. The two are different estimators of different things and must
    never be quoted against each other.
    """
    m = np.isfinite(d[column]) & np.isfinite(d["circular_error_norm"])
    r, p = stats.pearsonr(d[column][m], d["circular_error_norm"][m] ** 2)
    return float(r), float(p)


def manifest_convention_pearson(d):
    """The correlation exactly as runs/stage0/*/manifest.yaml recorded it:
    predictive_uncertainty (= the EPISTEMIC term, beta/(nu*(alpha-1))) against
    the ABSOLUTE wrapped error, not total uncertainty against squared error.
    Reported alongside the Section 5.1 convention so the two are never
    silently interchanged -- experiments/stage0_baseline/run.py:evaluate."""
    m = np.isfinite(d["epistemic"])
    r, p = stats.pearsonr(d["epistemic"][m], np.abs(d["circular_error_norm"][m]))
    return float(r), float(p)


def rms_by_uncertainty_decile(d, column: str = "total") -> list[dict]:
    """RMS circular error within each decile of predictive uncertainty.
    Deciles are taken on rank, so ties cannot pile into one bin."""
    m = np.isfinite(d[column])
    u, e = d[column][m], d["circular_error_norm"][m]
    order = np.argsort(u, kind="stable")
    bins = np.array_split(order, 10)
    out = []
    for i, idx in enumerate(bins, start=1):
        rms = float(np.sqrt(np.mean(e[idx] ** 2)))
        out.append({
            "decile": i, "n": int(len(idx)),
            "u_mean": float(np.mean(u[idx])), "u_median": float(np.median(u[idx])),
            "rms_error_norm": rms, "rms_error_deg": rms * 180.0,
            "mean_abs_error_deg": float(np.mean(np.abs(e[idx]))) * 180.0,
        })
    return out


def interval_coverage(d, levels=NOMINAL_LEVELS) -> list[dict]:
    """Empirical coverage of the Student-t predictive interval.

    Location gamma, df 2*alpha, scale sqrt(beta*(1+nu)/(nu*alpha)) -- the
    marginal predictive of models/evidential.py:student_t_marginal.

    Containment is tested with CIRCULAR WRAPPING: the target is inside iff
    |circular_diff(target, gamma)| <= half-width, and circular_error_norm is
    already that wrapped difference. A half-width at or beyond the half-period
    (1.0 normalised = 180 deg) covers the entire circle, so such intervals are
    trivially satisfied -- counted and reported rather than silently inflating
    coverage.
    """
    err = np.abs(d["circular_error_norm"])
    dof, scale = 2.0 * d["alpha"], d["scale"]
    usable = np.isfinite(scale) & np.isfinite(dof) & (d["alpha_floor_bound"] == 0)

    rows = []
    for level in levels:
        q = stats.t.ppf(0.5 + level / 2.0, df=dof[usable])
        half = q * scale[usable]
        inside = err[usable] <= half
        rows.append({
            "nominal": float(level),
            "empirical": float(inside.mean()),
            "n_used": int(usable.sum()),
            "n_excluded_alpha_floor": int((~usable).sum()),
            "frac_interval_covers_circle": float((half >= HALF_PERIOD).mean()),
            "mean_half_width_deg": float(np.mean(half)) * 180.0,
            "median_half_width_deg": float(np.median(half)) * 180.0,
        })
    return rows


def summarise(d) -> dict:
    """The full derived set for one per-sample dump."""
    e = d["circular_error_norm"]
    r_tot, p_tot = crosssample_uncertainty_error_pearson(d, "total")
    r_epi, p_epi = crosssample_uncertainty_error_pearson(d, "epistemic")
    r_ale, p_ale = crosssample_uncertainty_error_pearson(d, "aleatoric")
    r_man, p_man = manifest_convention_pearson(d)
    fin = lambda k: d[k][np.isfinite(d[k])]
    cov = interval_coverage(d)
    return {
        "n_samples": int(len(e)),
        "circular_mse_norm": float(np.mean(e ** 2)),
        "rms_error_deg": float(np.sqrt(np.mean(e ** 2))) * 180.0,
        "mean_abs_error_deg": float(np.mean(np.abs(e))) * 180.0,
        "median_abs_error_deg": float(np.median(np.abs(e))) * 180.0,
        "p90_abs_error_deg": float(np.percentile(np.abs(e), 90)) * 180.0,
        "max_abs_error_deg": float(np.max(np.abs(e))) * 180.0,
        "mean_epistemic": float(np.mean(fin("epistemic"))),
        "mean_aleatoric": float(np.mean(fin("aleatoric"))),
        "mean_total": float(np.mean(fin("total"))),
        "mean_nu": float(np.mean(d["nu"])),
        "mean_alpha": float(np.mean(d["alpha"])),
        "mean_beta": float(np.mean(d["beta"])),
        "uncertainty_error_pearson_r_total": r_tot,
        "uncertainty_error_pearson_p_total": p_tot,
        "uncertainty_error_pearson_r_epistemic": r_epi,
        "uncertainty_error_pearson_p_epistemic": p_epi,
        "uncertainty_error_pearson_r_aleatoric": r_ale,
        "uncertainty_error_pearson_p_aleatoric": p_ale,
        "manifest_convention_pearson_r": r_man,
        "manifest_convention_pearson_p": p_man,
        "n_alpha_floor_bound": int(d["alpha_floor_bound"].sum()),
        "frac_alpha_floor_bound": float(d["alpha_floor_bound"].mean()),
        "coverage": {f"{r['nominal']:.2f}": r["empirical"] for r in cov},
        "coverage_detail": cov,
    }


def write_rows_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path
