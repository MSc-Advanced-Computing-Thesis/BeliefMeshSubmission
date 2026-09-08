# Cert-MSE correlation conventions (2026-08 reporting correction). Pure
# analysis code -- computes correlations from already-saved
# cell_mse_steps/cell_cert_steps arrays; does not touch any experiment code
# path or change what any run actually did.
#
# Three conventions, all operating on the same (T, G, G) cell_mse_steps /
# cell_cert_steps arrays every run_mesh_experiment()-based script (and
# run_node_failure.py / run_certainty_calibration_diagnostic.py, which
# replicate the identical best-certainty-per-cell bookkeeping) already saves:
#
#   averaged_then_correlated (OLD, the convention used everywhere until now):
#       average each cell's mse and cert over the window FIRST, then
#       correlate the ~484 averaged (cert, mse) pairs across cells. Averaging
#       shrinks the cross-cell variance of an already tightly-clustered
#       certainty variable, which attenuates r through range restriction
#       independent of whether real discriminative signal exists.
#
#   pooled_raw: pool every (cell, step) pair in the window into one big
#       correlation (n up to G*G*window). No averaging, but conflates
#       "does certainty rank cells within a moment" with "does certainty
#       track the whole population's improvement over time" -- a cell late
#       in a converging run has both higher cert AND lower mse than a cell
#       early in the run, which pooled_raw partly picks up as if it were
#       discrimination.
#
#   per_timestep_then_averaged (NEW convention): correlate across cells
#       WITHIN each timestep, then average those per-timestep r values over
#       the window. Answers "does certainty rank cells by reliability right
#       now" -- the operationally relevant question -- without conflating it
#       with population-level convergence over time. Reports (mean, std,
#       n_valid_timesteps) since the std indicates how stable the
#       discrimination is; a single timestep's per-cell MSE rests on one
#       prediction per cell and can be noisy.

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

MIN_CELLS_PER_TIMESTEP = 8
MIN_CELLS_FOR_AVERAGED = 3


def averaged_then_correlated(cell_mse: np.ndarray, cell_cert: np.ndarray,
                             lo: int, hi: int) -> float:
    avg_mse = np.nanmean(cell_mse[lo:hi], axis=0)
    avg_cert = np.nanmean(cell_cert[lo:hi], axis=0)
    valid = np.isfinite(avg_mse) & np.isfinite(avg_cert)
    if valid.sum() < MIN_CELLS_FOR_AVERAGED:
        return float("nan")
    r, _ = pearsonr(avg_cert[valid], avg_mse[valid])
    return float(r)


def pooled_raw(cell_mse: np.ndarray, cell_cert: np.ndarray,
               lo: int, hi: int) -> tuple[float, int]:
    m = cell_mse[lo:hi]
    c = cell_cert[lo:hi]
    valid = np.isfinite(m) & np.isfinite(c)
    if valid.sum() < MIN_CELLS_PER_TIMESTEP:
        return float("nan"), int(valid.sum())
    r, _ = pearsonr(c[valid], m[valid])
    return float(r), int(valid.sum())


def per_timestep_then_averaged(cell_mse: np.ndarray, cell_cert: np.ndarray,
                               lo: int, hi: int) -> tuple[float, float, int, np.ndarray]:
    rs = []
    for t in range(lo, hi):
        m, c = cell_mse[t], cell_cert[t]
        valid = np.isfinite(m) & np.isfinite(c)
        if valid.sum() < MIN_CELLS_PER_TIMESTEP:
            continue
        r, _ = pearsonr(c[valid], m[valid])
        rs.append(r)
    rs = np.array(rs, dtype=float)
    if len(rs) == 0:
        return float("nan"), float("nan"), 0, rs
    return float(rs.mean()), float(rs.std()), len(rs), rs


def all_three(cell_mse: np.ndarray, cell_cert: np.ndarray,
             lo: int, hi: int) -> dict:
    old = averaged_then_correlated(cell_mse, cell_cert, lo, hi)
    pooled, n_pooled = pooled_raw(cell_mse, cell_cert, lo, hi)
    new_mean, new_std, n_t, _ = per_timestep_then_averaged(cell_mse, cell_cert, lo, hi)
    return dict(old_averaged=old, pooled_raw=pooled, pooled_raw_n=n_pooled,
               new_mean=new_mean, new_std=new_std, new_n_timesteps=n_t)
