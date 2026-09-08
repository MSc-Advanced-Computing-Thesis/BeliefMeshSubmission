"""The cell-space readout every Chapter 5 table and figure reports.

A cell is covered by one or more nodes; the reported estimate fuses ALL the
covering beliefs under Average Fusion (report Section 3.4: the closed-form
NIG fusion with per-contributor weights 1/N), so that

    nu*    = mean(nu_i)
    alpha* = mean(alpha_i)
    beta*  = mean(beta_i) + 0.5 (mean(nu_i x_i^2) - nu* x*^2)
    gamma* = sum nu_i x_i / sum nu_i

At N = 1 every parameter returns the sole contributor's own belief, which
validate_reduction() asserts against the stored per-cell MSE.

The 'argmax' readout (the single highest-certainty covering belief, as stored
in cell_nig_steps / cell_mse_steps) is kept alongside for the comparison in
Section 5.3.2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats as sps

from _shared.artefacts import load_array, manifest
from _shared.truth import truth_for_run

NU_FLOOR, ALPHA_FLOOR, BETA_FLOOR = 1e-6, 1.0 + 1e-3, 1e-6


def wrap_np(x):
    return x - 2.0 * np.round(x / 2.0)


def averaged_params(bel, nc):
    """(gamma*, nu*, alpha*, beta*) per (t, cell) under w_i = 1/N."""
    fin = np.isfinite(bel[..., 0])
    g, nu, al, be = (bel[..., i] for i in range(4))
    N = np.maximum(nc, 1)[..., None].astype(float)
    w = np.where(fin, 1.0 / N, 0.0)
    ref = g[..., 0]
    x = np.where(fin, wrap_np(g - ref[..., None]), 0.0)
    nu_f = np.where(fin, nu, 0.0)

    nu_s = np.maximum((w * nu_f).sum(-1), NU_FLOOR)
    x_s = (w * nu_f * x).sum(-1) / nu_s
    gam = wrap_np(ref + x_s)
    al_s = np.maximum((w * np.where(fin, al, 0.0)).sum(-1), ALPHA_FLOOR)
    sq = (w * nu_f * x * x).sum(-1)
    be_s = np.maximum((w * np.where(fin, be, 0.0)).sum(-1)
                      + 0.5 * (sq - nu_s * x_s * x_s), BETA_FLOOR)
    return gam, nu_s, al_s, be_s


def half_width(nu, al, be, q=0.95):
    """Half-width of the central 90% interval of the Student-t predictive."""
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.sqrt(be * (1 + nu) / (nu * al))
    return sps.t.ppf(q, df=2 * al) * sc


def certainty(nu, al, be):
    """Epistemic certainty of report Equation 3.3 (u_max = 10)."""
    epi = be / (np.maximum(nu, 1e-6) * np.maximum(al - 1.0, 1e-6))
    return 1.0 / (1.0 + np.minimum(epi, 10.0))


def load_run(d: Path, seed: int):
    """Both readouts for one run, as (squared error, half-width) pairs."""
    d = Path(d)
    bel = load_array(d, "cell_beliefs_steps").astype(np.float64)
    nc = load_array(d, "cell_ncov_steps").astype(int)
    nig = load_array(d, "cell_nig_steps")
    mse = load_array(d, "cell_mse_steps")
    T, G = mse.shape[0], mse.shape[1]

    hw_arg = np.where(nig[..., 1] > 1.0,
                      half_width(nig[..., 0], nig[..., 1], nig[..., 2]), np.nan)
    gam, nu_s, al_s, be_s = averaged_params(bel, nc)
    e2_avg = wrap_np(gam - truth_for_run(d, seed, T, G)) ** 2
    return dict(argmax=(mse, hw_arg),
                averaged=(e2_avg, half_width(nu_s, al_s, be_s)),
                nc=nc, T=T)


def validate_reduction(d: Path, seed: int):
    """At n_cov == 1 the averaged rule must return the sole contributor's own
    belief, so MSE and half-width must equal argmax's exactly."""
    d = Path(d)
    bel = load_array(d, "cell_beliefs_steps").astype(np.float64)
    nc = load_array(d, "cell_ncov_steps").astype(int)
    R = load_run(d, seed)
    m = (nc == 1)
    if m.sum() == 0:
        return None
    gam, nu_s, al_s, be_s = averaged_params(bel, nc)
    b0 = bel[..., 0, :]
    return {
        "gamma": float(np.nanmax(np.abs(wrap_np(gam[m] - b0[..., 0][m])))),
        "nu": float(np.nanmax(np.abs(nu_s[m] - b0[..., 1][m]))),
        "alpha": float(np.nanmax(np.abs(al_s[m] - b0[..., 2][m]))),
        "beta": float(np.nanmax(np.abs(be_s[m] - b0[..., 3][m]))),
        "mse_vs_argmax": float(np.nanmax(np.abs(R["averaged"][0][m] - R["argmax"][0][m]))),
        "hw_vs_argmax": float(np.nanmax(np.abs(R["averaged"][1][m] - R["argmax"][1][m]))),
        "n_cells": int(m.sum()),
    }


def metrics(R, last=50, mask=None):
    """whole-run MSE, last-50 MSE, 90% coverage, hw:RMS under both readouts."""
    T = R["T"]
    sl = slice(T - last, T)
    out = {}
    for arm in ("argmax", "averaged"):
        e2, hw = R[arm]
        w = float(np.nanmean(e2))
        ok = np.isfinite(e2[sl]) & np.isfinite(hw[sl])
        if mask is not None:
            ok &= mask[sl] if mask.shape == e2.shape else mask
        m = float(np.mean(e2[sl][ok]))
        out[arm] = dict(whole=w, last50=m, n=int(ok.sum()),
                        cov=float((np.sqrt(e2[sl][ok]) <= hw[sl][ok]).mean()),
                        ratio=float(np.mean(hw[sl][ok]) * 180 / (np.sqrt(m) * 180)))
    return out


def run_metrics(d: Path, last=50) -> dict:
    """Averaged-readout metrics of one stored run (seed read from its manifest)."""
    return metrics(load_run(d, int(manifest(d)["env_seed"])), last=last)["averaged"]


def arm_summary(dirs, last=50) -> dict | None:
    """Mean and sd across the runs of one arm, per metric."""
    import statistics as st
    M = [run_metrics(d, last) for d in dirs]
    if not M:
        return None
    ms = lambda k: (st.mean([m[k] for m in M]),
                    st.stdev([m[k] for m in M]) if len(M) > 1 else 0.0)
    return dict(n=len(M), whole=ms("whole"), last50=ms("last50"),
                cov=ms("cov"), ratio=ms("ratio"))
