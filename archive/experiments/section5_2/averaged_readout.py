# Averaged-NIG readout convention for Chapter 5.
#
# RECOMPUTATION over stored covering beliefs. The training-time fusion rule is
# unchanged (still the NIG product); only the per-cell READOUT changes, from
# argmax (single highest-certainty covering belief) to the averaged rule:
# fuse_nig_product with per-contributor weights 1/N, i.e. n_eff = 1.
#
#     nu*    = mean(nu_i)
#     alpha* = mean(alpha_i)          (the product's +1.5(N-1) vanishes)
#     beta*  = mean(beta_i) + 0.5(mean(nu_i x_i^2) - nu* x*^2)
#     gamma* = sum nu_i x_i / sum nu_i   -- algebraically the product's gamma*
#
# At N = 1 every parameter returns the sole contributor's own belief, so
# single-covered cells are identical to argmax by construction. That is
# asserted, not assumed, by validate_reduction() below.
#
# Shared by every Chapter 5 table and figure that reports a cell-space
# quantity. Import from here rather than reimplementing.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from analyse_estimators import EX, truth_for
from beliefmesh.data.grid_environment import GridEnvironment
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

STATIC_FIELD = Path("runs/stage6/offset_world/dynamic_static_spatial/field.npy")
_tcache: dict = {}


def offset_field_for(run_dir: Path, G: int, T: int):
    """The field a run ACTUALLY used. truth_for() hardcodes the dynamic field,
    which is wrong for the 5.4.2 static arm (max |truth error| 0.877) and for
    the 5.6 multifield variants. Resolved from the run's own path/condition."""
    rd = str(run_dir).replace("\\", "/")
    if "s5_4_2_static" in rd:
        return np.load(STATIC_FIELD)
    if "s5_6_multifield" in rd:
        from stage6_spatial_mesh.offset_field_variants import build_field
        name = run_dir.name.split("_seed", 1)[1]
        name = name.split("_", 1)[1] if "_" in name else "field0_original"
        return np.asarray(build_field(name, G, T))
    return np.asarray(build_dynamic_offset_field(G, T))


def truth_for_run(run_dir: Path, seed: int, T: int, G: int):
    """Ground truth using the run's own offset field."""
    field = offset_field_for(Path(run_dir), G, T)
    key = (seed, T, G, float(np.asarray(field).sum()))
    if key in _tcache:
        return _tcache[key]
    env = GridEnvironment(G, np.full((T, G, G), 0.5), rotation_seed=seed,
                          offset_field=field, excluded_rotation_ranges=EX,
                          apply_colour_filter=False)
    rot = env.cell_rotations
    t = np.empty((T, G, G))
    for s_ in range(T):
        for r in range(G):
            for c in range(G):
                t[s_, r, c] = env._label(rot[s_, r, c], r, c, s_).item()
    _tcache[key] = t
    return t

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
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.sqrt(be * (1 + nu) / (nu * al))
    return sps.t.ppf(q, df=2 * al) * sc


def load_run(d: Path, seed: int):
    """Both conventions for one run, as (err^2, half-width) pairs."""
    d = Path(d)
    bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(d / "cell_ncov_steps.npy").astype(int)
    nig = np.load(d / "cell_nig_steps.npy")
    mse = np.load(d / "cell_mse_steps.npy")
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
    belief, so MSE and half-width must equal argmax's EXACTLY."""
    d = Path(d)
    bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(d / "cell_ncov_steps.npy").astype(int)
    R = load_run(d, seed)
    m = (nc == 1)
    if m.sum() == 0:
        return None
    gam, nu_s, al_s, be_s = averaged_params(bel, nc)
    # bel is (T, G, G, slot, param): slot 0 is the sole contributor here
    b0 = bel[..., 0, :]
    dev = {
        "gamma": float(np.nanmax(np.abs(wrap_np(gam[m] - b0[..., 0][m])))),
        "nu": float(np.nanmax(np.abs(nu_s[m] - b0[..., 1][m]))),
        "alpha": float(np.nanmax(np.abs(al_s[m] - b0[..., 2][m]))),
        "beta": float(np.nanmax(np.abs(be_s[m] - b0[..., 3][m]))),
        "mse_vs_argmax": float(np.nanmax(np.abs(
            R["averaged"][0][m] - R["argmax"][0][m]))),
        "hw_vs_argmax": float(np.nanmax(np.abs(
            R["averaged"][1][m] - R["argmax"][1][m]))),
        "n_cells": int(m.sum()),
    }
    return dev


def metrics(R, last=50, mask=None):
    """whole-run MSE, last-50 MSE, coverage, hw:RMS under both conventions."""
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
