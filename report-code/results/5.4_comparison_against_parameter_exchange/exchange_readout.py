"""Readout for the parameter-exchange comparison (Section 5.4, Appendix D).

Five offset environments x four exchange mechanisms x five seeds, stored under
artefacts/5.4_comparison_against_parameter_exchange/five_environments/
<field>/gossip_cmp_<mode>[_seed<seed>]/.

Two readouts are provided:

    truth="own"        each run's cell error is taken against the offset field
                       that run was actually generated with (field0 ... field4).
    truth="field0"     every run is scored against the field0 field. This is
                       how the numbers published in the report's Table 5.3 and
                       Figures 5.7 / D.1 were computed (the readout at the time
                       did not resolve the field from the run), and is kept so
                       the published numbers remain reproducible. It is wrong
                       for field1 ... field4 and is written with the suffix
                       _as_published.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent)]

from _shared.artefacts import load_array, manifest, run_dirs  # noqa: E402
from _shared.averaged_readout import averaged_params, half_width, metrics, wrap_np  # noqa: E402
from _shared.truth import truth_for_field  # noqa: E402
from beliefmesh.simulation.field_variants import build_field  # noqa: E402

FIELDS = ["field0_original", "field1_seed11", "field2_seed23", "field3_seed37", "field4_seed51_dual"]
MODES = ["fusion", "gossip_uniform", "gossip_weighted", "fedavg_global"]
N_NODES, T_STEPS, G = 36, 390, 22
LAST = 50


def load_run_with_truth(d: Path, seed: int, field_name: str):
    """Averaged readout of one run against a named offset field."""
    bel = load_array(d, "cell_beliefs_steps").astype(np.float64)
    nc = load_array(d, "cell_ncov_steps").astype(int)
    nig = load_array(d, "cell_nig_steps")
    mse = load_array(d, "cell_mse_steps")
    T, Gr = mse.shape[0], mse.shape[1]
    field = np.asarray(build_field(field_name, Gr, T))
    gam, nu_s, al_s, be_s = averaged_params(bel, nc)
    e2 = wrap_np(gam - truth_for_field(field, seed, T, Gr)) ** 2
    hw_arg = np.where(nig[..., 1] > 1.0, half_width(nig[..., 0], nig[..., 1], nig[..., 2]), np.nan)
    return dict(argmax=(mse, hw_arg), averaged=(e2, half_width(nu_s, al_s, be_s)), nc=nc, T=T)


def runs_for(root: Path, field: str, mode: str) -> list[Path]:
    out = []
    for d in run_dirs(root / field / ("gossip_cmp_%s*" % mode)):
        if d.name.replace("gossip_cmp_", "").rsplit("_seed", 1)[0] == mode:
            out.append(d)
    return out


def collect(root: Path, truth: str = "own") -> dict:
    """{(field, mode): [per-run dict(metrics + comm_total + comm_step + trace)]}"""
    out = {}
    for f in FIELDS:
        for m in MODES:
            rows = []
            for d in runs_for(root, f, m):
                man = manifest(d)
                seed = int(man["env_seed"])
                R = load_run_with_truth(d, seed, f if truth == "own" else "field0_original")
                mm = metrics(R, last=LAST)["averaged"]
                r = man.get("results", {}) or {}
                mm["comm_total"] = r.get("comm_bytes_total", float("nan"))
                mm["comm_step"] = r.get("comm_bytes_mean_per_step", float("nan"))
                with np.errstate(invalid="ignore"):
                    mm["trace"] = np.nanmean(R["averaged"][0], axis=(1, 2))
                rows.append(mm)
            out[(f, m)] = rows
    return out


ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def pooled(C: dict, mode: str) -> list[dict]:
    return [r for f in FIELDS for r in C[(f, mode)]]


def offset_trace(field: str) -> np.ndarray:
    return np.abs(np.asarray(build_field(field, G, T_STEPS))).mean(axis=(1, 2))
