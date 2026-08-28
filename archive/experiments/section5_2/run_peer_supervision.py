# Section 5.2 -- peer supervision as a training signal (RQ1).
#
# Two nodes with overlapping FOVs. No wider mesh, no multi-hop BFS: with two
# nodes the mesh's own BFS is exactly one hop, so this uses beliefmesh's real
# Mesh/MeshNode rather than a reimplementation -- the update rule, loss,
# optimiser, certainty weighting and per-timestep step count are the mesh
# experiments' own, unmodified. Nothing on a reported code path is touched;
# this file only drives it.
#
#   anchor   trains on wearable ground truth inside its FOV (hop 0).
#   student  receives the anchor's beliefs on the shared cells, fuses them
#            (a no-op at N=1 contributor, which is provably the identity for
#            product-of-NIG fusion), trains on the fused point estimate with
#            the fused epistemic certainty as the per-sample weight. It never
#            receives a wearable measurement and never inherits the anchor's
#            uncertainty -- its (nu, alpha, beta) come from its own head.
#
# WHY mode="nig_product" IMPLEMENTS THE SPEC'S WEIGHTING. In _aggregate the
# single-contributor branch returns (gamma_i, cert_i, 1.0) with
# cert_i = 1/(1 + min(epistemic, 10)), and run_timestep sets
# cell_trusts = agreement * inherited = cert_i, which becomes the per-sample
# `weights` passed to nig_loss when temper_gradient is True (the default).
# So the student's gradient IS tempered by the fused epistemic certainty.
# The naive/certainty modes return 1.0 there and would train unweighted.
#
# ARMS (identical data, seeds, wearable trajectory):
#   student + anchor   one run, mode="nig_product"; both nodes reported.
#   frozen             same construction, mode="frozen" -- no training (floor).
#   direct             a SECOND wearable is placed on the shared cells so the
#                      student is itself an anchor there and trains on ground
#                      truth over the same cells the student arm gets
#                      pseudo-labels for (ceiling). The anchor's own wearable
#                      is unchanged, so its trajectory stays comparable.
#
# METRICS ARE CELL SPACE (item H convention): for each node, the mean over
# (step, cell in that node's FOV) of the squared wrapped error of THAT node's
# belief. With one belief per cell per node this is the single-node reduction
# of runner.py's best-certainty-per-cell array, so it sits on the same ruler
# as the frozen 0.09697 reference and the aggregation arms at 0.0204-0.0214.
#
# Run: python -u experiments/section5_2/run_peer_supervision.py --world offset
#      python -u experiments/section5_2/run_peer_supervision.py --world colour

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT / "section5_1"))
from stage6_spatial_mesh.cert_mse_metrics import per_timestep_then_averaged
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

from beliefmesh.config import load_config
from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.metrics.circular import circular_diff
from beliefmesh.node.mesh import Mesh, fov_cells


class SamplingMesh(Mesh):
    """Mesh with ONE optional change, default OFF: the student's per-cell
    training target may be DRAWN from the fused Student-t predictive rather
    than taken at its mode.

    Implemented as a subclass so beliefmesh/node/mesh.py -- a reported code
    path -- is not touched. With sample_target=False (the default) every call
    returns exactly what Mesh._aggregate returns, so the existing arms are
    bit-identical to the completed 5.2 run.

    The fused belief at N=1 contributor IS that contributor's belief
    (product-of-NIG fusion is provably the identity there), which is the only
    case this two-node setup ever produces; N>1 raises rather than guessing.
    """

    def __init__(self, *args, sample_target: bool = False,
                 target_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_target = sample_target
        self._target_rng = np.random.default_rng(target_seed)
        self.sample_deviations_deg: list[float] = []

    def _aggregate(self, contributions, own_belief=None, cell=None):
        label, agreement, inherited = super()._aggregate(contributions, own_belief, cell)
        if not self.sample_target:
            return label, agreement, inherited
        if len(contributions) != 1 or (self.self_weight > 0 and own_belief is not None):
            raise RuntimeError("sampled-target arm assumes exactly one contributor "
                               f"per cell; got {len(contributions)}")
        _, nu, al, be = contributions[0][1]
        scale = float(np.sqrt(be * (1.0 + nu) / (nu * al)))
        draw = label + scale * float(self._target_rng.standard_t(2.0 * al))
        draw = ((draw + 1.0) % 2.0) - 1.0
        self.sample_deviations_deg.append(
            abs(circular_diff(torch.tensor(draw), torch.tensor(label)).item()) * 180.0)
        return float(draw), agreement, inherited

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
OUT_ROOT = Path("runs/section5_2")

# Matched to run_aggregation_comparator.py so 5.2 transfers to the comparators.
LR = 3e-5
LAM = 5.0
FOV = 7
N_WEARABLE_SAMPLES = 1
N_TRAIN_REPEATS = 1
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
SEEDS = [42, 1042, 2042, 3042, 4042]
LAST_N = 50

# Two adjacent nodes from the comparator's own placement (stride 3, fov 7),
# so the overlap geometry is the mesh's, not a new one invented here.
ANCHOR_CENTRE, STUDENT_CENTRE = (3, 3), (6, 3)

# Colour conditions. GridEnvironment has NO blend-weight parameter -- the
# R1b filter_strength axis lives in data/filters.apply_colour_filter, which
# the mesh never calls. Adding one would modify a reported code path, so the
# divergence axis here is env_value (plus a genuine no-filter condition),
# which is the only shift magnitude the mesh environment can express.
# "none" = apply_colour_filter False (the pretraining render exactly).
COLOUR_CONDITIONS = [("none", None), ("v0.0", 0.0), ("v0.5", 0.5),
                     ("v1.0", 1.0), ("v1.5", 1.5)]


def build_env(world: str, condition, grids_shape, seed: int) -> GridEnvironment:
    T, G = grids_shape
    if world == "offset":
        grids = np.full((T, G, G), 0.5)
        return GridEnvironment(G, grids, rotation_seed=seed,
                               offset_field=build_dynamic_offset_field(G, T),
                               excluded_rotation_ranges=EXCLUDED_RANGES,
                               apply_colour_filter=False)
    value = condition
    grids = np.full((T, G, G), 0.5 if value is None else float(value))
    return GridEnvironment(G, grids, rotation_seed=seed,
                           excluded_rotation_ranges=EXCLUDED_RANGES,
                           apply_colour_filter=(value is not None))


def wearable_paths(anchor_only_cells, shared_cells, T, seed, with_student: bool):
    """Deterministic cyclic sweeps. The anchor's wearable NEVER enters the
    student's FOV, so the student receives no ground truth in the student /
    frozen arms. The direct arm adds a second wearable confined to the SHARED
    cells -- the same cells the student is peer-supervised on."""
    rng = np.random.default_rng(seed)
    a = list(anchor_only_cells)
    rng.shuffle(a)
    paths = [np.array([a[t % len(a)] for t in range(T)], dtype=int)]
    if with_student:
        s = list(shared_cells)
        rng.shuffle(s)
        paths.append(np.array([s[t % len(s)] for t in range(T)], dtype=int))
    return paths


def run_arm(arm: str, world: str, condition, seed: int, cfg, device, out_dir: Path):
    grids_ref = np.load(ENV / "environment_grids.npy")
    T, G = grids_ref.shape[0], grids_ref.shape[1]
    env = build_env(world, condition, (T, G), seed)

    centres = np.array([ANCHOR_CENTRE, STUDENT_CENTRE])
    a_cells = set(fov_cells(*ANCHOR_CENTRE, FOV, G))
    s_cells = set(fov_cells(*STUDENT_CENTRE, FOV, G))
    shared = sorted(a_cells & s_cells)
    anchor_only = sorted(a_cells - s_cells)
    assert shared and anchor_only, "the two FOVs must overlap but not coincide"

    mode = "frozen" if arm == "frozen" else "nig_product"
    paths = wearable_paths(anchor_only, shared, T, seed, with_student=(arm == "direct"))

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mesh = SamplingMesh(centres, fov_size=FOV, grid_size=G, environment=env,
                        pretrained_path=CKPT, lr=LR,
                        fusion_grid=circular_grid(cfg.fusion.grid_size), mode=mode,
                        device=device, sample_seed=seed, rho=cfg.consensus.rho,
                        lam=LAM,
                        sample_target=(arm == "student_sampled"),
                        target_seed=seed)
    anchor_id, student_id = 0, 1

    node_mse = np.full((T, 2), np.nan)                  # cell-space, per node
    stu_mse = np.full((T, G, G), np.nan)
    stu_cert = np.full((T, G, G), np.nan)
    anc_epi = np.full((T, G, G), np.nan)
    stu_epi = np.full((T, G, G), np.nan)
    cov_rows = []                                       # student interval coverage

    for step in range(T):
        mesh.run_timestep([p[step] for p in paths], step,
                          n_wearable_samples=N_WEARABLE_SAMPLES,
                          n_train_repeats=N_TRAIN_REPEATS)
        for nid, node in mesh.nodes.items():
            errs = []
            for cell, (g, nu, al, be) in node.cell_beliefs.items():
                _, truth = env.get_cell_input(cell[0], cell[1], step)
                e2 = circular_diff(torch.tensor(g), truth).item() ** 2
                errs.append(e2)
                epi = be / (max(nu, 1e-6) * max(al - 1.0, 1e-6))
                if nid == student_id:
                    stu_mse[step, cell[0], cell[1]] = e2
                    stu_cert[step, cell[0], cell[1]] = 1.0 / (1.0 + min(epi, 10.0))
                    stu_epi[step, cell[0], cell[1]] = epi
                    if step >= T - LAST_N:
                        scale = float(np.sqrt(be * (1.0 + nu) / (nu * al)))
                        cov_rows.append((float(np.sqrt(e2)), float(2 * al), scale))
                else:
                    anc_epi[step, cell[0], cell[1]] = epi
            if errs:
                node_mse[step, nid] = float(np.mean(errs))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "node_mse_steps.npy", node_mse)
    np.save(out_dir / "student_cell_mse_steps.npy", stu_mse)
    np.save(out_dir / "student_cell_cert_steps.npy", stu_cert)
    np.save(out_dir / "student_cell_epi_steps.npy", stu_epi)
    np.save(out_dir / "anchor_cell_epi_steps.npy", anc_epi)

    # ── student calibration and the inheritance diagnostic ────────────────
    from scipy import stats as sps
    r_mean, r_sd, n_t, _ = per_timestep_then_averaged(stu_mse, stu_cert, T - LAST_N, T)

    cov90 = None
    if cov_rows:
        e = np.array([r[0] for r in cov_rows]); dof = np.array([r[1] for r in cov_rows])
        sc = np.array([r[2] for r in cov_rows])
        ok = np.isfinite(sc) & np.isfinite(dof) & (dof > 2)
        half = sps.t.ppf(0.95, df=dof[ok]) * sc[ok]
        cov90 = {"empirical": float((e[ok] <= half).mean()),
                 "n": int(ok.sum()),
                 "frac_interval_covers_circle": float((half >= 1.0).mean()),
                 "mean_half_width_deg": float(np.mean(half) * 180.0)}

    own, inherited = [], []
    sh = np.array(shared)
    for step in range(T - LAST_N, T):
        se = stu_epi[step, sh[:, 0], sh[:, 1]]
        sm = stu_mse[step, sh[:, 0], sh[:, 1]]
        ae = anc_epi[step, sh[:, 0], sh[:, 1]]
        m = np.isfinite(se) & np.isfinite(sm) & np.isfinite(ae)
        if m.sum() >= 8 and np.std(se[m]) > 0:
            if np.std(sm[m]) > 0:
                own.append(sps.pearsonr(se[m], sm[m])[0])
            if np.std(ae[m]) > 0:
                inherited.append(sps.pearsonr(se[m], ae[m])[0])

    res = {
        "arm": arm, "world": world, "condition": str(condition), "seed": seed,
        "mode": mode,
        "whole_run_mse_cell_space": {
            "anchor": float(np.nanmean(node_mse[:, anchor_id])),
            "student": float(np.nanmean(node_mse[:, student_id]))},
        "last50_mse_cell_space": {
            "anchor": float(np.nanmean(node_mse[-LAST_N:, anchor_id])),
            "student": float(np.nanmean(node_mse[-LAST_N:, student_id]))},
        "student_cert_mse_r_per_timestep": {"mean": float(r_mean),
                                            "sd": float(r_sd), "n_timesteps": int(n_t)},
        "student_coverage_90": cov90,
        "sampled_target": {
            "enabled": bool(mesh.sample_target),
            "n_draws": len(mesh.sample_deviations_deg),
            "mean_abs_deviation_from_mode_deg": (
                float(np.mean(mesh.sample_deviations_deg))
                if mesh.sample_deviations_deg else None),
            "median_abs_deviation_from_mode_deg": (
                float(np.median(mesh.sample_deviations_deg))
                if mesh.sample_deviations_deg else None),
            "p90_abs_deviation_from_mode_deg": (
                float(np.percentile(mesh.sample_deviations_deg, 90))
                if mesh.sample_deviations_deg else None)},
        "uncertainty_source": {
            "r_student_epi_vs_own_error": (float(np.mean(own)) if own else None),
            "r_student_epi_vs_anchor_epi": (float(np.mean(inherited)) if inherited else None),
            "n_timesteps": len(own)},
    }
    manifest = {
        "stage": "section5_2", "condition": f"{world}_{condition}_{arm}_seed{seed}",
        "arm": arm, "mode": mode, "world": world, "colour_condition": str(condition),
        "seed": seed, "n_nodes": 2, "fov_size": FOV,
        "anchor_centre": list(ANCHOR_CENTRE), "student_centre": list(STUDENT_CENTRE),
        "n_shared_cells": len(shared), "n_anchor_only_cells": len(anchor_only),
        "total_steps": T, "lr": LR, "lam": LAM,
        "n_wearable_samples": N_WEARABLE_SAMPLES, "n_train_repeats": N_TRAIN_REPEATS,
        "temper_gradient": True, "uncertainty_measure": "epistemic",
        "excluded_rotation_ranges": [list(t) for t in EXCLUDED_RANGES],
        "excluded_rotation_ranges_applied": True,
        "baseline_checkpoint": str(CKPT),
        "metric_convention": "cell space: mean over (step, cell in node FOV) of "
                             "that node's squared wrapped error",
        "results": res,
    }
    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    return res


def main(world: str, arms):
    global ARMS
    ARMS = arms
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = COLOUR_CONDITIONS if world == "colour" else [("offset", None)]
    all_res = []
    for cname, cval in conditions:
        for arm in ARMS:
            for seed in SEEDS:
                out = OUT_ROOT / world / cname / f"{arm}_seed{seed}"
                r = run_arm(arm, world, cval, seed, cfg, device, out)
                all_res.append(r | {"condition_name": cname})
                print(f"[{world}/{cname}/{arm}/seed{seed}] "
                      f"whole anchor={r['whole_run_mse_cell_space']['anchor']:.5f} "
                      f"student={r['whole_run_mse_cell_space']['student']:.5f} "
                      f"last50 student={r['last50_mse_cell_space']['student']:.5f} "
                      f"r={r['student_cert_mse_r_per_timestep']['mean']:+.4f} "
                      f"cov90={(r['student_coverage_90'] or {}).get('empirical', float('nan')):.4f}",
                      flush=True)
    tag = "_".join(ARMS) if set(ARMS) != {"student", "frozen", "direct"} else "all"
    with open(OUT_ROOT / f"{world}_{tag}_results.json", "w") as f:
        json.dump(all_res, f, indent=1)
    print(f"wrote {OUT_ROOT / (world + '_' + tag + '_results.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["offset", "colour"], required=True)
    ap.add_argument("--arms", type=str, default="student,frozen,direct",
                    help="comma-separated subset of "
                         "student,frozen,direct,student_sampled")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    main(args.world, [a.strip() for a in args.arms.split(",")])
