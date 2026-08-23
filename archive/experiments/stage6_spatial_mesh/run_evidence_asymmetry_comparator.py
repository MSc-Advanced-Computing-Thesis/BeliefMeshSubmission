# Evidence-asymmetry environments (2026-08): two deliberate constructions
# meant to break the spatial symmetry that made contributors to a shared
# cell carry near-identical evidence (nu) in the smooth offset world (CV
# 0.044, max-weight-minus-uniform 0.016 -- see the per-event nu diagnostic).
# Both otherwise reuse the standard offset-world harness (36 nodes, 7x7
# stride 3, random_wander except where the construction itself restricts it,
# lam=5, lr=3e-5, angle exclusion) unchanged.
#
# Construction "boundary": a piecewise offset field -- the normal smooth
# build_dynamic_offset_field() PLUS a sharp step discontinuity added on one
# side of a fixed column, positioned at the overlap band between node
# columns 9 and 12 (node centres sit at cols 3/6/9/12/15/18; FOV half=3, so
# nodes at 9 and 12 share cells in columns 9-12, straddling the cut). A node
# whose FOV lies wholly on one side is well-adapted to a stable local
# mapping; a node straddling the boundary sees genuinely conflicting labels
# from its own two sub-regions -- the debris-front/flood-edge case, not a
# smooth gradient.
#
# Construction "restricted_coverage": wearables are confined to the LEFT
# columns (0-10) for the whole run, via build_restricted_wander_path()
# (same momentum-driven random walk as build_random_wander_path, but with
# the column axis's reflecting boundary pulled in). Nodes at columns 12/15/18
# never receive a direct wearable visit -- they are trained ENTIRELY via BFS
# propagation from the covered side, at strictly greater hop distance and
# with far fewer direct anchor events than nodes on the covered side.
#
# Run: python -u experiments/stage6_spatial_mesh/run_evidence_asymmetry_comparator.py --construction boundary --mode nig_product --seeds 42

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.cert_mse_metrics import per_timestep_then_averaged
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/evidence_asymmetry_comparator")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODES = ["naive", "certainty", "nig_product"]
CONSTRUCTIONS = ["boundary", "restricted_coverage"]
BOUNDARY_COL = 11          # overlap band between node columns 9 and 12
GAP_DEG = 90.0             # step discontinuity magnitude
COL_MAX = 11               # restricted_coverage: wearables confined to cols 0-10


def build_boundary_offset_field(grid_size: int, total_steps: int, boundary_col: int = BOUNDARY_COL,
                                gap_deg: float = GAP_DEG, seed: int = 7) -> np.ndarray:
    base = build_dynamic_offset_field(grid_size, total_steps, seed=seed)  # (T,H,W) degrees, smooth
    step = np.zeros((grid_size, grid_size))
    step[:, boundary_col:] = gap_deg
    return base + step[None, :, :]


def build_restricted_wander_path(total_steps: int, grid_size: int, seed: int, col_max: int,
                                 step_mean: float = 0.35, step_std: float = 0.20,
                                 momentum: float = 0.85) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo_row, hi_row = 0.1, grid_size - 1.1
    lo_col, hi_col = 0.1, col_max - 1.1
    pos = np.array([rng.uniform(lo_row + 2, hi_row - 2), rng.uniform(lo_col + 2, hi_col - 2)])
    velocity = rng.normal(0, step_mean, size=2)
    path = [pos.copy()]
    for _ in range(total_steps - 1):
        velocity = momentum * velocity + (1 - momentum) * rng.normal(0, step_mean, size=2)
        speed = np.linalg.norm(velocity)
        target_speed = max(rng.normal(step_mean, step_std), 0.02)
        if speed > 1e-6:
            velocity = velocity / speed * target_speed
        pos = pos + velocity
        if pos[0] < lo_row:
            pos[0] = lo_row + (lo_row - pos[0]); velocity[0] *= -1
        elif pos[0] > hi_row:
            pos[0] = hi_row - (pos[0] - hi_row); velocity[0] *= -1
        if pos[1] < lo_col:
            pos[1] = lo_col + (lo_col - pos[1]); velocity[1] *= -1
        elif pos[1] > hi_col:
            pos[1] = hi_col - (pos[1] - hi_col); velocity[1] *= -1
        path.append(pos.copy())
    return np.array(path)


def run_one(construction: str, mode: str, seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    root = ROOT / construction
    root.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    wearable_seed_base = 200 if seed == SEED else seed + 200

    if construction == "boundary":
        field = build_boundary_offset_field(G, T)
        paths = [__import__("stage6_spatial_mesh.run_offset_experiments", fromlist=["build_random_wander_path"])
                .build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]
    elif construction == "restricted_coverage":
        field = build_dynamic_offset_field(G, T)
        paths = [build_restricted_wander_path(T, G, seed=wearable_seed_base + i, col_max=COL_MAX)
                for i in range(N_WEARABLES)]
    else:
        raise ValueError(construction)

    tag = f"agg_cmp_{mode}" if seed == SEED else f"agg_cmp_{mode}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=f"{construction}_{tag}", run_dir=root / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Evidence-asymmetry ({construction}, {mode}, seed={seed})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        track_disagreement=(mode == "nig_product"),
    )
    cell_mse_steps = np.load(root / tag / "cell_mse_steps.npy")
    cell_cert_steps = np.load(root / tag / "cell_cert_steps.npy")
    T = cell_mse_steps.shape[0]
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    new_mean, new_std, n_t, _ = per_timestep_then_averaged(cell_mse_steps, cell_cert_steps, T - 50, T)
    print(f"\n### {construction}/{tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} "
          f"r(new)={new_mean:+.4f}+/-{new_std:.4f} (n_t={n_t})")
    dstats = res.get("disagreement_stats")
    if dstats:
        print(f"    disagreement_stats: {dstats}")
    return whole_run_mse, res['mean_mse_last_50'], new_mean, new_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction", choices=CONSTRUCTIONS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seeds", type=str, default=str(SEED))
    args = parser.parse_args()
    for seed in [int(s) for s in args.seeds.split(",")]:
        run_one(args.construction, args.mode, seed)
    print("=== DONE ===")
