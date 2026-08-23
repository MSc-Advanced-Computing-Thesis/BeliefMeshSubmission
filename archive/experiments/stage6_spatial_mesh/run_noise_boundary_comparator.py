# Spatially-varying observation noise (2026-08): tests whether a node's
# accumulated evidence (nu) genuinely differs from a neighbour's when one
# covers a quiet region and the other a noisy one, since that -- not visual
# diversity, not a difficulty gradient -- is the condition under which
# fusion diverges from naive aggregation (established this session via the
# per-event nu diagnostic).
#
# Construction: GridEnvironment(noise_field=...) (2026-08 addition, opt-in,
# default None) adds Gaussian jitter to the TRAINING label only, inside
# get_multiple_rotations() -- never to get_cell_input()'s label, which is
# what both evaluation and propagation's rendering use, so reported MSE
# always reflects the true angle, never the jittered training signal.
#
# Field: two contiguous regions split at column 11 -- REUSING the exact same
# boundary column the earlier "boundary" offset-field construction used,
# because that boundary was already verified there to sit in a genuinely
# high-overlap band (nodes at columns 9/12 straddling it have mean degree
# 19.0, above the mesh's overall mean of 15.0 -- reconfirmed below). Quiet
# side (col < 11): noise_std=0.02 (~3.6 deg). Noisy side (col >= 11):
# noise_std=0.2 (~36 deg) -- a 10x ratio, matching "roughly an order of
# magnitude" between quiet and noisy regions.
#
# Run: python -u experiments/stage6_spatial_mesh/run_noise_boundary_comparator.py --mode nig_product --seeds 42

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
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config
from beliefmesh.node.mesh import fov_cells

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/noise_boundary_comparator")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODES = ["naive", "certainty", "nig_product"]
BOUNDARY_COL = 11
QUIET_STD = 0.02
NOISY_STD = 0.20


def build_noise_field(grid_size: int) -> np.ndarray:
    field = np.full((grid_size, grid_size), QUIET_STD)
    field[:, BOUNDARY_COL:] = NOISY_STD
    return field


def verify_boundary_overlap(centres: np.ndarray, grid_size: int) -> float:
    fov_sets = [set(fov_cells(int(cx), int(cy), 7, grid_size)) for cx, cy in centres]
    degrees = np.zeros(len(centres), dtype=int)
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            if fov_sets[i] & fov_sets[j]:
                degrees[i] += 1; degrees[j] += 1
    straddle = [i for i, (cx, cy) in enumerate(centres) if cx in (9, 12)]
    mean_straddle = float(degrees[straddle].mean())
    mean_overall = float(degrees.mean())
    print(f"[boundary check] mean degree at columns 9/12 (straddling col {BOUNDARY_COL}) = "
          f"{mean_straddle:.1f}  vs overall mean = {mean_overall:.1f}")
    return mean_straddle


def run_one(mode: str, seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    noise_field = build_noise_field(G)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    tag = f"agg_cmp_{mode}" if seed == SEED else f"agg_cmp_{mode}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=f"noise_boundary_{tag}", run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Noise-boundary comparator ({mode}, seed={seed})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        noise_field=noise_field, noise_seed=seed,
        track_disagreement=(mode == "nig_product"),
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    cell_cert_steps = np.load(ROOT / tag / "cell_cert_steps.npy")
    T = cell_mse_steps.shape[0]
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    new_mean, new_std, n_t, _ = per_timestep_then_averaged(cell_mse_steps, cell_cert_steps, T - 50, T)
    print(f"\n### noise_boundary/{tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} "
          f"r(new)={new_mean:+.4f}+/-{new_std:.4f} (n_t={n_t})")
    dstats = res.get("disagreement_stats")
    if dstats:
        print(f"    disagreement_stats: {dstats}")
    return whole_run_mse, res['mean_mse_last_50'], new_mean, new_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seeds", type=str, default=str(SEED))
    args = parser.parse_args()
    centres = np.load(ENV / "node_centres.npy")
    verify_boundary_overlap(centres, 22)
    for seed in [int(s) for s in args.seeds.split(",")]:
        run_one(args.mode, seed)
    print("=== DONE ===")
