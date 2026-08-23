# Colour-world generalisation check for nig_product (2026-08): same node
# placement (7x7 stride 3, environment_v2/node_centres.npy), same
# random_wander wearable-generation convention, same seed/step count/rotation
# exclusion as run_nig_product_comparator.py's offset-world runs -- the ONLY
# thing that changes is which grid the environment actually renders:
# environment_v2/environment_grids.npy (the real colour/red-blue field this
# directory was built for) instead of a synthetic uniform 0.5 grid + a
# separately-generated offset field. No colour-specific pretrained checkpoint
# exists anywhere in this repo (checked: only stage0's width/seed variants of
# pretrained_digit7.pth) -- every existing colour-world Stage 6 script
# (run_6d_three_way_comparison.py, run_dynamic_v2_consensus.py) uses the same
# plain-digit checkpoint, so this does too. Also runs mode="frozen" (no
# training at all) at the same config, as the "appropriate pretrained
# baseline for the colour world" -- the honest zero-adaptation reference
# point, since there is no colour-adapted checkpoint to compare against
# instead.
#
# Run: python -u experiments/stage6_spatial_mesh/run_colour_world_nig_product_comparator.py --modes frozen,nig_product --seeds 42

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_random_wander_path

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/colour_world/nig_product_comparator")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODES = ["frozen", "nig_product"]


def run_one(mode: str, seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    all_grids = np.load(ENV / "environment_grids.npy")  # real colour field, NOT synthetic
    T, G = all_grids.shape[0], all_grids.shape[1]
    centres = np.load(ENV / "node_centres.npy")
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    tag = f"colour_cmp_{mode}" if seed == SEED else f"colour_cmp_{mode}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=all_grids, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"colour-world comparator ({mode}, seed={seed})",
        env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    r = res.get("certainty_mse_pearson_r")
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} r={r:.4f}")
    stats = res.get("applied_uncertainty_stats")
    if stats:
        print(f"    applied_uncertainty_stats: {stats}")
    return whole_run_mse, res['mean_mse_last_50'], r


def main(modes: list[str], seeds: list[int]):
    for mode in modes:
        for seed in seeds:
            run_one(mode, seed)
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=str, default=",".join(MODES))
    parser.add_argument("--seeds", type=str, default=str(SEED))
    args = parser.parse_args()
    main(args.modes.split(","), [int(s) for s in args.seeds.split(",")])
