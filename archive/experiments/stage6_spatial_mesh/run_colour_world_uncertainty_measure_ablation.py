# Colour-world repeat of run_uncertainty_measure_ablation.py (2026-08): same
# three measures (epistemic/aleatoric/total), same node placement/wearable
# convention/seed as run_colour_world_nig_product_comparator.py -- the only
# change from the offset-world ablation is the environment grid (real
# environment_v2/environment_grids.npy colour field instead of synthetic
# uniform+offset), matching that script's own reasoning for the switch.
#
# Run: python -u experiments/stage6_spatial_mesh/run_colour_world_uncertainty_measure_ablation.py --measure aleatoric --seeds 42

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
ROOT = Path("runs/stage6/colour_world/uncertainty_measure_ablation")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MEASURES = ["epistemic", "aleatoric", "total"]


def run_one(measure: str, seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    all_grids = np.load(ENV / "environment_grids.npy")
    T, G = all_grids.shape[0], all_grids.shape[1]
    centres = np.load(ENV / "node_centres.npy")
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    tag = f"colour_unc_{measure}" if seed == SEED else f"colour_unc_{measure}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=all_grids, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="nig_product", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"colour-world uncertainty-measure ablation ({measure}, seed={seed})",
        env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        uncertainty_measure=measure,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    r = res.get("certainty_mse_pearson_r")
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} r={r:.4f}")
    stats = res.get("applied_uncertainty_stats")
    if stats:
        print(f"    applied_uncertainty_stats: {stats}")
    return whole_run_mse, res['mean_mse_last_50'], r


def main(measure: str, seeds: list[int], root: str | None = None):
    global ROOT
    if root:
        ROOT = Path(root)
    for seed in seeds:
        run_one(measure, seed)
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", choices=MEASURES, required=True)
    parser.add_argument("--seeds", type=str, default=str(SEED))
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    args = parser.parse_args()
    main(args.measure, [int(s) for s in args.seeds.split(",")], args.root)