# Tests wearable_policy="uncertainty_guided" (mesh.py/runner.py) against the
# existing random-wander baseline, on the same static-spatial v2 world used
# for every ablation today (CoordConv + excluded rotation ranges active).
#
# Uses lam=5 (not today's original 0.1 default): the routing autopsy found
# uncertainty-directed movement only works with honestly-calibrated
# uncertainty, and the lam ablation showed lam=5-10 is what gets fusion's
# cert-MSE correlation from ~-0.57 to ~-0.83. Testing the new routing policy
# on top of a config known to have bad-ish calibration would confound "does
# zoned/commit routing work" with "is there even a usable signal to route on".
#
# Arms:
#   random_wander      -- today's baseline: 3 independent random walks (replay)
#   uncertainty_guided -- 3 wearables, one fixed compact Voronoi region each
#                         (wearable_zones()), commit-then-reassess to the
#                         current uncertainty argmax in their own region
#
# Result (static-spatial v2, lam=5, seed=42): whole_run MSE ties the random
# baseline (0.0281 vs 0.0277) and beats it on the last-50-step window (0.0152
# vs 0.0208) -- targeted visits to high-uncertainty cells measurably help the
# model, confirming the mechanism works, even though cert-MSE r collapses to
# ~0 here (a probable range-restriction artifact: the policy actively
# depletes the high-uncertainty tail a correlation coefficient needs variance
# in -- expected to matter less once the world is temporally dynamic and new
# uncertainty keeps being generated). Kept as a permanent wearable_policy
# option (alongside None/random replay) for the upcoming dynamic-world stage.
#
# Run: python -u experiments/stage6_spatial_mesh/run_zoned_routing_test.py

from __future__ import annotations

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
FIELD = Path("runs/stage6/offset_world/dynamic_static_spatial_v2/field.npy")
ROOT = Path("runs/stage6/offset_world/zoned_routing_test")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]


def main():
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.load(FIELD)
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, wearable_policy):
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode="fusion", baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Zoned routing test ({tag})",
            offset_field=field, env_seed=SEED,
            excluded_rotation_ranges=EXCLUDED_RANGES,
            lam=LAM, wearable_policy=wearable_policy, policy_step_size=0.4,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f} r={r:.4f}")

    run("random_wander", None)
    run("uncertainty_guided", "uncertainty_guided")

    print("\n=== ZONED ROUTING RESULT (lam=5, lr=3e-5, seed=42) ===")
    for tag, (last50, whole, r) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f} r={r:.4f}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
