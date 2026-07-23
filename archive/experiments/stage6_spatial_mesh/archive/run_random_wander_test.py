# Follow-up to run_multi_wearable_test.py. Online routing (epistemic_
# staleness, any cooldown) collapsed all 3 wearables onto the same location
# (Christian: "they all end up in the same location") -- the shared signal
# pulls everyone toward the same strongest local maximum. Per Christian's
# direction: go back to the ORIGINAL colour-stage-6 style of wearable
# movement -- independent random walks, each started at a random position,
# just wandering (no uncertainty-driven targeting at all). Three of these,
# run simultaneously, same field, fusion vs global.
#
# The original environment_v2/wearable_path.npy is a single such walk
# (smoothed/momentum random walk, boundary-reflecting, mean step ~0.35,
# std ~0.20 -- measured from the saved array). Its generator script no
# longer exists in this repo, so build_random_wander_path() below
# reconstructs the same style (momentum + reflection) to produce 3
# independent walks with different seeds and random start positions.
#
# Run: python -u experiments/stage6_spatial_mesh/run_random_wander_test.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/dynamic_offset")
N_WEARABLES = 3


def build_random_wander_path(total_steps: int, grid_size: int, seed: int,
                             step_mean: float = 0.35, step_std: float = 0.20,
                             momentum: float = 0.85) -> np.ndarray:
    """Smoothed, momentum-driven random walk with boundary reflection --
    matches the measured step statistics of the original wearable_path.npy
    (mean step ~0.35, std ~0.20). Random start position each call (seeded)."""
    rng = np.random.default_rng(seed)
    lo, hi = 0.1, grid_size - 1.1
    pos = rng.uniform(lo + 2, hi - 2, size=2)
    velocity = rng.normal(0, step_mean, size=2)
    path = [pos.copy()]
    for _ in range(total_steps - 1):
        velocity = momentum * velocity + (1 - momentum) * rng.normal(0, step_mean, size=2)
        speed = np.linalg.norm(velocity)
        target_speed = max(rng.normal(step_mean, step_std), 0.02)
        if speed > 1e-6:
            velocity = velocity / speed * target_speed
        pos = pos + velocity
        for axis in range(2):
            if pos[axis] < lo:
                pos[axis] = lo + (lo - pos[axis])
                velocity[axis] *= -1
            elif pos[axis] > hi:
                pos[axis] = hi - (pos[axis] - hi)
                velocity[axis] *= -1
        path.append(pos.copy())
    return np.array(path)


def main():
    cfg = load_config()
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]
    for i, p in enumerate(paths):
        print(f"wearable {i} start {p[0].round(1)} bbox rows[{p[:,0].min():.1f},{p[:,0].max():.1f}] "
              f"cols[{p[:,1].min():.1f},{p[:,1].max():.1f}]")

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Dynamic offset world, {N_WEARABLES} random-wandering wearables ({tag})",
            offset_field=field,  # wearable_policy=None -> replays the given paths
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        results[tag] = (res["mean_mse_last_50"], whole_run_mse)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}")

    run("dynamic_fusion_randomwander3", "fusion")
    run("dynamic_global_randomwander3", "fedavg_global")

    print("\n=== RANDOM WANDER (3 wearables) TEST SUMMARY ===")
    for tag, (last50, whole) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}")
    print("(reference: routed 1-wearable fusion whole_run=0.0842, "
          "routed 3-wearable (clumped) fusion whole_run=0.0400, "
          "global (any routing so far) whole_run~0.025-0.027)")
    print("=== RANDOM WANDER TEST DONE ===")


if __name__ == "__main__":
    main()
