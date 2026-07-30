# Diagnostic pilot for the chaotic offset world (run_chaotic_offset_test.py):
# the first full run showed fusion's hop0 (ground-truth-trained, no fusion or
# propagation involved) flat and noisy around 0.10-0.17 MSE for the whole
# run, ~17x worse than the theoretical per-node floor (0.0076) -- not a
# fusion/propagation problem, a base convergence problem. Revisit frequency
# was ruled out directly (median gap between visits to any node = 1 step).
# The remaining lever: n_train_repeats was left at 1 -- a single gradient
# step per visit was enough to track the original field's 0.06 deg/step
# drift, nowhere near enough for this field's 2.4 deg/step drift and up to
# 100 deg swings. This bumps n_train_repeats to 4 and runs fusion ONLY
# (cheapest arm) to check whether hop0 actually converges before re-running
# the full 3-arm comparison.
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_pilot.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_chaotic_offset_field,
                                                         build_random_wander_path,
                                                         dynamic_analytic_floors)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_chaotic")
N_WEARABLES = 3
LR = 1e-3
N_TRAIN_REPEATS = 4


def main():
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.load(ROOT / "chaotic_field.npy")  # reuse exact field from the killed run

    gf, nf = dynamic_analytic_floors(field, centres, 7, G, window=50)
    print(f"analytic floors: global={gf:.4f} per_node={nf:.4f} (theoretical best case)")
    print(f"n_train_repeats={N_TRAIN_REPEATS} (was 1), lr={LR}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = "chaotic_fusion_pilot_repeats4"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="fusion", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=N_TRAIN_REPEATS,
        title=f"Chaotic pilot, n_train_repeats={N_TRAIN_REPEATS}",
        offset_field=field,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}")
    print(f"(reference, n_train_repeats=1 attempt: hop0 flat ~0.10-0.17, whole_run=0.1388)")
    print(f"(theoretical per-node floor: {nf:.4f})")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
