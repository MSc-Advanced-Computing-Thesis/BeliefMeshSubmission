# Corrected trivial baseline for the static-spatial comparison. Christian
# caught that the earlier "best constant guess" baseline (MSE~0.0347) was
# computed against the offset field alone, not the real evaluation target --
# every cell/step's ground truth is (independently random rotation + offset),
# per GridEnvironment.cell_rotations (rng.uniform(-180,180) per (step,row,col)),
# so a constant guess can never track the rotation component at all and isn't
# a fair floor. The correct trivial comparator is `frozen` mode: the
# pretrained checkpoint, zero further training -- it already knows how to
# read rotation (Stage 0 baseline ~0.0123 MSE) but has no idea about the
# location-dependent offset. This gives the real "doing nothing" number to
# compare fusion (0.0464 pre-fix) and fedavg_global (0.0517) against.
#
# Same static field.npy, same seed=42, same 36-node/3-wearable layout as
# today's other static-spatial runs (wearable presence only matters for
# frozen mode insofar as it doesn't train on it -- predict_cells runs for
# every node's own FOV regardless of anchor status, so all cells still get
# evaluated).
#
# Run: python -u experiments/stage6_spatial_mesh/run_static_spatial_frozen_baseline.py

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
ROOT = Path("runs/stage6/offset_world/dynamic_static_spatial")
N_WEARABLES = 3
LR = 3e-4


def main():
    cfg = load_config()
    cfg.model.lr = LR

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.load(ROOT / "field.npy")  # exact same static field as today's other runs

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = "static_spatial_frozen"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="frozen", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Static spatial-only, frozen baseline ({tag})",
        offset_field=field,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")
    print("(reference, pre-fix fusion: whole_run=0.0464)")
    print("(reference, fedavg_global: whole_run=0.0517)")
    print("(this is the 'zero further training' floor -- fusion/fedavg should beat THIS, "
          "not a constant-offset guess)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
