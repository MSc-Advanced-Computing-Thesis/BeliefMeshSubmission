# Rerun of run_static_spatial_test.py's fusion arm with the self-consistency
# fix active (mesh.py: a receiving node's own prior belief is now included
# as an extra contributor when fusing propagated beliefs, instead of being
# fully overwritten toward a single neighbour's current belief -- Spec: the
# echo-chamber-drift fix, diagnosed and validated via the triplet
# visualisation on this exact same field). fedavg_global is unaffected by
# this change (it never calls Mesh._aggregate at all -- separate code path),
# so it isn't rerun; comparing against the existing saved result
# (whole_run=0.0517).
#
# Original (pre-fix) fusion result on this field: last50=0.0632,
# whole_run=0.0464, r=+0.1577 (wrong-signed) -- won on whole_run, lost on
# last50, backwards certainty correlation. This is the direct test of
# whether the fix changes that picture.
#
# Reuses the exact saved field.npy, same seed=42, lr=3e-4, lam=0.1.
#
# Run: python -u experiments/stage6_spatial_mesh/run_static_spatial_selfconsistency_test.py

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
    field = np.load(ROOT / "field.npy")  # exact same static field as before

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = "static_spatial_fusion_selfconsistency"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="fusion", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Static spatial-only, self-consistency fix ({tag})",
        offset_field=field,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")
    print("(reference, pre-fix fusion: last50=0.0632 whole_run=0.0464 r=+0.1577)")
    print("(reference, fedavg_global (unaffected by fix): last50=0.0605 whole_run=0.0517 r=-0.1021)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
