# 5-seed replication of the "best guess" consensus config from the lam/rho
# sweep: lam=0.1 (default -- the lam sweep looked like pure noise, no
# justification to move it), rho=0.05 (the most-often-good point across
# three separate measurements: r=-0.5682, -0.4521, -0.3650 -- noisy, but
# directionally still ahead of the default rho=0.2's single reading of
# -0.2308 even at its worst).
#
# Motivation: a same-seed, same-config rerun during the sweep showed r
# swinging from -0.5682 to -0.3650 purely from GPU floating-point
# non-determinism compounding over 390 sequential training steps (confirmed
# directly -- a 15-step repro showed a tiny per-step divergence, not a
# scripting bug). Every single-run point in the whole sweep therefore carries
# BOTH this GPU noise and the already-known wearable/env-seed variance. This
# script runs the chosen config across the standard 5 seeds
# (42/1042/2042/3042/4042) to get a proper mean+-std before trusting it.
#
# Run: python -u experiments/stage6_spatial_mesh/run_best_consensus_config_5seed.py

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_best_consensus_5seed")
N_WEARABLES = 3
LAM = 0.1
RHO = 0.05
SEEDS = [42, 1042, 2042, 3042, 4042]


def main():
    cfg = load_config()
    seed = int(os.environ.get("EXP_SEED", cfg.seed))
    root = ROOT
    if seed != cfg.seed:
        root = root.parent / f"{root.name}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)  # fixed task, not resampled per seed

    wearable_seed_base = 200 if seed == cfg.seed else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
             for i in range(N_WEARABLES)]

    tag = "best_consensus"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=root / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="consensus", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Best-guess consensus (lam={LAM}, rho={RHO}), seed={seed}",
        offset_field=field, env_seed=seed, lam=LAM, rho=RHO,
    )
    cell_mse_steps = np.load(root / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    print(f"### seed={seed}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")


if __name__ == "__main__":
    main()
