# Generic 5-seed replication runner for a given (lam, rho) consensus config.
# Consolidates run_best_consensus_config_5seed.py into a reusable form so we
# can compare this morning's untuned default (lam=0.1, rho=0.2) against
# today's best candidate (lam=10.0, rho=0.99) on equal footing -- neither
# had a proper 5-seed number before now, only single noisy runs.
#
# Run: EXP_SEED=42 python -u experiments/stage6_spatial_mesh/run_5seed_config.py --lam 0.1 --rho 0.2 --tag default_baseline
#      EXP_SEED=42 python -u experiments/stage6_spatial_mesh/run_5seed_config.py --lam 10.0 --rho 0.99 --tag lam10_rho099

from __future__ import annotations

import argparse
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
N_WEARABLES = 3


def main(lam: float, rho: float, tag: str):
    cfg = load_config()
    seed = int(os.environ.get("EXP_SEED", cfg.seed))
    root = Path(f"runs/stage6/offset_world/dynamic_5seed_{tag}")
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

    run_tag = tag
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=run_tag, run_dir=root / run_tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="consensus", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"{tag} (lam={lam}, rho={rho}), seed={seed}",
        offset_field=field, env_seed=seed, lam=lam, rho=rho,
    )
    cell_mse_steps = np.load(root / run_tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    print(f"### {tag} seed={seed}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()
    main(args.lam, args.rho, args.tag)
