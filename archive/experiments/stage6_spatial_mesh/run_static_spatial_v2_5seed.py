# 5-seed replication of run_static_spatial_v2_test.py's single-run result
# (fusion 0.0289 vs fedavg_global 0.0504, ~43% gap, the first clean,
# decisive fusion win of the whole investigation) -- with CoordConv
# (local-FOV coordinate channels, mesh.py), excluded_rotation_ranges
# (removes a confirmed visual ambiguity in the base digit), and lr=3e-5
# (confirmed to cut steady-state noise 3-6x on well-evidenced cells) all
# active. Given everything learned today about GPU non-determinism and
# seed-to-seed variance corrupting single-run conclusions, this needs
# proper averaging before being trusted as a settled result.
#
# Runs ONE mode across all 5 standard seeds per invocation (fusion,
# fedavg_global), so both can run as parallel background processes. Same
# static field (fixed task, not resampled per seed), only training seed and
# wearable-path seeds vary per repeat -- same convention as every other
# stage6 5-seed script today.
#
# Run: python -u experiments/stage6_spatial_mesh/run_static_spatial_v2_5seed.py --mode fusion
#      python -u experiments/stage6_spatial_mesh/run_static_spatial_v2_5seed.py --mode fedavg_global

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
FIELD = Path("runs/stage6/offset_world/dynamic_static_spatial/field.npy")  # exact same static field as v1/v2
ROOT = Path("runs/stage6/offset_world/dynamic_static_spatial_v2_5seed")
N_WEARABLES = 3
LR = 3e-5
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
SEEDS = [42, 1042, 2042, 3042, 4042]


def main(mode: str):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.load(FIELD)  # fixed task, not resampled per seed

    results = []
    for seed in SEEDS:
        wearable_seed_base = 200 if seed == cfg.seed else seed + 200
        paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
                 for i in range(N_WEARABLES)]

        tag = f"{mode}_seed{seed}"
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Static spatial v2 5-seed ({mode}, seed={seed})",
            offset_field=field, env_seed=seed,
            excluded_rotation_ranges=EXCLUDED_RANGES,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        results.append((whole_run_mse, r))
        print(f"### {mode} seed={seed}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f} r={r:.4f}")

    mses = np.array([x[0] for x in results])
    rs = np.array([x[1] for x in results])
    print(f"\n=== {mode} 5-SEED SUMMARY ===")
    print(f"whole_run MSE: mean={mses.mean():.4f} std={mses.std():.4f} values={list(np.round(mses,4))}")
    print(f"r: mean={rs.mean():.4f} std={rs.std():.4f} values={list(np.round(rs,4))}")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fusion", "fedavg_global"], required=True)
    args = parser.parse_args()
    main(args.mode)
