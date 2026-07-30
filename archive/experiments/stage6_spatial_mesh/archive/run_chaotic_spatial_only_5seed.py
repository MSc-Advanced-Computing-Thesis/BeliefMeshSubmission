# 5-seed replication of the winning 36-node config (run_chaotic_spatial_only_test.py):
# new-spatial/old-temporal field (wide/sharp/big-amplitude regions, n_keyframes=4
# for the original slow ~0.11 deg/step drift), lr=3e-4 (reverted after the
# LR-bump bug), lam=0.1 (default -- the lam=10 follow-up showed no real
# calibration gain, just wider decorative spread). Single-seed result was
# fusion=0.0531 vs fedavg_global=0.0674 (fusion +21%), the first and only
# clean fusion win of the whole day's investigation -- but every other
# single-run number today has moved substantially on reseed (established
# GPU-non-determinism + wearable/env-seed variance), so this needs a proper
# mean+-std before it's trustworthy enough to write up as evidence.
#
# The field itself is a FIXED task (built once, not resampled per seed, same
# convention as every other stage6 5-seed script) -- only the training seed
# (random/np/torch) and wearable path seeds vary per repeat.
#
# Runs ONE mode across all 5 seeds per invocation, so fusion and
# fedavg_global can run as two parallel background processes.
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_spatial_only_5seed.py --mode fusion
#      python -u experiments/stage6_spatial_mesh/run_chaotic_spatial_only_5seed.py --mode fedavg_global

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_chaotic_offset_field, build_random_wander_path

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_chaotic_spatial_only_5seed")
N_WEARABLES = 3
LR = 3e-4
DEFAULT_SEEDS = [42, 1042, 2042, 3042, 4042]


def main(mode: str, seeds: list[int]):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_chaotic_offset_field(G, T, n_keyframes=4)  # fixed task, not resampled per seed

    results = []
    for seed in seeds:
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
            title=f"Chaotic spatial-only 5-seed ({mode}, seed={seed})",
            offset_field=field, env_seed=seed,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        results.append(whole_run_mse)
        print(f"### {mode} seed={seed}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f}")

    results = np.array(results)
    print(f"\n=== {mode} 5-SEED SUMMARY ===")
    print(f"whole_run MSE: mean={results.mean():.4f} std={results.std():.4f} "
          f"values={list(np.round(results, 4))}")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fusion", "fedavg_global"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    main(args.mode, args.seeds)
