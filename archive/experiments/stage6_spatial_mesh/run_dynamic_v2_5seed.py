# 5-seed replication of the dynamic-world v2 result (run_dynamic_v2_test.py).
#
# RESULT (fusion, both policies, already complete):
#   fusion_random:              mean=0.0215 std=0.0015 (0.0208/0.0214/0.0221/0.0193/0.0238)
#   fusion_uncertainty_guided:  mean=0.0210 std=0.0017 (0.0195/0.0211/0.0242/0.0203/0.0197)
# Essentially tied -- routing's single-seed MSE edge did not survive 5-seed
# averaging as a clear win, though both are comfortably ahead of fedavg's
# single-seed reference (0.0237).
#
# This invocation: fedavg_global+random_wander for seeds 1042/2042/3042/4042
# (--skip-seed42, since seed 42 is already done from the original single-seed
# test: whole_run=0.0237 r=-0.118) -- completing a MATCHED 5-seed set (same
# field, same per-seed wearable-path seeds as the fusion batches above) so a
# proper paired significance test can be run across all three arms.
#
# Same field every seed (deterministic generator, not resampled -- it's the
# validated task, not something to average over); only training seed and
# wearable-walk seeds vary per repeat, matching every other 5-seed script
# today.
#
# Run:
#   python -u experiments/stage6_spatial_mesh/run_dynamic_v2_5seed.py --mode fedavg_global --skip-seed42

from __future__ import annotations

import argparse
import csv
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
ROOT = Path("runs/stage6/offset_world/dynamic_v2_5seed")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
SEEDS = [42, 1042, 2042, 3042, 4042]


def main(mode: str, policy: str, skip_seed42: bool, seeds_override: list[int] | None = None):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)  # deterministic, fixed across all seeds
    wearable_policy = None if policy == "random" else policy
    seeds = seeds_override if seeds_override is not None else \
            [s for s in SEEDS if not (skip_seed42 and s == 42)]

    results = []
    for seed in seeds:
        wearable_seed_base = 200 if seed == cfg.seed else seed + 200
        paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
                 for i in range(N_WEARABLES)]

        tag = f"{mode}_{policy}_seed{seed}"
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Dynamic v2 5-seed ({mode}, {policy}, seed={seed})",
            offset_field=field, env_seed=seed,
            excluded_rotation_ranges=EXCLUDED_RANGES,
            lam=LAM, wearable_policy=wearable_policy, policy_step_size=0.4,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        results.append((whole_run_mse, r))
        print(f"### {mode}_{policy} seed={seed}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f} r={r:.4f}")

    mses = np.array([x[0] for x in results])
    rs = np.array([x[1] for x in results])
    print(f"\n=== {mode}_{policy} {len(seeds)}-SEED SUMMARY (this invocation) ===")
    print(f"whole_run MSE: mean={mses.mean():.4f} std={mses.std():.4f} values={list(np.round(mses,4))}")
    print(f"r: mean={rs.mean():.4f} std={rs.std():.4f} values={list(np.round(rs,4))}")
    if skip_seed42:
        print("(seed 42 not re-run: whole_run=0.0237 r=-0.118, from the original single-seed test)")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fusion", "fedavg_global"], default="fusion")
    parser.add_argument("--policy", choices=["random", "uncertainty_guided"], default="random")
    parser.add_argument("--skip-seed42", action="store_true",
                        help="skip seed 42 (already run separately) -- combine manually at analysis time")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated explicit seed list, e.g. '2042,3042' -- "
                             "overrides --skip-seed42, for splitting a batch across parallel processes")
    args = parser.parse_args()
    seeds_override = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    main(args.mode, args.policy, args.skip_seed42, seeds_override)
