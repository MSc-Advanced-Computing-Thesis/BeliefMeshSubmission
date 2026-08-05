# Closed-form product-of-NIG fusion (nig_product / nig_product_weighted,
# 2026-08) vs the existing grid-search fusion arm, on the SAME dynamic-world
# field/seed/wearable-policy/node-placement as run_gossip_comparator.py, so
# results sit on the same axis and the existing "fusion" seed-42/5-seed data
# (runs/stage6/offset_world/gossip_comparator/gossip_cmp_fusion*) is reused
# directly as the comparison baseline rather than rerun.
#
# Verification (single-contributor identity, two-identical-contributors
# agreement, beta_star-increases-under-disagreement, nu_star/alpha_star
# guards, circular-wraparound safety) lives in tests/test_nig_product.py and
# passes before this script is ever run -- see fusion/nig_product.py for the
# closed-form derivation.
#
# Run: python -u experiments/stage6_spatial_mesh/run_nig_product_comparator.py --mode nig_product --seeds 42
#   where <mode> in {nig_product, nig_product_weighted}

from __future__ import annotations

import argparse
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
ROOT = Path("runs/stage6/offset_world/nig_product_comparator")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODES = ["nig_product", "nig_product_weighted"]


def run_one(mode: str, seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)  # deterministic, same field every arm/seed
    # same convention as run_gossip_comparator.py / run_dynamic_v2_5seed.py:
    # seed 42 keeps the original wander paths (200+i).
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    tag = f"nig_cmp_{mode}" if seed == SEED else f"nig_cmp_{mode}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"nig_product comparator ({mode}, seed={seed})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    r = res.get("certainty_mse_pearson_r")
    fusion_time_total = res["fusion_time_total_sec"]
    fusion_calls = res["fusion_call_count"]
    fusion_time_per_call = res["fusion_time_mean_per_call_sec"]
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} "
          f"r={r:.4f}")
    print(f"    fusion_time_total={fusion_time_total:.4f}s over {fusion_calls} calls "
          f"({fusion_time_per_call*1e6:.2f} us/call)")
    return whole_run_mse, res['mean_mse_last_50'], r, fusion_time_total, fusion_calls


def main(mode: str, seeds: list[int]):
    results = []
    for seed in seeds:
        results.append(run_one(mode, seed))
    if len(seeds) > 1:
        arr = np.array([(r[0], r[1], r[2]) for r in results])
        print(f"\n=== {mode} {len(seeds)}-SEED SUMMARY (seeds={seeds}) ===")
        print(f"whole_run MSE: mean={arr[:,0].mean():.4f} std={arr[:,0].std():.4f} "
              f"values={list(np.round(arr[:,0],4))}")
        print(f"last50 MSE:    mean={arr[:,1].mean():.4f} std={arr[:,1].std():.4f} "
              f"values={list(np.round(arr[:,1],4))}")
        print(f"r:             mean={arr[:,2].mean():.4f} std={arr[:,2].std():.4f} "
              f"values={list(np.round(arr[:,2],4))}")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seeds", type=str, default=str(SEED),
                        help="comma-separated seed list, e.g. '42,1042,2042,3042,4042'")
    args = parser.parse_args()
    main(args.mode, [int(s) for s in args.seeds.split(",")])
