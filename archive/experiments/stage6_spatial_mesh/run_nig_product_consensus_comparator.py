# nig_product_consensus (closed-form product-of-NIG fusion + live JS-
# divergence consensus tempering, 2026-08) vs plain nig_product, on the SAME
# offset-world field/seed/wearable-policy/node-placement as
# run_nig_product_comparator.py, so the existing nig_product seed-42/5-seed
# data (runs/stage6/offset_world/nig_product_comparator/nig_cmp_nig_product*)
# is reused directly as the comparison baseline rather than rerun.
#
# rho=0.2 is the codebase-wide default consensus EMA rate (Mesh.__init__'s
# default, also run_ablation_grid.py's DEFAULT_RHO) -- used here rather than
# a value picked from the existing lam/rho ablation grid, because that grid
# predates the CoordConv fix (see THESIS_EXPERIMENTS_SUMMARY.md "Next steps"
# item 2) and cannot be trusted to identify a "best" rho under the current
# architecture. Using the standing default keeps this run uncherry-picked.
#
# Verification that nig_product_consensus reduces to nig_product_weighted's
# fused parameters when self.consensus is pinned at unity, and that its
# agreement score drops under contributor disagreement, lives in
# tests/test_mesh.py (test_nig_product_consensus_matches_weighted_when_
# consensus_unity, test_nig_product_consensus_agreement_drops_under_
# disagreement) and passes before this script is ever run.
#
# Run: python -u experiments/stage6_spatial_mesh/run_nig_product_consensus_comparator.py --seeds 42

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
ROOT = Path("runs/stage6/offset_world/nig_product_consensus_comparator")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
RHO = 0.2
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODE = "nig_product_consensus"


def run_one(seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    tag = f"nig_cmp_{MODE}" if seed == SEED else f"nig_cmp_{MODE}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=MODE, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"nig_product_consensus comparator (seed={seed})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, rho=RHO, wearable_policy=None, policy_step_size=0.4,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    r = res.get("certainty_mse_pearson_r")
    final_consensus = res["final_consensus"]
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} "
          f"r={r:.4f}")
    cvals = np.array(list(final_consensus.values()))
    print(f"    final_consensus: mean={cvals.mean():.4f} std={cvals.std():.4f} "
          f"min={cvals.min():.4f} max={cvals.max():.4f}")
    return whole_run_mse, res['mean_mse_last_50'], r


def main(seeds: list[int]):
    results = []
    for seed in seeds:
        results.append(run_one(seed))
    if len(seeds) > 1:
        arr = np.array(results)
        print(f"\n=== {MODE} {len(seeds)}-SEED SUMMARY (seeds={seeds}) ===")
        print(f"whole_run MSE: mean={arr[:,0].mean():.4f} std={arr[:,0].std():.4f} "
              f"values={list(np.round(arr[:,0],4))}")
        print(f"last50 MSE:    mean={arr[:,1].mean():.4f} std={arr[:,1].std():.4f} "
              f"values={list(np.round(arr[:,1],4))}")
        print(f"r:             mean={arr[:,2].mean():.4f} std={arr[:,2].std():.4f} "
              f"values={list(np.round(arr[:,2],4))}")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=str(SEED),
                        help="comma-separated seed list, e.g. '42,1042,2042,3042,4042'")
    args = parser.parse_args()
    main([int(s) for s in args.seeds.split(",")])
