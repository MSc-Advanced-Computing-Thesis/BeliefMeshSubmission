# Uncertainty-measure ablation (2026-08): which quantity feeds the fused
# certainty that tempers the training gradient under nig_product -- epistemic
# (beta/(nu*(alpha-1)), the current default), aleatoric (beta/(alpha-1), no
# nu term), or total predictive variance (beta*(1+nu)/(nu*(alpha-1)), their
# sum). Same offset-world field/seed/wearable-policy/node-placement as
# run_nig_product_comparator.py, so results sit on the same axis; "epistemic"
# reuses that script's nig_cmp_nig_product* data as its baseline rather than
# rerunning it, since Mesh(uncertainty_measure="epistemic") is verified
# (tests/test_mesh.py::test_uncertainty_measure_epistemic_is_default_and_
# matches_unspecified) to reproduce plain nig_product bit-for-bit.
#
# Run: python -u experiments/stage6_spatial_mesh/run_uncertainty_measure_ablation.py --measure aleatoric --seeds 42

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
ROOT = Path("runs/stage6/offset_world/uncertainty_measure_ablation")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MEASURES = ["epistemic", "aleatoric", "total"]


def run_one(measure: str, seed: int):
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

    tag = f"unc_cmp_{measure}" if seed == SEED else f"unc_cmp_{measure}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="nig_product", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"uncertainty-measure ablation ({measure}, seed={seed})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        uncertainty_measure=measure,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    r = res.get("certainty_mse_pearson_r")
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} r={r:.4f}")
    return whole_run_mse, res['mean_mse_last_50'], r


def main(measure: str, seeds: list[int], root: str | None = None):
    global ROOT
    if root:
        ROOT = Path(root)
    results = []
    for seed in seeds:
        results.append(run_one(measure, seed))
    if len(seeds) > 1:
        arr = np.array(results)
        print(f"\n=== {measure} {len(seeds)}-SEED SUMMARY (seeds={seeds}) ===")
        print(f"whole_run MSE: mean={arr[:,0].mean():.4f} std={arr[:,0].std():.4f} "
              f"values={list(np.round(arr[:,0],4))}")
        print(f"last50 MSE:    mean={arr[:,1].mean():.4f} std={arr[:,1].std():.4f} "
              f"values={list(np.round(arr[:,1],4))}")
        print(f"r:             mean={arr[:,2].mean():.4f} std={arr[:,2].std():.4f} "
              f"values={list(np.round(arr[:,2],4))}")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", choices=MEASURES, required=True)
    parser.add_argument("--seeds", type=str, default=str(SEED),
                        help="comma-separated seed list, e.g. '42,1042,2042,3042,4042'")
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    args = parser.parse_args()
    main(args.measure, [int(s) for s in args.seeds.split(",")], args.root)