# Objective #2 (2026-08): decentralised parameter-exchange comparator.
# Isolates the unit of exchange as the ONLY variable -- everything else (node
# placement 7x7 FOV/stride 3, wearable motion, environment schedule, timestep
# count, in-coverage rule, BFS order, lr, optimiser) is identical across arms:
#
#   fusion           -- belief exchange (NIG params), the system under test
#   gossip_uniform   -- parameter exchange, receiver = unweighted mean of
#                       trained overlap neighbours' full model weights
#   gossip_weighted  -- same, weighted by each contributor's accumulated
#                       n_samples_trained (FedAvg-style, the "stronger"
#                       parametric baseline)
#   fedavg_global    -- canonical FedAvg: one shared model, averaged from
#                       this round's anchors, broadcast to every node
#
# All FOUR arms use wearable_policy=None (random_wander) -- NOT
# uncertainty_guided -- deliberately, so routing isn't a second confounded
# variable on top of the exchange-mechanism comparison this script exists to
# isolate.
#
# Same dynamic-world field as every other dynamic_v2 script this session
# (build_dynamic_offset_field, default params -- the "highly dynamic
# environment" referred to in the spec), same Stage 6 fixes (CoordConv,
# angle exclusion, lam=5, lr=3e-5) applied throughout.
#
# Communication volume (objective #2's other deliverable) is logged
# automatically by runner.py/mesh.py's comm_bytes instrumentation --
# comm_bytes_steps.npy (per-step) and manifest.yaml's comm_bytes_total /
# comm_bytes_mean_per_step (cumulative) need no separate handling here.
#
# Single seed=42 first (sanity check) -- replicate across 5 seeds only if
# this looks sane, matching every other stage this session.
#
# Run: python -u experiments/stage6_spatial_mesh/run_gossip_comparator.py --mode <arm>
#   where <arm> in {fusion, gossip_uniform, gossip_weighted, fedavg_global}

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
ROOT = Path("runs/stage6/offset_world/gossip_comparator")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODES = ["fusion", "gossip_uniform", "gossip_weighted", "fedavg_global"]


def run_one(mode: str, seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)  # deterministic, same field every arm/seed
    # matches every other 5-seed script this session: seed 42 keeps the
    # original wander paths (200+i) so the single-seed result is reproduced
    # exactly; other seeds get their own wander paths (seed+200+i).
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    tag = f"gossip_cmp_{mode}" if seed == SEED else f"gossip_cmp_{mode}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Gossip comparator ({mode}, seed={seed})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    r = res.get("certainty_mse_pearson_r")
    comm_total = res["comm_bytes_total"]
    comm_mean = res["comm_bytes_mean_per_step"]
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} "
          f"r={r:.4f}")
    print(f"    comm_bytes_total={comm_total:,} ({comm_total/1e6:.2f} MB over the run) "
          f"mean/step={comm_mean:,.0f} bytes")
    return whole_run_mse, res['mean_mse_last_50'], r


def main(mode: str, seeds: list[int], root: str | None = None):
    global ROOT
    if root:
        ROOT = Path(root)
    results = []
    for seed in seeds:
        results.append(run_one(mode, seed))
    if len(seeds) > 1:
        arr = np.array(results)  # (n_seeds, 3): whole_run, last50, r
        print(f"\n=== {mode} {len(seeds)}-SEED SUMMARY (this invocation, seeds={seeds}) ===")
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
                        help="comma-separated seed list, e.g. '1042,2042,3042,4042'")
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    args = parser.parse_args()
    main(args.mode, [int(s) for s in args.seeds.split(",")], args.root)
