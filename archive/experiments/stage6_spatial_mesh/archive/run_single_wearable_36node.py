# Single-wearable variant of the standard 36-node/22x22 dynamic offset world
# (same task every lam/rho sweep run today used, just N_WEARABLES=1 instead
# of 3). Sparser coverage means longer gaps between visits to any given
# region, which should sharpen exactly the staleness/coverage-quality effect
# already established earlier in the project (wearable-coverage correlated
# with the fusion-vs-global gap, r~=-0.77) -- and is the more natural regime
# for testing whether tuned consensus (trusting recent, well-covered nodes
# over stale ones) earns its keep, vs. the well-covered 3-wearable case where
# most cells stay fairly fresh regardless.
#
# Three arms: fusion (plain unweighted product-of-experts), fedavg_global
# (parameter-averaging baseline), consensus with today's 5-seed-validated
# tuned config (lam=10.0, rho=0.99). Single seed=42 -- exploratory, not a
# replicated result.
#
# Run: python -u experiments/stage6_spatial_mesh/run_single_wearable_36node.py

from __future__ import annotations

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
ROOT = Path("runs/stage6/offset_world/dynamic_single_wearable")
N_WEARABLES = 1
LAM = 10.0
RHO = 0.99


def main(arms: list[str] = ("fusion", "fedavg_global", "consensus")):
    cfg = load_config()
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode, **kwargs):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Single wearable, 36-node ({tag})",
            offset_field=field, **kwargs,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}{r_str}")

    if "fusion" in arms:
        run("single_fusion", "fusion")
    if "fedavg_global" in arms:
        run("single_fedavg_global", "fedavg_global")
    if "consensus" in arms:
        run("single_consensus_lam10_rho099", "consensus", lam=LAM, rho=RHO)

    print("\n=== SINGLE WEARABLE 36-NODE RESULT ===")
    for tag, (last50, whole, r) in results.items():
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}{r_str}")
    print("=== DONE ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global", "consensus"],
                        default=["fusion", "fedavg_global", "consensus"])
    args = parser.parse_args()
    main(args.arms)
