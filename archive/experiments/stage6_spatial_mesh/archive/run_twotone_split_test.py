# Cleanest possible spatial test: a hard two-tone vertical split (+70 deg
# left half, -70 deg right half of the grid), temporally static. Validated
# analytically first (dynamic_analytic_floors/analytic_floors take no GPU
# time): ratio=3.67x, lower than the "winning" RBF-based field's 9.54x --
# because with stride=3 < FOV=7, neighbouring nodes always overlap, so 12/36
# nodes structurally straddle ANY split boundary and see huge internal
# variance regardless of where the line is drawn. This is the best a clean
# two-region design can do with the current node layout, not a tuning
# failure -- flagged honestly before running.
#
# Standard 36-node/3-wearable layout, lr=3e-4, lam=0.1 (default), fusion vs
# fedavg_global, single seed=42.
#
# Run: python -u experiments/stage6_spatial_mesh/run_twotone_split_test.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_random_wander_path, analytic_floors

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_twotone_split")
N_WEARABLES = 3
LR = 3e-4
A = 70.0


def main(arms: list[str] = ("fusion", "fedavg_global")):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")

    snapshot = np.zeros((G, G))
    snapshot[:, :G // 2] = A
    snapshot[:, G // 2:] = -A
    field = np.tile(snapshot[None], (T, 1, 1))
    np.save(ROOT / "field.npy", field)

    gf, nf = analytic_floors(snapshot, centres, 7, G)
    print(f"analytic floors: global={gf:.4f} per_node={nf:.4f} ratio={gf/nf:.2f}x "
          f"tone_diff={2*A:.0f} deg lr={LR}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Two-tone vertical split ({tag})",
            offset_field=field,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}{r_str}")

    if "fusion" in arms:
        run("twotone_fusion", "fusion")
    if "fedavg_global" in arms:
        run("twotone_fedavg_global", "fedavg_global")

    print("\n=== TWO-TONE SPLIT RESULT ===")
    for tag, (last50, whole, r) in results.items():
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}{r_str}")
    print("=== DONE ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global"],
                        default=["fusion", "fedavg_global"])
    args = parser.parse_args()
    main(args.arms)
