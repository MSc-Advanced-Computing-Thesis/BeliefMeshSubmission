# New-spatial/old-temporal combo: the chaotic field's spatial redesign
# (wide control spacing matching the original's 4-corner layout, sharper
# RBF transitions, bigger amplitude -- floor ratio 4.97x->8.89x) was
# validated as a genuine improvement, but the temporal redesign (n_keyframes
# 4->30, drift 0.06->2.4 deg/step) turned out too fast for the per-node
# model to track at all (hop0 never converged, certainty-MSE correlation
# collapsed to ~0 even after fixing an unrelated LR bug). This keeps the
# improved spatial design but reverts n_keyframes to the ORIGINAL 4 (drift
# back down to ~0.11 deg/step, close to the original's 0.06) -- checked
# analytically first: global=0.0335 per_node=0.0035 ratio=9.67x, the best
# ratio of any variant tried, with a drift rate that should actually be
# trackable given lr=3e-4 (also reverted -- the fast-temporal test's LR bump
# was diagnosed as actively harmful, not helpful).
#
# Standard 36-node/22x22 layout, 3 wearables (unchanged from every other
# stage6 config), fusion vs fedavg_global.
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_spatial_only_test.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_chaotic_offset_field,
                                                         build_random_wander_path,
                                                         dynamic_analytic_floors)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_chaotic_spatial_only")
N_WEARABLES = 3
LR = 3e-4
N_KEYFRAMES = 4  # reverted to original temporal pace


def main(arms: list[str] = ("fusion", "fedavg_global")):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_chaotic_offset_field(G, T, n_keyframes=N_KEYFRAMES)
    np.save(ROOT / "field.npy", field)

    gf, nf = dynamic_analytic_floors(field, centres, 7, G, window=50)
    cell = (11, 11)
    drift = np.abs(np.diff(field[:, cell[0], cell[1]])).mean()
    print(f"analytic floors: global={gf:.4f} per_node={nf:.4f} ratio={gf/nf:.2f}x "
          f"peak={np.abs(field).max():.1f} drift/step={drift:.3f} lr={LR}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"New-spatial/old-temporal ({tag})",
            offset_field=field,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}{r_str}")

    if "fusion" in arms:
        run("spatial_only_fusion", "fusion")
    if "fedavg_global" in arms:
        run("spatial_only_fedavg_global", "fedavg_global")

    print("\n=== NEW-SPATIAL/OLD-TEMPORAL RESULT ===")
    for tag, (last50, whole, r) in results.items():
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}{r_str}")
    print("(reference, new-spatial/new-temporal @ lr=3e-4: fusion whole_run=0.1001, "
          "fedavg_global whole_run=0.0771)")
    print("(reference, original field: fusion/global both ~0.02-0.03 range)")
    print("=== DONE ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global"],
                        default=["fusion", "fedavg_global"])
    args = parser.parse_args()
    main(args.arms)
