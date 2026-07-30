# Final test in the chaotic-offset-world line of investigation: does scaling
# to 10x10 nodes (100, up from 36) change the fusion-vs-fedavg_global
# picture? At 36 nodes, on this same chaotic field (bigger disagreement
# between regions, floor ratio 8.89x, at the corrected lr=3e-4),
# fedavg_global still won (0.0771 vs fusion's 0.1001) -- narrower than the
# original gentle-field gap, but unchanged in direction. The earlier smooth-
# field 10x10 test (density-matched wearables) was inconclusive/close;
# Christian's recollection is that scaling helped fusion there, worth
# re-checking on a field that's actually hard enough for it to matter.
#
# --n-wearables is a CLI arg so the 6-wearable (density-matched, 3*(31/22)^2)
# and 3-wearable (same count as the 36-node runs, i.e. sparser per unit area
# at this scale) variants can run as two separate, parallel invocations.
#
# G=31, FOV=7, STRIDE=3 -> 100 nodes (node_grid, same rule as every other
# stage6 config). lr=3e-4 (corrected), n_train_repeats=1. Analytic floors at
# this scale (checked before running): global=0.0518 per_node=0.0072
# ratio=7.17x -- close to the 36-node design (8.89x), not badly diluted.
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_10x10_test.py --n-wearables 6
#      python -u experiments/stage6_spatial_mesh/run_chaotic_10x10_test.py --n-wearables 3

from __future__ import annotations

import argparse
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
from stage6_spatial_mesh.run_6abc_overlap_ablation import node_grid

from beliefmesh.config import load_config

CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
G = 31
T = 390
FOV = 7
STRIDE = 3
LR = 3e-4


def main(n_wearables: int, arms: list[str]):
    cfg = load_config()
    cfg.model.lr = LR
    root = Path(f"runs/stage6/offset_world/dynamic_chaotic_10x10/w{n_wearables}")
    root.mkdir(parents=True, exist_ok=True)

    centres = node_grid(FOV, STRIDE, G)
    print(f"nodes: {len(centres)} (expect 100), n_wearables={n_wearables}, lr={LR}")
    uniform = np.full((T, G, G), 0.5)

    field = build_chaotic_offset_field(G, T)
    np.save(root / "chaotic_field_10x10.npy", field)
    gf, nf = dynamic_analytic_floors(field, centres, FOV, G, window=50)
    print(f"analytic floors: global={gf:.4f} per_node={nf:.4f} ratio={gf/nf:.2f}x "
          f"peak={np.abs(field).max():.1f}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(n_wearables)]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=root / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=FOV, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Chaotic 10x10, {n_wearables} wearables ({tag})",
            offset_field=field,
        )
        cell_mse_steps = np.load(root / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        results[tag] = (res["mean_mse_last_50"], whole_run_mse)
        print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}")

    if "fusion" in arms:
        run("chaotic10x10_fusion", "fusion")
    if "fedavg_global" in arms:
        run("chaotic10x10_fedavg_global", "fedavg_global")

    print(f"\n=== CHAOTIC 10x10 RESULT (n_wearables={n_wearables}) ===")
    for tag, (last50, whole) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}")
    print("(reference, 36-node chaotic @ lr=3e-4: fusion whole_run=0.1001, "
          "fedavg_global whole_run=0.0771)")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wearables", type=int, required=True)
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global"],
                        default=["fusion", "fedavg_global"])
    args = parser.parse_args()
    main(args.n_wearables, args.arms)
