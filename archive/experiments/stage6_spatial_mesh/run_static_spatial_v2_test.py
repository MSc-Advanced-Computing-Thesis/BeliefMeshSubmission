# V2: the same decisive static-spatial diagnostic, rerun with everything
# validated during today's deep-dive into WHY the original run (fusion
# 0.0464 vs fedavg 0.0517, ambiguous) never showed a clean result:
#   - CoordConv local-FOV coordinate channels (mesh.py) -- fixes the
#     confirmed architectural inability to represent two different true
#     values within one node's own FOV (multi-region diagnostic)
#   - excluded_rotation_ranges -- removes a confirmed, specific visual
#     ambiguity in the base digit at ~120-150 deg and near +-180 deg
#     (excursion-frame analysis), ~20% error reduction, ~3x fewer severe
#     errors, on its own
#   - lr=3e-5 (down from 3e-4) -- confirmed to cut steady-state noise
#     3-6x on well-evidenced cells in the 3-node multi-region test; the one
#     real cost (slower correction on weakly-evidenced/underrepresented
#     cells) is judged less relevant here since wearables move (unlike that
#     static diagnostic), which should let different regions take turns
#     being freshly anchored
#
# Same field, same node/wearable layout, same lam=0.1 as the original run,
# so this is a clean, direct before/after comparison.
#
# Run: python -u experiments/stage6_spatial_mesh/run_static_spatial_v2_test.py

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
                                                         analytic_floors)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_static_spatial_v2")
N_WEARABLES = 3
LR = 3e-5
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]


def main(arms: list[str] = ("fusion", "fedavg_global")):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")

    dynamic_field = build_chaotic_offset_field(G, T, n_keyframes=4)
    snapshot = dynamic_field[T // 2]  # a single representative frame
    field = np.tile(snapshot[None], (T, 1, 1))  # genuinely time-invariant
    np.save(ROOT / "field.npy", field)

    gf, nf = analytic_floors(field[0], centres, 7, G)
    print(f"analytic floors (static, exact): global={gf:.4f} per_node={nf:.4f} "
          f"ratio={gf/nf:.2f}x peak={np.abs(field).max():.1f} lr={LR}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Static spatial-only v2 ({tag})",
            offset_field=field,
            excluded_rotation_ranges=EXCLUDED_RANGES,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res.get("certainty_mse_pearson_r")
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}{r_str}")

    if "fusion" in arms:
        run("static_spatial_fusion", "fusion")
    if "fedavg_global" in arms:
        run("static_spatial_fedavg_global", "fedavg_global")

    print("\n=== STATIC-SPATIAL RESULT (temporally frozen, purely spatial disagreement) ===")
    for tag, (last50, whole, r) in results.items():
        r_str = f" r={r:.4f}" if r is not None else ""
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}{r_str}")
    print("(reference, ORIGINAL v1 single-run @ lr=3e-4, no CoordConv, no angle exclusion: "
          "fusion whole_run=0.0464 r=+0.1577, fedavg_global whole_run=0.0517 r=-0.1021)")
    print("=== DONE ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global"],
                        default=["fusion", "fedavg_global"])
    args = parser.parse_args()
    main(args.arms)
