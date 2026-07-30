# Scaled-environment test: 10x10 nodes (100, up from the standard 6x6/36) on
# a proportionally bigger grid, testing whether more spatial extent widens
# fusion's advantage over fedavg_global -- the A.1-vs-A.2 mechanism (global
# floor = Var(delta) over the WHOLE area; per-node floor = mean WITHIN-FOV
# variance only) predicts it should, if extent grows without also making
# each node's own local patch more internally chaotic.
#
# Design choices (Christian's, this is a single exploratory run, not a
# replicated result):
#   - grid_size=31, fov=7, stride=3 -> exactly 100 nodes via the existing
#     node_grid() helper (same rule as every other stage6 config).
#   - The field's dynamism is kept at the SAME SCALE, not intensified --
#     build_dynamic_offset_field now scales its control-point density and
#     wake size proportionally to grid_size (verified bit-for-bit identical
#     to the original field at grid_size=22), so this is "more of the same
#     environment", not "a rougher one". Verified: peak offset ~104 deg,
#     comfortably under the 180 deg wrap ceiling; whole-run spatial variance
#     549 vs the original 397 (up ~38%, the intended effect); no new
#     tail-flattening.
#   - Wearable count scales with area to hold DENSITY constant (3 on the
#     original 22x22 grid -> round(3*(31/22)**2) = 6 here), not held fixed at
#     3 -- this is what avoids confounding the floor-widening test with "the
#     same coverage effort spread over more area", which would independently
#     hurt fusion via the already-known propagation-latency/hop-distance
#     mechanism (unrelated to A.1/A.2). Node density is unchanged by
#     construction (same fov=7/stride=3 rule at both grid sizes), so this is
#     the closest one-run approximation to a controlled isolation of the
#     floor effect available without a full seed-repeat study.
#
# Run: python -u experiments/stage6_spatial_mesh/run_scaled_10x10_test.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path,
                                                         dynamic_analytic_floors)
from stage6_spatial_mesh.run_6abc_overlap_ablation import node_grid

from beliefmesh.config import load_config

CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_scaled_10x10")
G = 31
T = 390
FOV = 7
STRIDE = 3
# same wearable DENSITY as the original 3-wearable/22x22 design, not the same
# count: 3 * (31/22)^2 = 5.96 -> 6. Keeping density fixed means coverage
# difficulty doesn't confound the floor-widening test with "fewer wearables
# per unit area than before".
N_WEARABLES = round(3 * (G / 22) ** 2)


def main(arms: list[str] = ("fusion", "fedavg_global")):
    cfg = load_config()
    ROOT.mkdir(parents=True, exist_ok=True)

    centres = node_grid(FOV, STRIDE, G)
    print(f"nodes: {len(centres)} (expect 100)")
    uniform = np.full((T, G, G), 0.5)

    field = build_dynamic_offset_field(G, T)
    np.save(ROOT / "scaled_field.npy", field)
    spatial_var = field.reshape(T, -1).var(axis=1)
    print(f"field check: whole-run var={spatial_var.mean():.1f}, "
          f"last-50 var={spatial_var[-50:].mean():.1f}, "
          f"peak abs offset={np.abs(field).max():.1f} deg (ceiling 180)")

    gf, nf = dynamic_analytic_floors(field, centres, FOV, G, window=50)
    print(f"analytic floors (last-50 instantaneous mean): global={gf:.4f} per_node={nf:.4f}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=FOV, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Scaled 10x10 dynamic offset world ({tag})",
            offset_field=field,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        results[tag] = (res["mean_mse_last_50"], whole_run_mse)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}")

    if "fusion" in arms:
        run("scaled_fusion", "fusion")
    if "fedavg_global" in arms:
        run("scaled_fedavg_global", "fedavg_global")

    print("\n=== SCALED 10x10 RESULT ===")
    for tag, (last50, whole) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}")
    print("(reference, 36-node/22-grid Stage 4 5-seed mean: "
          "fusion 0.0308+-0.0068, global 0.0232+-0.0018, ~33% mean gap)")
    print("(fusion arm from this batch, already saved: last50=0.0226 whole_run=0.0347)")
    print("=== DONE ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global"],
                        default=["fusion", "fedavg_global"])
    args = parser.parse_args()
    main(args.arms)
