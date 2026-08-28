# Stage 6a: Overlap threshold ablation under TRUE product-of-experts fusion.
# Experiment Specification Sec 7 & Sec 9 step 4.
#
# Three spatial configurations on the STATIC environment (first V2 frame tiled
# across all 390 timesteps), single wearable (V2 generation -- the spec-exact
# configuration: 36 nodes, 390 steps, one path; Sec 8 + Sec 10):
#   7x7 stride 3 -- 36 nodes (6x6), max cell coverage 9  (prior: learned well)
#   5x5 stride 3 -- 36 nodes (6x6), max cell coverage 4  (prior: failed)
#   5x5 stride 2 -- 81 nodes (9x9), max cell coverage 9  (prior: failed)
# The prior threshold ("nine-way overlap sufficient, four-way not") was
# measured under scalar averaging and is provisional (Spec Sec 7). This run
# regenerates it under corrected fusion: it may move, sharpen, or dissolve.
#
# Run from the project root:
#   python experiments/stage6_spatial_mesh/run_6abc_overlap_ablation.py
#   python experiments/stage6_spatial_mesh/run_6abc_overlap_ablation.py --configs 7x7_s3

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment

from beliefmesh.config import load_config

ENV_DIR = Path("experiments/stage6_spatial_mesh/environment_v2")
BASELINE_CHECKPOINT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")


def node_grid(fov: int, stride: int, grid_size: int) -> np.ndarray:
    """Node centres for the 5x5 configs, using the prior repo's layout rule:
    centres run to the far edge (slight FOV clipping there) so the ENTIRE grid
    is covered. Full coverage matters: the single wearable must never be
    outside all FOVs, or the mesh idles those steps and the ablation confounds
    shallow overlap with coverage holes."""
    half = fov // 2
    positions = list(range(half, grid_size, stride))
    return np.array([[cx, cy] for cy in positions for cx in positions])


# Chapter 5 (2026-08-27) sweep geometry. Sec 5.4.1's axis is MEAN PER-CELL
# COVERAGE, not stride and not node-graph degree (the 6.37 figure belongs to
# Sec 5.7's connectivity argument and is a different property).
#
# PLACEMENT IRREGULARITY, stated rather than implied: strides 5, 4 and 2 use
# node_grid() placement; stride 3 uses the INHERITED environment_v2 centres --
# 36 nodes at mean coverage 3.64, not node_grid's 49 at 4.37 -- so that row
# reads directly against Sec 5.3.1 and every other comparator. If the stride-3
# point falls off the trend set by the other three, the placement difference is
# the first thing to check before it is read as a result. Report against mean
# per-cell coverage, not against stride.
#
# NODE COUNT IS NOT SEPARABLE FROM DENSITY here: at fixed fov, raising density
# means adding nodes. This design does not control for it.
CH5_GEOMS = {
    "7x7_s5": dict(fov=7, stride=5, placement="node_grid"),
    "7x7_s4": dict(fov=7, stride=4, placement="node_grid"),
    "7x7_s3": dict(fov=7, stride=3, placement="inherited_environment_v2"),
    "7x7_s2": dict(fov=7, stride=2, placement="node_grid"),
}


def geometry_stats(centres, fov, grid_size):
    """Mean per-cell coverage (the swept axis), node-graph degree, node count."""
    from beliefmesh.node.mesh import fov_cells
    cov = np.zeros((grid_size, grid_size), int)
    sets = []
    for cx, cy in centres:
        cells = fov_cells(int(cx), int(cy), fov, grid_size)
        sets.append(set(cells))
        for r, c in cells:
            cov[r, c] += 1
    nz = cov[cov > 0]
    deg = [sum(1 for j, b in enumerate(sets) if j != i and (a & b))
           for i, a in enumerate(sets)]
    return dict(n_nodes=len(centres), mean_cell_coverage=float(nz.mean()),
                max_cell_coverage=int(nz.max()), mean_node_degree=float(np.mean(deg)),
                uncovered_cells=int((cov == 0).sum()))


def main_chapter5(geom_names, seeds, root):
    """Sec 5.4.1 regeneration: matched to run_aggregation_comparator in every
    respect except geometry -- offset world, nig_product, lr 3e-5, lam 5.0,
    3 wearables, repeats 1, rotation exclusions applied and recorded."""
    import random
    import torch
    from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                            build_random_wander_path)
    EXCLUDED = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
    grids = np.load(ENV_DIR / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    for name in geom_names:
        g = CH5_GEOMS[name]
        centres = (np.load(ENV_DIR / "node_centres.npy")
                   if g["placement"] == "inherited_environment_v2"
                   else node_grid(g["fov"], g["stride"], G))
        stats = geometry_stats(centres, g["fov"], G)
        for seed in seeds:
            cfg = load_config()
            cfg.model.lr = 3e-5
            paths = [build_random_wander_path(T, G, seed=(200 if seed == 42 else seed + 200) + i)
                     for i in range(3)]
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            run_mesh_experiment(
                cfg, condition=f"5_4_1_{name}_seed{seed}",
                run_dir=Path(root) / f"{name}_seed{seed}",
                all_grids=np.full((T, G, G), 0.5), wearable_paths=paths,
                node_centres=centres, fov_size=g["fov"], mode="nig_product",
                baseline_checkpoint=BASELINE_CHECKPOINT,
                n_wearable_samples=1, n_train_repeats=1,
                title=f"5.4.1 overlap density: {name} (mean cov "
                      f"{stats['mean_cell_coverage']:.2f}), seed={seed}",
                offset_field=build_dynamic_offset_field(G, T), env_seed=seed,
                excluded_rotation_ranges=EXCLUDED, lam=5.0,
                wearable_policy=None, policy_step_size=0.4,
                extra_manifest={"geometry": name, "fov": g["fov"],
                                "stride": g["stride"], "placement_rule": g["placement"],
                                **stats,
                                "density_note": "node count is NOT separable from "
                                                "density at fixed fov; this design "
                                                "does not control for it"},
            )
            print(f"### 5.4.1 {name} seed{seed}: mean_cov="
                  f"{stats['mean_cell_coverage']:.2f} nodes={stats['n_nodes']} "
                  f"deg={stats['mean_node_degree']:.2f}", flush=True)


def main(config_names: list[str]):
    cfg = load_config()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    grids_dynamic = np.load(ENV_DIR / "environment_grids.npy")
    grid_size = grids_dynamic.shape[1]
    total_steps = grids_dynamic.shape[0]
    # static: first frame tiled (Spec: the ablation is run on the static condition)
    all_grids = np.tile(grids_dynamic[0:1], (total_steps, 1, 1))
    wearable_paths = [np.load(ENV_DIR / "wearable_path.npy")]  # single wearable, Sec 10

    configs = {
        "7x7_s3": dict(fov_size=7, node_centres=np.load(ENV_DIR / "node_centres.npy"),
                       title="Stage 6a: static, 7x7 stride 3 (36 nodes, max coverage 9)"),
        "5x5_s3": dict(fov_size=5, node_centres=node_grid(5, 3, grid_size),
                       title="Stage 6a: static, 5x5 stride 3 (49 nodes, max coverage 4)"),
        "5x5_s2": dict(fov_size=5, node_centres=node_grid(5, 2, grid_size),
                       title="Stage 6a: static, 5x5 stride 2 (100 nodes, max coverage 9)"),
    }

    for name in config_names:
        c = configs[name]
        run_mesh_experiment(
            cfg, condition=f"6a_{name}",
            run_dir=Path("runs/stage6/colour_world/static/overlap_density_ablation") / name,
            all_grids=all_grids, wearable_paths=wearable_paths,
            node_centres=c["node_centres"], fov_size=c["fov_size"],
            mode="fusion", baseline_checkpoint=BASELINE_CHECKPOINT,
            n_wearable_samples=1, n_train_repeats=10, title=c["title"],
            extra_manifest={"environment": "static (V5 frame 0 tiled)",
                            "ablation": "overlap threshold, Spec Sec 9 step 4"},
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+",
                        choices=["7x7_s3", "5x5_s3", "5x5_s2"],
                        default=["7x7_s3", "5x5_s3", "5x5_s2"])
    parser.add_argument("--profile", choices=["legacy", "chapter5"], default="legacy",
                        help="legacy reproduces the original single-seed colour-world "
                             "ablation exactly; chapter5 runs the Sec 5.4.1 sweep")
    parser.add_argument("--geoms", nargs="+", default=list(CH5_GEOMS))
    parser.add_argument("--seeds", type=str, default="42,1042,2042,3042,4042")
    parser.add_argument("--root", type=str, default="runs/chapter5_v2/s5_4_1_overlap")
    a = parser.parse_args()
    if a.profile == "chapter5":
        main_chapter5(a.geoms, [int(x) for x in a.seeds.split(",")], a.root)
    else:
        main(a.configs)
