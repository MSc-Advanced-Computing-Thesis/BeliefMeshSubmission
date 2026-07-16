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
            run_dir=Path("runs/stage6/6a_overlap_ablation") / name,
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
    main(parser.parse_args().configs)
