# Decisive diagnostic: purely spatially dynamic, TEMPORALLY STATIC field --
# the theoretically ideal case for fusion. With zero drift over time, there
# is no staleness at all: a belief trained/propagated at step 5 is exactly
# as valid at step 300 as at step 6. fedavg_global's one real advantage
# (never going stale because it's always freshly retrained) becomes
# irrelevant here -- both mechanisms are working with a fixed ground truth,
# so this isolates whether per-node specialisation (fusion's actual selling
# point) beats parameter-averaging when staleness literally cannot be a
# factor. If fusion doesn't win clearly here, the problem isn't environment
# tuning -- it's the mechanism itself.
#
# Field: same spatial design as the winning 36-node config (wide control
# spacing matching the original's layout, sharp RBF transitions, 100 deg
# amplitude -- floor ratio ~9x), but a SINGLE frame (middle of the dynamic
# version, to avoid any edge-of-keyframe artefact) tiled across all 390
# steps -- genuinely time-invariant, not just slow-changing.
#
# lr=3e-4 (the corrected value), lam=0.1 (default -- lam=10 was shown not to
# help calibration), standard 36-node/3-wearable layout, fusion vs
# fedavg_global, single seed=42 first pass.
#
# Run: python -u experiments/stage6_spatial_mesh/run_static_spatial_test.py

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
ROOT = Path("runs/stage6/offset_world/dynamic_static_spatial")
N_WEARABLES = 3
LR = 3e-4


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
            title=f"Static spatial-only ({tag})",
            offset_field=field,
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
    print("(reference, new-spatial/old-temporal @ lr=3e-4: fusion 5-seed mean=0.0570, "
          "fedavg_global 5-seed mean=0.0651)")
    print("=== DONE ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global"],
                        default=["fusion", "fedavg_global"])
    args = parser.parse_args()
    main(args.arms)
