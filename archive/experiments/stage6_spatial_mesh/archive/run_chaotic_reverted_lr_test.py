# Direct answer to the only question that actually matters: does fusion beat
# fedavg_global on the chaotic field, full 390 steps -- not whether hop0's
# error trend rises or falls over time. The lr/repeats diagnostics
# (run_chaotic_lr_repeats_sweep.py / _confirm.py) never found a config where
# hop0 cleanly converges, but that was orthogonal to the actual comparison:
# the ONE full run we have (chaotic_fusion/chaotic_fedavg_global) used the
# bumped lr=1e-3 for BOTH arms, so fedavg_global may have been hurt by that
# same instability too, never tested at the original lr. This reverts lr to
# 3e-4 (original default, n_train_repeats=1, i.e. literally just undoing the
# "jack up the learning rate" change) and runs fusion + fedavg_global head to
# head, full 390 steps, same chaotic_field.npy, for a clean answer.
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_reverted_lr_test.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_random_wander_path

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_chaotic")
N_WEARABLES = 3
LR = 3e-4  # reverted to original default


def main(arms: list[str] = ("fusion", "fedavg_global")):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.load(ROOT / "chaotic_field.npy")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Chaotic offset world, reverted lr={LR} ({tag})",
            offset_field=field,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        results[tag] = (res["mean_mse_last_50"], whole_run_mse)
        print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}")

    if "fusion" in arms:
        run("chaotic_fusion_lr3e4", "fusion")
    if "fedavg_global" in arms:
        run("chaotic_fedavg_global_lr3e4", "fedavg_global")

    print("\n=== REVERTED-LR CHAOTIC RESULT ===")
    for tag, (last50, whole) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}")
    print("(reference, lr=1e-3 attempt: chaotic_fusion whole_run=0.1388, "
          "chaotic_fedavg_global whole_run=0.0801)")
    print("=== DONE ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["fusion", "fedavg_global"],
                        default=["fusion", "fedavg_global"])
    args = parser.parse_args()
    main(args.arms)
