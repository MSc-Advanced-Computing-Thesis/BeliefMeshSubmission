# Confirmation follow-up to run_chaotic_lr_repeats_sweep.py's 80-step result:
# lr=3e-4 (original, unmodified) + n_train_repeats=3 was the only one of four
# combos where hop0 actually trended down (0.067->0.051); bumping lr to 1e-3
# made every repeats setting worse or flat, ruling out "jack up the learning
# rate" as the fix -- it was the opposite lever (keep lr conservative, add
# more gradient steps per visit). This re-checks that result over a longer
# 150-step slice, and also tries repeats=5 to see if convergence keeps
# improving or plateaus/reverses.
#
# fusion mode only. Same chaotic_field.npy as before (sliced to 150 steps).
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_lr_repeats_confirm.py

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
ROOT = Path("runs/stage6/offset_world/dynamic_chaotic/lr_repeats_confirm")
N_WEARABLES = 3
SLICE = 150

COMBOS = [
    (3e-4, 3),
    (3e-4, 5),
]


def main():
    cfg = load_config()
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T_full, G = grids.shape[0], grids.shape[1]
    uniform = np.full((SLICE, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field_full = np.load(Path("runs/stage6/offset_world/dynamic_chaotic/chaotic_field.npy"))
    field = field_full[:SLICE]

    paths_full = [build_random_wander_path(T_full, G, seed=200 + i) for i in range(N_WEARABLES)]
    paths = [p[:SLICE] for p in paths_full]

    for lr, repeats in COMBOS:
        cfg.model.lr = lr
        tag = f"lr{lr}_rep{repeats}"
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode="fusion", baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=repeats,
            title=f"lr={lr} repeats={repeats}",
            offset_field=field,
        )
        hop_hist = np.load(ROOT / tag / "hop_mse_history.npy", allow_pickle=True).item()
        hop0 = [v for v in hop_hist[0] if v is not None]
        early = np.mean(hop0[:len(hop0)//3]) if hop0 else float("nan")
        late = np.mean(hop0[-len(hop0)//3:]) if hop0 else float("nan")
        trend = "IMPROVING" if late < early * 0.9 else ("WORSENING" if late > early * 1.1 else "FLAT")
        print(f"\n### lr={lr} repeats={repeats}: hop0 early-third={early:.4f} "
              f"late-third={late:.4f} -> {trend}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
