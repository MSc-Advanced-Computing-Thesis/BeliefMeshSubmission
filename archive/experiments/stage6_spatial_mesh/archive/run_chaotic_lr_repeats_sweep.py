# Fast diagnostic: the chaotic-field pilot bumped lr (3e-4->1e-3) AND
# n_train_repeats (1->4) simultaneously, and hop0 got WORSE, not better
# (0.148->0.281 over 100 steps, clearly climbing) -- worse than the original
# n_train_repeats=1/lr=1e-3 attempt (flat/noisy ~0.10-0.17, not improving but
# not diverging either). That rules out "more repeats fixes it" and raises
# the question of whether the LR bump itself is the destabiliser (4
# consecutive Adam steps on the same batch can build up momentum and
# overshoot, especially chasing a fast-moving target). Neither prior attempt
# isolated lr from n_train_repeats -- this sweeps both on a SHORT 80-step
# slice (cheap, ~2-3 min/run) to see which combination actually makes hop0
# trend down before spending a full 390-step run on anything again.
#
# fusion mode only. Same chaotic_field.npy as before (sliced to 80 steps).
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_lr_repeats_sweep.py

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
ROOT = Path("runs/stage6/offset_world/dynamic_chaotic/lr_repeats_sweep")
N_WEARABLES = 3
SLICE = 80

COMBOS = [
    (3e-4, 1),  # original default, unmodified -- isolates whether the FIELD alone breaks it
    (3e-4, 3),
    (1e-3, 1),  # the first full attempt's setting -- flat/noisy, not diverging
    (1e-3, 3),  # the failed pilot's setting (was 4, using 3 here for a mid-point)
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
