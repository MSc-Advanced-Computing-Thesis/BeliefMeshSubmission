# Follow-up to run_multi_wearable_test.py. The 3-wearable online routing
# collapsed to a single clump (Christian: "they all end up in the same
# location") despite sequential per-step greedy assignment intended to spread
# them -- the shared epistemic signal pulls every wearable toward the same
# strongest local maximum (the wake) once cooldown windows overlap.
#
# Every dynamic-offset-world run so far has used ONLINE epistemic/staleness
# routing -- the ORIGINAL static random-walk path (environment_v2/
# wearable_path.npy, used throughout the original colour-world stage 6 work)
# has never actually been tried on THIS field. It's a single smoothed random
# walk that already covers ~3/4 quadrants and the full grid extent (bbox
# rows 0.1-21.9, cols 0.1-21.9) with no camping by construction -- worth
# testing plain, before any further routing engineering.
#
# Run: python -u experiments/stage6_spatial_mesh/run_static_replay_test.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/dynamic_offset")


def main():
    cfg = load_config()
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    path = [np.load(ENV / "wearable_path.npy")]
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=path, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Dynamic offset world, static random-walk replay ({tag})",
            offset_field=field,  # wearable_policy=None -> replays the static path
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        results[tag] = (res["mean_mse_last_50"], whole_run_mse)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}")

    run("dynamic_fusion_staticreplay", "fusion")
    run("dynamic_global_staticreplay", "fedavg_global")

    print("\n=== STATIC REPLAY TEST SUMMARY ===")
    for tag, (last50, whole) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}")
    print("(reference: 1-wearable epistaleness+cooldown60 fusion whole_run=0.0842, "
          "3-wearable epistaleness+cooldown60 fusion whole_run=0.0400, "
          "global (any routing so far) whole_run~0.025-0.027)")
    print("=== STATIC REPLAY TEST DONE ===")


if __name__ == "__main__":
    main()
