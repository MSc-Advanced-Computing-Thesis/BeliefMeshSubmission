# THE current Stage 4 result -- see THESIS_EXPERIMENTS_SUMMARY.md Stage 4,
# especially SS4.4/4.7. Three independent random-wandering wearables
# (no uncertainty-driven routing -- the simplest policy tried, and it beat
# every engineered one), fusion vs fedavg_global, on the field-generation-bug
# FIXED dynamic offset world (build_dynamic_offset_field's keyframes are
# normalised to consistent spatial spread so no segment of the run can
# randomly go quiet -- see run_offset_experiments.py).
#
# Result: fusion whole-run MSE 0.0252 vs global 0.0228 (~10.5% gap) -- down
# from ~3x on the first (routing-confounded) attempt at this world. The
# archive/ subfolder holds the superseded intermediate scripts that document
# how this number was reached (routing confound, cooldown sweep, multi-
# wearable clumping, the field-flattening bug) if that narrative needs
# revisiting.
#
# Run: python -u experiments/stage6_spatial_mesh/run_stage4_dynamic_world_final.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/dynamic_offset")
N_WEARABLES = 3


def main():
    cfg = load_config()
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)  # keyframe-normalised, sustained-dynamism field
    np.save(ROOT / "dynamic_field_v2.npy", field)

    spatial_var = field.reshape(T, -1).var(axis=1)
    print(f"field check: whole-run var={spatial_var.mean():.1f}, "
          f"last-50 var={spatial_var[-50:].mean():.1f}, "
          f"ratio={spatial_var.mean()/spatial_var[-50:].mean():.2f}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Stage 4 dynamic world, {N_WEARABLES} random-wandering wearables ({tag})",
            offset_field=field,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        results[tag] = (res["mean_mse_last_50"], whole_run_mse)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}")

    run("dynamic_fusion_randomwander3_v2field", "fusion")
    run("dynamic_global_randomwander3_v2field", "fedavg_global")

    print("\n=== STAGE 4 FINAL RESULT ===")
    for tag, (last50, whole) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
