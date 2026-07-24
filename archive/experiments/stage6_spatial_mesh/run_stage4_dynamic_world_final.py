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
# EXP_SEED env var: overrides the training seed (random/np/torch), the Mesh
# sample seed (env_seed), and the wearable-walk seeds together, so a seed
# repeat resamples all the actual stochastic elements of one deployment (the
# model's training trajectory AND where the wearables happen to wander) --
# not just the model. The FIELD itself stays fixed regardless of EXP_SEED:
# it's the validated task being tested, not something to resample per repeat.
# At the default seed (cfg.seed), wearable-walk seeds are EXACTLY 200/201/202
# as before -- fully backward compatible with the already-reported number.
#
# Run: python -u experiments/stage6_spatial_mesh/run_stage4_dynamic_world_final.py
# Run: EXP_SEED=1042 python -u experiments/stage6_spatial_mesh/run_stage4_dynamic_world_final.py

from __future__ import annotations

import os
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
ROOT = Path("runs/stage6/offset_world/dynamic")
N_WEARABLES = 3


def main():
    cfg = load_config()
    seed = int(os.environ.get("EXP_SEED", cfg.seed))
    root = ROOT
    if seed != cfg.seed:
        root = root.parent / f"{root.name}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)  # fixed task, not resampled per seed
    np.save(root / "dynamic_field_v2.npy", field)

    spatial_var = field.reshape(T, -1).var(axis=1)
    print(f"field check: whole-run var={spatial_var.mean():.1f}, "
          f"last-50 var={spatial_var[-50:].mean():.1f}, "
          f"ratio={spatial_var.mean()/spatial_var[-50:].mean():.2f}")

    # 200/201/202 exactly, at the default seed (backward compatible); a
    # distinct, deterministic triple per repeat seed otherwise
    wearable_seed_base = 200 if seed == cfg.seed else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
             for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=root / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Stage 4 dynamic world, {N_WEARABLES} random-wandering wearables ({tag})",
            offset_field=field,
            env_seed=seed,
        )
        cell_mse_steps = np.load(root / tag / "cell_mse_steps.npy")
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
