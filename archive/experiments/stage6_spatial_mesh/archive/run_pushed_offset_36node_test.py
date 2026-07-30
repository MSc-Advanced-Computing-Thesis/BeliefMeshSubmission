# Standard 36-node/22-grid Stage 4 setup, but with the offset field pushed
# as far as it can safely go without risking the circular wraparound ceiling
# (background_max_deg 60->70, wake_amplitude 50->55): span 152.3 -> 171.1 deg
# (ceiling 180, ~9 deg safety buffer -- 75/60 already crosses 180, too risky),
# whole-run spatial variance 396.6 -> 527.7 (+33%), no new tail-flattening.
#
# Tests whether increasing Var(delta) alone (same node/wearable setup as the
# validated 5-seed Stage 4 result, only the field's magnitude changes) widens
# fusion's advantage over fedavg_global -- the direct A.1 prediction, without
# the scale/density confounds of the 10x10-node test (which came back
# inconclusive: within existing seed-to-seed noise, not a clean widen/no-widen
# signal).
#
# Run: python -u experiments/stage6_spatial_mesh/run_pushed_offset_36node_test.py

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
ROOT = Path("runs/stage6/offset_world/dynamic_pushed_offset")
N_WEARABLES = 3
BACKGROUND_MAX_DEG = 70.0
WAKE_AMPLITUDE = 55.0


def main():
    cfg = load_config()
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")

    field = build_dynamic_offset_field(G, T, background_max_deg=BACKGROUND_MAX_DEG,
                                       wake_amplitude=WAKE_AMPLITUDE)
    np.save(ROOT / "pushed_field.npy", field)
    spatial_var = field.reshape(T, -1).var(axis=1)
    print(f"field check: whole-run var={spatial_var.mean():.1f}, "
          f"last-50 var={spatial_var[-50:].mean():.1f}, "
          f"span={field.max()-field.min():.1f} deg (ceiling 180)")
    print("(reference, original field: whole-run var=396.6, span=152.3)")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Pushed-offset dynamic world ({tag})",
            offset_field=field,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        results[tag] = (res["mean_mse_last_50"], whole_run_mse)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}")

    run("pushed_fusion", "fusion")
    run("pushed_fedavg_global", "fedavg_global")

    print("\n=== PUSHED-OFFSET 36-NODE RESULT ===")
    for tag, (last50, whole) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}")
    print("(reference, original-field 36-node 5-seed mean: "
          "fusion 0.0308+-0.0068, global 0.0232+-0.0018, ~33% mean gap)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
