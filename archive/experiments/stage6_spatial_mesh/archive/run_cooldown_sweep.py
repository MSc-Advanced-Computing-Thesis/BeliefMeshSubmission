# Follow-up to run_routing_fix_test.py. cooldown=15 improved accuracy a lot
# (0.0347 -> 0.0203) but the wearable never actually left the BL quadrant --
# 15 steps * 0.4 step size ~= 6 grid units is small enough that BL alone
# always has an eligible (non-cooled-down) cell to chase, so the policy never
# gets forced outward. Sweeping to much larger cooldowns to see whether that
# forces real cross-quadrant travel, and whether it can combine staleness's
# flat hop-gradient with epistemic's targeting accuracy.
#
# Also: the standard last-50-step evaluation window turns out to land in a
# low-diversity trough of the field (spatial variance ~165 vs whole-run mean
# ~575, peak ~1575 around step 100-150) -- so mean_mse_last_50 alone
# structurally favours whichever arm does best on a near-static field. This
# script also reports whole-trajectory mean MSE for a fairer comparison.
#
# Run: python -u experiments/stage6_spatial_mesh/run_cooldown_sweep.py

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
COOLDOWNS = [30, 60, 100]


def main():
    cfg = load_config()
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    path = [np.load(ENV / "wearable_path.npy")]
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)

    results = {}
    for cd in COOLDOWNS:
        tag = f"dynamic_fusion_epistaleness_cooldown{cd}"
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=path, node_centres=centres,
            fov_size=7, mode="fusion", baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Dynamic offset world, cooldown={cd} ({tag})",
            offset_field=field, wearable_policy="epistemic_staleness",
            policy_cooldown=cd,
        )
        p = np.load(ROOT / tag / "realised_wearable_path.npy")
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        quads = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
        for r, c in p:
            key = ("T" if r < G / 2 else "B") + ("L" if c < G / 2 else "R")
            quads[key] += 1
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, quads,
                        (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()))
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f} | quadrants {quads} | "
              f"bbox rows[{p[:,0].min():.1f},{p[:,0].max():.1f}] "
              f"cols[{p[:,1].min():.1f},{p[:,1].max():.1f}]")

    print("\n=== COOLDOWN SWEEP SUMMARY ===")
    for tag, (last50, whole, quads, bbox) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f} quadrants={quads}")
    print("=== COOLDOWN SWEEP DONE ===")


if __name__ == "__main__":
    main()
