# Follow-up to run_dynamic_offset_test.py. Christian noticed (from
# experiment_animation.mp4) that the fusion arm's wearable, under
# epistemic_staleness routing, never left the bottom-left quadrant across all
# 390 steps -- 100% BL occupancy -- while the global arm's wearable, under the
# IDENTICAL policy, swept all four quadrants (29/45/18/8% TL/TR/BL/BR).
# Bounding boxes confirmed it quantitatively: fusion rows 12.7-20.2/cols
# 0.6-7.8 vs global rows 7.4-20.1/cols 0.6-14.0.
#
# Mechanism: fusion's per-cell epistemic uncertainty is genuinely
# differentiated (the wake keeps refreshing it), so the epistemic term keeps
# outscoring staleness for cells near the wake -- the wearable camps instead
# of touring. Global's shared-weight uncertainty is comparatively flat, so its
# score defaults to staleness-driven wide coverage. This confounds the
# hop-3-latency conclusion from run_dynamic_offset_test.py: fusion's far
# corners are undervisited AS WELL AS relay-delayed.
#
# PRE-REGISTERED PREDICTION: a hard anti-camping cooldown (runner.py
# policy_cooldown -- excludes any cell visited within the last N steps from
# the argmax target, regardless of epistemic score) should break the camping
# and let coverage approach global's spread. Three fusion arms:
#   1. staleness only            -- coverage baseline, no epistemic signal at all
#   2. epistemic_staleness       -- the original (camping) policy, rerun here for
#                                    an apples-to-apples comparison under this script
#   3. epistemic_staleness + cooldown(15) -- the proposed fix
#
# Run: python -u experiments/stage6_spatial_mesh/run_routing_fix_test.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.evaluation import git_commit
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         dynamic_analytic_floors)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/dynamic_offset")
COOLDOWN = 15


def main():
    cfg = load_config()
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    path = [np.load(ENV / "wearable_path.npy")]
    centres = np.load(ENV / "node_centres.npy")

    field = build_dynamic_offset_field(G, T)
    gf, nf = dynamic_analytic_floors(field, centres, 7, G, window=50)
    print(f"DYNAMIC FLOORS (reference, same field as run_dynamic_offset_test.py): "
          f"global {gf:.4f} | per-node {nf:.4f}")

    results = {}

    def run(tag, mode, wearable_policy, policy_cooldown=0):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        results[tag] = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=path, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Dynamic offset world, routing fix ({tag})",
            offset_field=field, wearable_policy=wearable_policy,
            policy_cooldown=policy_cooldown,
        )
        p = np.load(ROOT / tag / "realised_wearable_path.npy")
        quads = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
        for r, c in p:
            key = ("T" if r < G / 2 else "B") + ("L" if c < G / 2 else "R")
            quads[key] += 1
        print(f"### {tag}: {results[tag]['mean_mse_last_50']:.4f} "
              f"(r {results[tag]['certainty_mse_pearson_r']:.3f}) | quadrants {quads}")

    run("dynamic_fusion_staleness", "fusion", "staleness")
    run("dynamic_fusion_epistaleness_v2", "fusion", "epistemic_staleness")
    run("dynamic_fusion_epistaleness_cooldown", "fusion", "epistemic_staleness",
        policy_cooldown=COOLDOWN)

    print("\n=== ROUTING FIX TEST SUMMARY ===")
    print(f"floors -- global {gf:.4f} | per-node {nf:.4f}")
    for tag, res in results.items():
        print(f"{tag}: {res['mean_mse_last_50']:.4f} (r {res['certainty_mse_pearson_r']:.3f})")
    print("(reference: original dynamic_fusion_epistaleness = 0.0283, "
          "dynamic_global_epistaleness = 0.0091)")
    print("=== ROUTING FIX TEST DONE ===")


if __name__ == "__main__":
    main()
