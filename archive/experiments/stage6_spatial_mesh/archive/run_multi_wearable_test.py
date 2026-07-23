# Follow-up to run_cooldown_sweep.py. Routing/cooldown tuning on a SINGLE
# wearable hit a wall: whole-run MSE stuck ~0.084-0.093 across cooldowns
# 15/30/60/100, still ~3.2x worse than global fedavg (0.0266). Christian's
# question: does adding wearables help fusion more than global? Mechanism
# argument: extra wearables let fusion cut its MAX HOP DISTANCE by covering
# multiple regions directly (hop-0) at once -- a structural fix, not a
# routing tweak. Global already broadcasts at hop-0 everywhere every step
# regardless of anchor count, so it has little to gain from more anchors
# beyond slightly better-averaged training signal.
#
# Deployment framing (Christian's, verbatim): wearable count doesn't affect
# feasibility of the core claim -- it's fine, even useful, to show the system
# can leverage MORE rescue resources if available, not just survive on one.
#
# Three wearables, epistemic_staleness routing with cooldown=60 (best
# whole-run performer from the single-wearable sweep), seeded at separated
# start positions so they don't immediately clump on the one strong signal
# (the wake). fusion vs fedavg_global, same field, same node layout.
#
# Run: python -u experiments/stage6_spatial_mesh/run_multi_wearable_test.py

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
COOLDOWN = 60
N_WEARABLES = 3
# spread starts: top-left, top-right, bottom-centre -- distinct regions so
# the three don't all beeline for the same wake signal from step 0
STARTS = [(2.0, 2.0), (2.0, 19.0), (19.0, 10.0)]


def main():
    cfg = load_config()
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    # wearable_paths only used to seed start positions under a policy --
    # a length-1 array per wearable is enough
    paths = [np.array([list(s)]) for s in STARTS]

    results = {}

    def run(tag, mode):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"Dynamic offset world, {N_WEARABLES} wearables ({tag})",
            offset_field=field, wearable_policy="epistemic_staleness",
            policy_cooldown=COOLDOWN,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        paths_arr = np.load(ROOT / tag / "realised_wearable_paths.npy")
        for wi in range(N_WEARABLES):
            p = paths_arr[wi]
            print(f"    wearable {wi}: bbox rows[{p[:,0].min():.1f},{p[:,0].max():.1f}] "
                  f"cols[{p[:,1].min():.1f},{p[:,1].max():.1f}]")
        results[tag] = (res["mean_mse_last_50"], whole_run_mse)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f}")

    run("dynamic_fusion_3wearable", "fusion")
    run("dynamic_global_3wearable", "fedavg_global")

    print("\n=== MULTI-WEARABLE TEST SUMMARY ===")
    for tag, (last50, whole) in results.items():
        print(f"{tag}: last50={last50:.4f} whole_run={whole:.4f}")
    print("(reference, 1 wearable: fusion best-case whole_run=0.0842 [cooldown60], "
          "global whole_run=0.0266)")
    print("=== MULTI-WEARABLE TEST DONE ===")


if __name__ == "__main__":
    main()
