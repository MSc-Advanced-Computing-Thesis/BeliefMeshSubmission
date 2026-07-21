# Priority batch: prove belief exchange > global FedAvg (P1) and > gossip (P2).
# PRE-REGISTERED 2026-07-21, before any runs.
#
# World: the SHADOW field — ±60° gradient + hard-edged +45° block present from
# step 0 (Christian's debris-shadow scenario, spatial not temporal). Uniform
# colour. Repeats=1 throughout (the better-calibrated regime). Floors for this
# field: global ≈ 0.0413 (+base 0.0536), per-node ≈ 0.0043 (+base 0.0166).
#
# Design principle: routing helps every anchored system (the wearable is shared
# infrastructure), so each system routes USING ITS OWN epistemic map. The
# replicated calibration asymmetry (fusion r −0.5/−0.7 vs global r ≈ 0) should
# then convert into an accuracy asymmetry: same policy, different instruments.
#
# PREDICTIONS:
#  - staleness-only routing improves BOTH systems vs random (coverage effect,
#    system-agnostic; kills camping by construction).
#  - epistemic+staleness improves fusion FURTHER (informative map) but adds
#    little or nothing to global (uninformative map). The differential is P1's
#    causal evidence.
#  - global remains ≥ its floor 0.054 everywhere; inside the shadow its
#    regional error is structurally high (single bias cannot fit the block);
#    routed fusion should beat it inside the shadow decisively.
#  - gossip@1 with random wearable lands near fusion@1-random (parity), and
#    cannot use epistemic routing as effectively IF its map is weaker —
#    measured, not assumed (gossip r was −0.16 at repeats=10; @1 unknown).
#
# Run: python -u experiments/stage6_spatial_mesh/run_priority_batch.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_offset_field, analytic_floors

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/priority_batch")


def main():
    cfg = load_config()
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    path = [np.load(ENV / "wearable_path.npy")]
    centres = np.load(ENV / "node_centres.npy")
    shadow = build_offset_field(G, T, event=True, event_step=0)  # block from step 0

    gf, ff = analytic_floors(shadow[0], centres, 7, G)
    print(f"SHADOW FLOORS: global {gf:.4f} (+base {gf+0.0123:.4f}) | "
          f"per-node {ff:.4f} (+base {ff+0.0123:.4f})")

    results = {}

    def run(tag, **kw):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        defaults = dict(all_grids=uniform, wearable_paths=path, node_centres=centres,
                        fov_size=7, mode="fusion", baseline_checkpoint=CKPT,
                        n_wearable_samples=1, n_train_repeats=1,
                        offset_field=shadow, title=tag)
        defaults.update(kw)
        results[tag] = run_mesh_experiment(cfg, condition=tag, run_dir=ROOT / tag,
                                           **defaults)
        print(f"### {tag}: {results[tag]['mean_mse_last_50']:.4f} "
              f"(r {results[tag]['certainty_mse_pearson_r']:.3f})")

    # baselines: random wearable
    run("shadow_fusion_random")
    run("shadow_global_random", mode="fedavg_global")
    run("shadow_gossip_random", mode="fedavg")           # matched rate: repeats=1
    # staleness-only routing (system-agnostic coverage)
    run("shadow_fusion_staleness", wearable_policy="staleness")
    run("shadow_global_staleness", mode="fedavg_global", wearable_policy="staleness")
    # each system routed by ITS OWN epistemic map + staleness
    run("shadow_fusion_epistaleness", wearable_policy="epistemic_staleness")
    run("shadow_global_epistaleness", mode="fedavg_global",
        wearable_policy="epistemic_staleness")
    # colour attribution cleanup (different world; rate-matching seed repeat)
    run("v2_fusion_repeats1_seed1042", all_grids=grids, offset_field=None,
        env_seed=1042)

    print("\n=== PRIORITY BATCH SUMMARY ===")
    print(f"floors: global {gf+0.0123:.4f} | per-node {ff+0.0123:.4f}")
    for tag, res in results.items():
        print(f"{tag}: {res['mean_mse_last_50']:.4f} (r {res['certainty_mse_pearson_r']:.3f})")
    print("=== PRIORITY BATCH DONE ===")


if __name__ == "__main__":
    main()
