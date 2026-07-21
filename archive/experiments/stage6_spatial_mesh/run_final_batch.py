# Final evidence batch before consolidation. One sequential run of everything
# still owed to the experiments chapter:
#   1-2. repeats sweep (3, 5) on the offset world -- fills in the
#        adaptation-rate vs dynamism matching curve between the measured
#        endpoints (repeats=10: 0.0978, repeats=1: 0.0837)
#   3-4. no-event controls (repeats 1 and 10) -- unconfounds event-recovery
#   5.   ROUTING: uncertainty-greedy wearable, offset world, repeats=1 --
#        the end-to-end test of retention-by-supply (Christian's mechanism)
#   6.   colour v2 fusion at repeats=1 -- does rate-matching help where the
#        belief arms previously tied? (Christian's "improve colour provably")
#   7-12. env-seed repeats (1042) of headline cells for statistical hygiene
#
# Run from the project root: python -u experiments/stage6_spatial_mesh/run_final_batch.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_offset_field

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
ENV3 = Path("experiments/stage6_spatial_mesh/environment_v3")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")


def seeded(cfg, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main():
    cfg = load_config()
    grids_v2 = np.load(ENV / "environment_grids.npy")
    grids_v3 = np.load(ENV3 / "environment_grids.npy")
    T, G = grids_v2.shape[0], grids_v2.shape[1]
    uniform = np.full((T, G, G), 0.5)
    path_v2 = [np.load(ENV / "wearable_path.npy")]
    path_v3 = [np.load(ENV / "wearable_path.npy")[:grids_v3.shape[0]]]
    centres = np.load(ENV / "node_centres.npy")
    field = build_offset_field(G, T)
    field_noevent = build_offset_field(G, T, event=False)
    results = {}

    def run(tag, **kw):
        seeded(cfg, kw.pop("torch_seed", cfg.seed))
        defaults = dict(all_grids=uniform, wearable_paths=path_v2,
                        node_centres=centres, fov_size=7, mode="fusion",
                        baseline_checkpoint=CKPT, n_wearable_samples=1,
                        n_train_repeats=1, title=tag)
        defaults.update(kw)
        results[tag] = run_mesh_experiment(
            cfg, condition=tag, run_dir=Path("runs/stage6/final_batch") / tag,
            **defaults)
        print(f"### {tag}: {results[tag]['mean_mse_last_50']:.4f}")

    # 1-2: repeats sweep on offset world
    run("offset_fusion_repeats3", offset_field=field, n_train_repeats=3)
    run("offset_fusion_repeats5", offset_field=field, n_train_repeats=5)
    # 3-4: no-event controls
    run("offset_fusion_repeats1_noevent", offset_field=field_noevent, n_train_repeats=1)
    run("offset_fusion_repeats10_noevent", offset_field=field_noevent, n_train_repeats=10)
    # 5: uncertainty-directed wearable (retention by supply)
    run("offset_fusion_repeats1_routing", offset_field=field, n_train_repeats=1,
        wearable_policy="epistemic")
    # 6: colour v2, rate-matched fusion
    run("v2_fusion_repeats1", all_grids=grids_v2, n_train_repeats=1)
    # 7-12: env-seed repeats of headline cells
    run("seed1042_v2_fusion", all_grids=grids_v2, n_train_repeats=10, env_seed=1042, torch_seed=1042)
    run("seed1042_v2_naive", all_grids=grids_v2, mode="naive", n_train_repeats=10, env_seed=1042, torch_seed=1042)
    run("seed1042_v3_fusion", all_grids=grids_v3, wearable_paths=path_v3, n_train_repeats=10, env_seed=1042, torch_seed=1042)
    run("seed1042_v3_naive", all_grids=grids_v3, wearable_paths=path_v3, mode="naive", n_train_repeats=10, env_seed=1042, torch_seed=1042)
    run("seed1042_offset_fusion_r1", offset_field=field, n_train_repeats=1, env_seed=1042, torch_seed=1042)
    run("seed1042_offset_fedavg_global", offset_field=field, mode="fedavg_global",
        n_train_repeats=10, env_seed=1042, torch_seed=1042)

    print("\n=== FINAL BATCH SUMMARY ===")
    for tag, res in results.items():
        print(f"{tag}: {res['mean_mse_last_50']:.4f} (r {res['certainty_mse_pearson_r']:.3f})")
    print("=== FINAL BATCH DONE ===")


if __name__ == "__main__":
    main()
