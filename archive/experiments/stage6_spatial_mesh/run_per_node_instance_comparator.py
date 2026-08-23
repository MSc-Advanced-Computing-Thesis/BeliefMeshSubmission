# Per-node-instance evidence-asymmetry diagnostic (2026-08): each node
# trains on its OWN fixed digit-seven instance rather than the shared/
# per-cell instance every prior construction used, and when supplying a
# belief about a shared cell to a receiving neighbour, predicts on the
# RECEIVING neighbour's own instance for that cell -- simulating differing
# physical viewpoints while preserving a common referent per recipient.
# See Mesh(per_node_instance_prediction=True) and MeshNode.predict_belief_for
# in src/beliefmesh/node/mesh.py for the implementation.
#
# Smooth offset world, seed 42, otherwise identical to the established
# baseline (run_aggregation_comparator.py) -- deliberately compared against
# the baseline, not an engineered construction.
#
# Run: python -u experiments/stage6_spatial_mesh/run_per_node_instance_comparator.py --mode nig_product --seeds 42

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.cert_mse_metrics import per_timestep_then_averaged
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/per_node_instance_comparator")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODES = ["naive", "certainty", "nig_product"]
NODE_INSTANCE_SEED = 42


def run_one(mode: str, seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    tag = f"agg_cmp_{mode}" if seed == SEED else f"agg_cmp_{mode}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=f"per_node_instance_{tag}", run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Per-node-instance comparator ({mode}, seed={seed})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        per_node_instance_prediction=True, node_instance_seed=NODE_INSTANCE_SEED,
        track_disagreement=(mode == "nig_product"),
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    cell_cert_steps = np.load(ROOT / tag / "cell_cert_steps.npy")
    T = cell_mse_steps.shape[0]
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    new_mean, new_std, n_t, _ = per_timestep_then_averaged(cell_mse_steps, cell_cert_steps, T - 50, T)
    print(f"\n### per_node_instance/{tag}: last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} "
          f"r(new)={new_mean:+.4f}+/-{new_std:.4f} (n_t={n_t})")
    dstats = res.get("disagreement_stats")
    if dstats:
        print(f"    disagreement_stats: {dstats}")
    fusion_calls = res.get("fusion_call_count")
    if fusion_calls:
        print(f"    fusion_call_count={fusion_calls} "
              f"(one predict_belief_for() forward pass per contributor-recipient-cell triple "
              f"feeds into each of these -- see compute-accounting note in mesh.py)")
    return whole_run_mse, res['mean_mse_last_50'], new_mean, new_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seeds", type=str, default=str(SEED))
    args = parser.parse_args()
    for seed in [int(s) for s in args.seeds.split(",")]:
        run_one(args.mode, seed)
    print("=== DONE ===")
