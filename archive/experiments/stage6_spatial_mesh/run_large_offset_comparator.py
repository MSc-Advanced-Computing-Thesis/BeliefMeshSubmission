# Large-offset-magnitude comparator (2026-08): does scaling up
# build_dynamic_offset_field's background_max_deg (the smooth baseline's own
# field, not an engineered discontinuity) change the naive-vs-fusion null
# established at the default 60 deg? Same offset world, seed 42, otherwise
# identical to run_aggregation_comparator.py -- only background_max_deg
# differs.
#
# Run: python -u experiments/stage6_spatial_mesh/run_large_offset_comparator.py --mode nig_product --background-max-deg 120 --seeds 42

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
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
MODES = ["naive", "certainty", "nig_product"]
DEFAULT_MAX_DEG = 60.0


def run_one(mode: str, seed: int, background_max_deg: float):
    cfg = load_config()
    cfg.model.lr = LR
    root = Path(f"runs/stage6/offset_world/large_offset_comparator_deg{background_max_deg:g}")
    root.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T, background_max_deg=background_max_deg)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    tag = f"agg_cmp_{mode}" if seed == SEED else f"agg_cmp_{mode}_seed{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=f"large_offset_deg{background_max_deg:g}_{tag}", run_dir=root / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Large-offset comparator ({mode}, seed={seed}, max_deg={background_max_deg})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        track_disagreement=(mode == "nig_product"),
        apply_colour_filter=False,
    )
    cell_mse_steps = np.load(root / tag / "cell_mse_steps.npy")
    cell_cert_steps = np.load(root / tag / "cell_cert_steps.npy")
    T = cell_mse_steps.shape[0]
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    new_mean, new_std, n_t, _ = per_timestep_then_averaged(cell_mse_steps, cell_cert_steps, T - 50, T)
    print(f"\n### large_offset(deg={background_max_deg})/{tag}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r(new)={new_mean:+.4f}+/-{new_std:.4f} (n_t={n_t})")
    dstats = res.get("disagreement_stats")
    if dstats:
        print(f"    disagreement_stats: {dstats}")
    return whole_run_mse, res['mean_mse_last_50'], new_mean, new_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seeds", type=str, default=str(SEED))
    parser.add_argument("--background-max-deg", type=float, default=DEFAULT_MAX_DEG)
    args = parser.parse_args()
    for seed in [int(s) for s in args.seeds.split(",")]:
        run_one(args.mode, seed, args.background_max_deg)
    print("=== DONE ===")
