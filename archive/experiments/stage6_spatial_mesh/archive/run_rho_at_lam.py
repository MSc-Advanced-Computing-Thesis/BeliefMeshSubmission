# Follow-up to run_rho_extremes.py + run_lam_extreme.py: rho's extremes test
# came back nearly flat (0.0001 vs 0.99 barely separated) at the DEFAULT lam,
# where certainty is compressed to a ~0.047-wide band -- too little variance
# for an EMA's response rate to matter regardless of rho. lam=1.0 widened
# that spread substantially (range 0.047->0.069, r -0.23->-0.44). Hypothesis:
# rho only becomes a meaningful lever once lam has unlocked real
# differentiation for it to track. This reruns the rho extremes AT lam=1.0
# instead of the default, one value per invocation (run twice in parallel
# with different --rho).
#
# Run: python -u experiments/stage6_spatial_mesh/run_rho_at_lam.py --rho 0.0001
#      python -u experiments/stage6_spatial_mesh/run_rho_at_lam.py --rho 0.99

from __future__ import annotations

import argparse
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
ROOT = Path("runs/stage6/offset_world/dynamic_lam_rho_sweep")
N_WEARABLES = 3
LAM = 1.0


def main(rho: float):
    cfg = load_config()
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = f"consensus_lam{LAM}_rho{rho}"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="consensus", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"rho-at-lam=1.0 ({tag})",
        offset_field=field, lam=LAM, rho=rho,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    cert_map = np.load(ROOT / tag / "avg_cert_map.npy")
    valid = cert_map[~np.isnan(cert_map)]

    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")
    print(f"certainty spread: mean={valid.mean():.4f} std={valid.std():.4f} "
          f"range={valid.max()-valid.min():.4f}")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, required=True)
    args = parser.parse_args()
    main(args.rho)
