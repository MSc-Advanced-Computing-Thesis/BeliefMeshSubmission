# Follow-up to run_rho_at_lam.py (rho extremes at lam=1.0: separation grew
# from 0.016 at default lam to 0.10 at lam=1.0, confirming lam gates rho).
# lam=10.0 gives by far the widest certainty spread of the whole sweep
# (std=0.0425, range=0.2122, vs 0.0112/0.0693 at lam=1.0) but the binned
# calibration ratio didn't improve over the default -- the extra spread
# isn't (yet) carrying extra signal on its own. Christian's hypothesis:
# rho tuning is what turns that raw spread into real accuracy/calibration
# gain -- so if rho matters anywhere, it should show up most clearly here,
# at the widest spread available. Testing the same valid extremes as before
# (0.0001, 0.99) at LAM=10.0 instead of 1.0.
#
# consensus mode, tempering left at its default (True) to match the existing
# consensus_lam10.0 baseline (rho=0.2 default) it's compared against -- only
# rho varies here. Single seed=42, run twice in parallel with different --rho.
#
# Run: python -u experiments/stage6_spatial_mesh/run_rho_at_lam10.py --rho 0.0001
#      python -u experiments/stage6_spatial_mesh/run_rho_at_lam10.py --rho 0.99

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
LAM = 10.0


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
        title=f"rho-at-lam=10.0 ({tag})",
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
    print("(reference, consensus_lam10.0 default rho=0.2: whole_run=0.0292 r=-0.3687 "
          "cert_std=0.0425 range=0.2122)")
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, required=True)
    args = parser.parse_args()
    main(args.rho)
