# Follow-up to run_chaotic_spatial_only_test.py's win (fusion 0.0531 vs
# fedavg_global 0.0674, first clean fusion win of the whole chaotic-field
# investigation, genuinely converged this time) -- but fusion's own
# certainty-MSE correlation was still weak there (r=0.0061, std=0.0046).
# Earlier today, raising lam (0.1->10) reliably widened raw certainty spread
# on the SMOOTH field, but the more rigorous binned calibration-ratio check
# showed that extra spread didn't carry extra information (flat ~2.2-2.8x
# ratio across the whole lam range). This checks whether that finding
# transfers to this new, now-converging, harder-spatial field -- or whether
# a genuinely-converged point-estimator lets lam's spread actually mean
# something here, unlike on the earlier broken fast-temporal attempt.
#
# fusion mode only, lam=10.0, same field.npy as the winning run (exact same
# task, only lam differs).
#
# Run: python -u experiments/stage6_spatial_mesh/run_chaotic_spatial_only_lam10.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_random_wander_path

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_chaotic_spatial_only")
N_WEARABLES = 3
LR = 3e-4
LAM = 10.0


def main():
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.load(ROOT / "field.npy")  # exact same field as the winning run

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = "spatial_only_fusion_lam10"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="fusion", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"New-spatial/old-temporal, lam={LAM} ({tag})",
        offset_field=field, lam=LAM,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    cert_map = np.load(ROOT / tag / "avg_cert_map.npy")
    valid = cert_map[~np.isnan(cert_map)]
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")
    print(f"certainty: mean={valid.mean():.4f} std={valid.std():.4f} "
          f"range={valid.max()-valid.min():.4f}")
    print("(reference, lam=0.1: whole_run=0.0531 r=0.0061 cert_mean=0.9920 "
          "cert_std=0.0046 range=0.0243)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
