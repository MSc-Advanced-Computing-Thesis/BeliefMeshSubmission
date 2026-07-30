# Diagnostic: test a genuinely high lam (1.0, vs the 0.01-0.4 range already
# swept) to check Christian's hypothesis directly -- that most nodes report
# certainty compressed near 1.0 (the interim report itself flagged this:
# "certainty scale is compressed to 0.92-0.99"), which range-restricts the
# achievable certainty-MSE correlation regardless of how good the underlying
# relationship is. Higher lam should push confidence down where errors occur,
# widening the certainty SPREAD -- reports that spread directly (std/range of
# avg_cert_map), not just r, so the mechanism can be checked, not just
# inferred.
#
# consensus mode, default rho=0.2 (kept fixed -- this is a lam-only test),
# single seed=42.
#
# Run: python -u experiments/stage6_spatial_mesh/run_lam_extreme.py

from __future__ import annotations

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


def main():
    cfg = load_config()
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = f"consensus_lam{LAM}"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="consensus", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"lam extreme ({tag})",
        offset_field=field, lam=LAM,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    cert_map = np.load(ROOT / tag / "avg_cert_map.npy")
    valid = cert_map[~np.isnan(cert_map)]

    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")
    print(f"certainty spread: mean={valid.mean():.4f} std={valid.std():.4f} "
          f"min={valid.min():.4f} max={valid.max():.4f} range={valid.max()-valid.min():.4f}")
    print("(reference, lam=0.1 default consensus rho=0.2, seed 42: "
          "whole_run=0.0256 r=-0.2308 -- certainty spread not previously measured, "
          "will need that run's avg_cert_map for a clean before/after)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
