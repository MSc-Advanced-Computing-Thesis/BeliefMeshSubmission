# Follow-up to run_lam_extreme.py (lam=1.0: widened certainty spread, r
# -0.23->-0.44, no accuracy cost) and run_rho_at_lam.py (confirmed lam gates
# rho -- separation between rho extremes went from 0.016 to 0.10 once lam
# unlocked real differentiation). This pushes lam further (2, 5, 10) to find
# where the benefit stops -- watching specifically for the mean certainty
# crashing toward its floor (rather than genuinely spreading), which would
# signal the regulariser has started dominating the loss and the model is
# just minimising it by reporting low confidence everywhere, not learning
# real differentiation.
#
# consensus mode, rho held at default (0.2) -- lam-only test, rho comes later
# per Christian's plan. Single seed=42, sequential (not parallel) so it can
# run unattended.
#
# Run: python -u experiments/stage6_spatial_mesh/run_lam_higher_sweep.py

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
LAM_VALUES = [2.0, 5.0, 10.0]


def main():
    cfg = load_config()
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    results = {}

    def run(lam):
        tag = f"consensus_lam{lam}"
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode="consensus", baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"higher lam sweep ({tag})",
            offset_field=field, lam=lam,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        cert_map = np.load(ROOT / tag / "avg_cert_map.npy")
        valid = cert_map[~np.isnan(cert_map)]
        r = res["certainty_mse_pearson_r"]
        results[lam] = (res["mean_mse_last_50"], whole_run_mse, r,
                        valid.mean(), valid.std(), valid.max() - valid.min())
        print(f"\n### lam={lam}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f} r={r:.4f}")
        print(f"    certainty: mean={valid.mean():.4f} std={valid.std():.4f} "
              f"range={valid.max()-valid.min():.4f}")

    for lam in LAM_VALUES:
        run(lam)

    print("\n=== HIGHER LAM SWEEP SUMMARY ===")
    print("(reference: lam=0.1 whole_run=0.0256 r=-0.2308 cert_mean=0.9928 cert_std=0.0071 range=0.0474)")
    print("(reference: lam=1.0 whole_run=0.0252 r=-0.4363 cert_mean=0.9862 cert_std=0.0112 range=0.0693)")
    for lam in LAM_VALUES:
        last50, whole, r, cmean, cstd, crange = results[lam]
        print(f"lam={lam}: last50={last50:.4f} whole_run={whole:.4f} r={r:.4f} "
              f"cert_mean={cmean:.4f} cert_std={cstd:.4f} cert_range={crange:.4f}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
