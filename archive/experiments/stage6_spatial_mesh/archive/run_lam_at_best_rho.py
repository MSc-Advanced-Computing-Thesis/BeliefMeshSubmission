# Follow-up to run_lam_rho_sweep.py. That sweep found: (a) lam showed a
# jagged, non-monotonic pattern across fusion runs -- likely just seed noise
# at n=1-per-point, no real signal visible; (b) rho showed a clean MONOTONIC
# trend on consensus -- lower rho (faster-adapting trust EMA) gave
# consistently stronger certainty-MSE correlation (0.05: r=-0.57, 0.1: -0.34,
# 0.2 [default]: -0.23), with accuracy roughly flat across all three.
#
# This script: (1) extends the rho trend down to 0.01, to see if it keeps
# improving or bottoms out; (2) determines the best rho found across the
# WHOLE rho sweep (reading the already-saved manifests from
# dynamic_lam_rho_sweep/, plus the new 0.01 point) by strongest |r|; (3)
# reruns the 5 lam values, this time in CONSENSUS mode at that best rho
# (rather than fusion mode at default rho=0.2) -- if lam's effect was being
# masked by consensus's own noisier-at-default-rho behaviour, pairing it with
# the more informative rho setting might reveal a real trend.
#
# Run: python -u experiments/stage6_spatial_mesh/run_lam_at_best_rho.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_lam_rho_sweep")
N_WEARABLES = 3
LAM_VALUES = [0.01, 0.05, 0.1, 0.2, 0.4]
PRIOR_RHO_VALUES = [0.05, 0.1, 0.2, 0.4, 0.6]  # already run by run_lam_rho_sweep.py


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

    def run(tag, mode, **kw):
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=ROOT / tag,
            all_grids=uniform, wearable_paths=paths, node_centres=centres,
            fov_size=7, mode=mode, baseline_checkpoint=CKPT,
            n_wearable_samples=1, n_train_repeats=1,
            title=f"lam-at-best-rho follow-up ({tag})",
            offset_field=field,
            **kw,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res["certainty_mse_pearson_r"]
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f} r={r:.4f}")

    # --- extend the rho trend down to 0.01 ---
    run("consensus_rho0.01", "consensus", rho=0.01)

    # --- pick the best rho across the whole sweep (strongest |r|) ---
    rho_r = {0.01: results["consensus_rho0.01"][2]}
    for rho in PRIOR_RHO_VALUES:
        tag = f"consensus_rho{rho}"
        manifest_path = ROOT / tag / "manifest.yaml"
        if manifest_path.exists():
            m = yaml.safe_load(manifest_path.read_text())
            rho_r[rho] = m["results"]["certainty_mse_pearson_r"]
    best_rho = min(rho_r, key=lambda k: rho_r[k])  # most negative r = strongest correlation
    print(f"\nrho results so far: {rho_r}")
    print(f"best rho by |r|: {best_rho} (r={rho_r[best_rho]:.4f})\n")

    # --- rerun the lam sweep, this time in consensus mode at best_rho ---
    for lam in LAM_VALUES:
        run(f"consensus_lam{lam}_rho{best_rho}", "consensus", lam=lam, rho=best_rho)

    print("\n=== LAM-AT-BEST-RHO SWEEP SUMMARY ===")
    print(f"best_rho used: {best_rho}")
    for lam in LAM_VALUES:
        last50, whole, r = results[f"consensus_lam{lam}_rho{best_rho}"]
        print(f"  lam={lam}: last50={last50:.4f} whole_run={whole:.4f} r={r:.4f}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
