# Diagnostic: test the true valid extremes of rho (an EMA weight, only
# meaningful in [0,1]) to check whether rho has a real, detectable effect at
# all, given the 0.01-0.6 sweep came back dominated by run-to-run noise
# (same-config reruns swung r from -0.52 to -0.15).
#
# Predictions (falsifiable, stated before running):
#   rho->0 (0.0001): trust barely moves from its initial 1.0 for the whole
#     run -> weighted fusion in consensus mode should become indistinguishable
#     from UNWEIGHTED fusion (both MSE and r should resemble plain `fusion`).
#   rho->1 (0.99): trust discards almost all history, snaps to the single
#     latest agreement score each step -- maximally reactive, no smoothing.
# If even these extremes don't separate more than the noise band already
# measured (~+-0.15 in r, ~+-0.007 in whole-run MSE), that's real evidence
# rho's effect is small relative to this system's noise floor, not just
# "we didn't scale far enough."
#
# Single seed (42) each -- a fast gut-check, not a replicated result.
#
# Run: python -u experiments/stage6_spatial_mesh/run_rho_extremes.py

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
RHO_EXTREMES = [0.0001, 0.99]


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
            title=f"rho extremes ({tag})",
            offset_field=field,
            **kw,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res["certainty_mse_pearson_r"]
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f} r={r:.4f}")

    for rho in RHO_EXTREMES:
        run(f"consensus_rho{rho}", "consensus", rho=rho)

    print("\n=== RHO EXTREMES SUMMARY ===")
    for rho in RHO_EXTREMES:
        last50, whole, r = results[f"consensus_rho{rho}"]
        print(f"  rho={rho}: last50={last50:.4f} whole_run={whole:.4f} r={r:.4f}")
    print("(reference: plain fusion (seed 42) whole_run=0.0252 r~-0.2 to -0.3 range; "
          "consensus default rho=0.2 (seed 42) whole_run=0.0256 r=-0.2308)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
