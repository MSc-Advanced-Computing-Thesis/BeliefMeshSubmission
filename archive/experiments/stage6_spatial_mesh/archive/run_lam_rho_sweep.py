# Calibration sweep: lam (evidence-regularisation coefficient in nig_loss,
# controls confidence-vs-sharpness tradeoff -- Amini et al. deep evidential
# regression) and rho (consensus's trust-EMA decay). Neither has ever been
# swept anywhere in this project; both call sites previously hardcoded their
# defaults (lam=0.1 everywhere nig_loss was called; rho=0.2, Mesh's default,
# never actually passed through by run_mesh_experiment).
#
# Motivation: the 100-node scaled test showed fusion's certainty-MSE
# correlation degrade sharply (r=-0.075, barely significant) vs the standard
# scale's r~-0.2 to -0.3 -- the mesh's whole benefit (efficient hop-to-hop
# belief transfer) depends on certainty actually tracking error. lam directly
# penalises overconfident-but-wrong predictions; rho controls how fast
# consensus's trust signal adapts. Both are candidate levers for improving
# that correlation.
#
# Design: standard 36-node/22-grid dynamic offset world (the validated Stage 4
# environment), 3 random-wandering wearables, same field/seed throughout --
# only lam (fusion arm) or rho (consensus arm) varies. Single seed per point
# (this is a sweep to find a promising region, not a replicated result).
#
# Run: python -u experiments/stage6_spatial_mesh/run_lam_rho_sweep.py

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

LAM_VALUES = [0.01, 0.05, 0.1, 0.2, 0.4]
RHO_VALUES = [0.05, 0.1, 0.2, 0.4, 0.6]


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
            title=f"lam/rho sweep ({tag})",
            offset_field=field,
            **kw,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res["certainty_mse_pearson_r"]
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f} r={r:.4f}")

    # --- lam sweep, fusion mode ---
    for lam in LAM_VALUES:
        run(f"fusion_lam{lam}", "fusion", lam=lam)

    # --- rho sweep, consensus mode ---
    for rho in RHO_VALUES:
        run(f"consensus_rho{rho}", "consensus", rho=rho)

    print("\n=== LAM/RHO SWEEP SUMMARY ===")
    print("-- lam (fusion) --")
    for lam in LAM_VALUES:
        last50, whole, r = results[f"fusion_lam{lam}"]
        print(f"  lam={lam}: last50={last50:.4f} whole_run={whole:.4f} r={r:.4f}")
    print("-- rho (consensus) --")
    for rho in RHO_VALUES:
        last50, whole, r = results[f"consensus_rho{rho}"]
        print(f"  rho={rho}: last50={last50:.4f} whole_run={whole:.4f} r={r:.4f}")
    print("(reference defaults: lam=0.1, rho=0.2 -- these ARE points 3/5 and 3/5 "
          "in the two sweeps above, i.e. the existing baseline)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
