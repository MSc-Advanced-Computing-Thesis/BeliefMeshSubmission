# Resume of run_lam_at_best_rho.py after the original process was killed by
# a session restart mid-sweep (consensus_lam0.1_rho0.05 died at step 290/390,
# no manifest saved -- needs a full redo, not a partial resume). Already
# completed and saved: rho=0.01 (r=-0.5199), lam=0.01@rho=0.05 (r=-0.3805),
# lam=0.05@rho=0.05 (r=-0.4521). This just runs the remaining lam=0.1/0.2/0.4
# at the already-determined best_rho=0.05.
#
# Run: python -u experiments/stage6_spatial_mesh/run_lam_at_best_rho_resume.py

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
BEST_RHO = 0.05
REMAINING_LAM = [0.1, 0.2, 0.4]


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
            title=f"lam-at-best-rho resume ({tag})",
            offset_field=field,
            **kw,
        )
        cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
        whole_run_mse = float(np.nanmean(cell_mse_steps))
        r = res["certainty_mse_pearson_r"]
        results[tag] = (res["mean_mse_last_50"], whole_run_mse, r)
        print(f"### {tag}: last50={res['mean_mse_last_50']:.4f} "
              f"whole_run={whole_run_mse:.4f} r={r:.4f}")

    for lam in REMAINING_LAM:
        run(f"consensus_lam{lam}_rho{BEST_RHO}", "consensus", lam=lam, rho=BEST_RHO)

    print("\n=== RESUME SWEEP SUMMARY (remaining lam values @ rho=0.05) ===")
    for lam in REMAINING_LAM:
        last50, whole, r = results[f"consensus_lam{lam}_rho{BEST_RHO}"]
        print(f"  lam={lam}: last50={last50:.4f} whole_run={whole:.4f} r={r:.4f}")
    print("(already completed: lam=0.01 r=-0.3805 whole_run=0.0263; "
          "lam=0.05 r=-0.4521 whole_run=0.0260)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
