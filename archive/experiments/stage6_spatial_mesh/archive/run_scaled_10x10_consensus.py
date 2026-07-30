# Adds a consensus arm to the existing scaled 10x10 test (run_scaled_10x10_test.py,
# which already produced scaled_fusion and scaled_fedavg_global -- not rerun here),
# using today's best 5-seed-validated config: lam=10.0, rho=0.99. Question: does
# tuned consensus actually beat fedavg_global (or plain fusion) at this scale, now
# that the certainty signal has real spread to work with?
#
# Reuses the exact saved scaled_field.npy from the original run for a fair,
# identical-task comparison (not regenerated).
#
# Run: python -u experiments/stage6_spatial_mesh/run_scaled_10x10_consensus.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_random_wander_path
from stage6_spatial_mesh.run_6abc_overlap_ablation import node_grid

from beliefmesh.config import load_config

CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_scaled_10x10")
G = 31
T = 390
FOV = 7
STRIDE = 3
N_WEARABLES = round(3 * (G / 22) ** 2)
LAM = 10.0
RHO = 0.99


def main():
    cfg = load_config()
    centres = node_grid(FOV, STRIDE, G)
    print(f"nodes: {len(centres)} (expect 100)")
    uniform = np.full((T, G, G), 0.5)

    field = np.load(ROOT / "scaled_field.npy")  # exact same task as fusion/global arms
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = "scaled_consensus_lam10_rho099"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=FOV, mode="consensus", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Scaled 10x10, tuned consensus (lam={LAM}, rho={RHO})",
        offset_field=field, lam=LAM, rho=RHO,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")
    print("(reference, already saved: scaled_fusion last50=0.0226 whole_run=0.0347)")
    print("(reference, already saved: scaled_fedavg_global last50=0.0197 whole_run=0.0260)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
