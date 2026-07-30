# Follow-up to run_twotone_split_test.py's diagnosis: certainty barely dips
# at the tone boundary (0.985 -> 0.958, ~2.7% relative) while MSE spikes
# hard there (0.211 -> 0.267, ~26% relative) -- the "confidently wrong"
# signature, explained by boundary-crossing training events being a tiny
# fraction of any node's total experience (most of a node's life is spent
# entirely inside one constant-ground-truth tone). Christian's question:
# does jacking lam way up (0.1 -> 15, well past the 10 already tried
# elsewhere today) force enough certainty differentiation to actually flag
# the boundary as untrustworthy, or does the same under-reaction persist
# regardless of lam?
#
# fusion mode only (lam only matters for the precision-weighted product-of-
# experts fusion uses; fedavg_global doesn't consume it), reusing the exact
# same two-tone field.npy already saved.
#
# Run: python -u experiments/stage6_spatial_mesh/run_twotone_lam15_test.py

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
ROOT = Path("runs/stage6/offset_world/dynamic_twotone_split")
N_WEARABLES = 3
LR = 3e-4
LAM = 15.0


def main():
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.load(ROOT / "field.npy")  # exact same two-tone field as before

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = "twotone_fusion_lam15"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="fusion", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Two-tone split, lam={LAM} ({tag})",
        offset_field=field, lam=LAM,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    print(f"\n### {tag}: last50={res['mean_mse_last_50']:.4f} "
          f"whole_run={whole_run_mse:.4f} r={res['certainty_mse_pearson_r']:.4f}")
    print("(reference, lam=0.1: whole_run=0.2156 r=-0.1998)")

    # same boundary-distance calibration check as before
    cert = np.load(ROOT / tag / "avg_cert_map.npy")
    mse = np.load(ROOT / tag / "avg_mse_map.npy")
    boundary_col = G // 2
    rows, cols = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    dist_to_boundary = np.abs(cols - boundary_col + 0.5)
    mask = ~np.isnan(cert) & ~np.isnan(mse)
    d, c, m = dist_to_boundary[mask], cert[mask], mse[mask]
    print("\nboundary-distance calibration check:")
    for lo, hi in [(0, 2), (2, 4), (4, 7), (7, 12)]:
        bm = (d >= lo) & (d < hi)
        if bm.sum():
            print(f"  dist {lo}-{hi}: mean_cert={c[bm].mean():.4f} mean_mse={m[bm].mean():.4f} n={bm.sum()}")
    print("(reference, lam=0.1: dist0-2 cert=0.9579 mse=0.2669 | dist7-12 cert=0.9853 mse=0.2113)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
