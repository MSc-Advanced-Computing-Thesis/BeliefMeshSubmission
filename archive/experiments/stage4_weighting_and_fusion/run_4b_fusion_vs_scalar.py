# Stage 4b: Bayesian (product-of-experts) fusion vs naive averaging.
# Experiment Specification Sec 8. Uses the corrected beliefmesh fuse() with
# the G=360 circular grid -- NOT the old flat 200-point function.
#
# Run from the project root: python experiments/stage4_weighting_and_fusion/run_4b_fusion_vs_scalar.py

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage4_weighting_and_fusion.harness import run_ramp_experiment

from beliefmesh.config import load_config
from beliefmesh.data.digits import get_digit7_splits
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.fusion.product_of_experts import fuse


def main():
    cfg = load_config()
    seed = int(os.environ.get("EXP_SEED", cfg.seed))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx, eval_idx = get_digit7_splits(eval_fraction=cfg.eval_holdout_fraction, seed=cfg.seed)

    run_dir = Path("runs/stage4/4b_fusion_vs_naive")
    if seed != cfg.seed:
        run_dir = run_dir.parent / f"{run_dir.name}_seed{seed}"

    grid = circular_grid(cfg.fusion.grid_size, device=device)

    def naive(a, c):
        return (a[0] + c[0]) / 2

    def fusion(a, c):
        modes, _ = fuse([a, c], grid)
        return modes

    final_mse, high_means = run_ramp_experiment(
        cfg, "4b_fusion_vs_naive", {"naive": naive, "fusion": fusion},
        run_dir, train_idx, eval_idx, device,
        extra_manifest={"grid_size": cfg.fusion.grid_size})

    print(f"\nHigh-strength mean MSE: fusion {high_means['fusion']:.4f} "
          f"vs naive {high_means['naive']:.4f} "
          f"({'fusion wins' if high_means['fusion'] < high_means['naive'] else 'NAIVE WINS -- investigate'})")


if __name__ == "__main__":
    main()
