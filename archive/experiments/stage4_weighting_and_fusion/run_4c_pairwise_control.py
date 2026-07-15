# Stage 4c: Pairwise control -- grid fusion vs optimisation fusion vs naive.
# Experiment Specification Sec 8 & Sec 9 step 2.
#
# The N=2 HARD GATE itself lives in tests/test_2_pairwise_control.py (bit-exact
# vs the old bayesian_fusion_grid on wrap-free inputs; passed). This script
# generates the experimental comparison: grid search and gradient-descent mode
# estimation should perform comparably (Spec Sec 4.1 -- confirming principled
# mode estimation, not the specific estimator, is the source of gain), with
# naive averaging behind both.
#
# The optimisation variant here uses WRAPPED log-densities (consistent with the
# corrected fusion), initialised at the circular midpoint of the two locations.
#
# Run from the project root: python experiments/stage4_weighting_and_fusion/run_4c_pairwise_control.py

from __future__ import annotations

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
from beliefmesh.metrics.circular import circular_diff

N_OPTIM_STEPS = 50
OPTIM_LR = 0.01


def optim_fusion(a, c):
    """Gradient-descent mode search over the wrapped joint log-density."""
    params = [tuple(t.detach() for t in b) for b in (a, c)]
    gamma_a = params[0][0]
    # initialise at the circular midpoint of the two locations
    theta = (gamma_a + circular_diff(params[1][0], gamma_a) / 2).clone().requires_grad_(True)
    optimiser = torch.optim.Adam([theta], lr=OPTIM_LR)

    for _ in range(N_OPTIM_STEPS):
        optimiser.zero_grad()
        log_p = torch.zeros_like(theta)
        for g, n, al, b in params:
            scale = torch.sqrt(b * (1 + n) / (n * al).clamp(min=1e-6))
            dist = torch.distributions.StudentT(df=2 * al, loc=torch.zeros_like(scale), scale=scale)
            log_p = log_p + dist.log_prob(circular_diff(theta, g))
        (-log_p.sum()).backward()
        optimiser.step()

    # wrap the free-running theta back onto [-1, 1)
    return circular_diff(theta.detach(), torch.zeros_like(theta))


def main():
    cfg = load_config()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx, eval_idx = get_digit7_splits(eval_fraction=cfg.eval_holdout_fraction, seed=cfg.seed)

    grid = circular_grid(cfg.fusion.grid_size, device=device)

    def naive(a, c):
        return (a[0] + c[0]) / 2

    def grid_fusion(a, c):
        modes, _ = fuse([a, c], grid)
        return modes

    final_mse, high_means = run_ramp_experiment(
        cfg, "4c_grid_vs_optim", {"naive": naive, "grid": grid_fusion, "optim": optim_fusion},
        Path("runs/stage4/4c_grid_vs_optim"), train_idx, eval_idx, device,
        extra_manifest={"grid_size": cfg.fusion.grid_size,
                        "optim_steps": N_OPTIM_STEPS, "optim_lr": OPTIM_LR})

    print(f"\nHigh-strength mean MSE: grid {high_means['grid']:.4f}, "
          f"optim {high_means['optim']:.4f}, naive {high_means['naive']:.4f}")


if __name__ == "__main__":
    main()
