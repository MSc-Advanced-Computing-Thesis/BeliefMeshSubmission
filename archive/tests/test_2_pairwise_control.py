"""Experiment Specification Sec 9, validation order step 2: two-expert control.
HARD GATE.

Unweighted product-of-experts at N=2 must reproduce the prior repo's
bayesian_fusion_grid on identical inputs. The oracle below is copied verbatim
from Experiments_node-learning/experiments/Stage 4/stage_4c.py -- do not 'fix' it.

One documented difference exists BY DESIGN: the old function evaluates
log-densities flat (no circular wrap), so on inputs where the wrap activates
near the winning mode -- boundary beliefs, or contributors on opposite sides of
the circle -- the corrected output legitimately differs (that is Defect-adjacent
behaviour the correction removes; see test_1). The gate therefore asserts:
  (a) EXACT equality on controlled inputs where wrapping cannot influence the
      argmax, and
  (b) on realistic model-output inputs, every disagreement is a genuine
      boundary case.
"""

from __future__ import annotations

import pytest
import torch

from beliefmesh.fusion.product_of_experts import fuse

N_POINTS = 200  # the old function's grid


def old_bayesian_fusion_grid(gamma_a, nu_a, alpha_a, beta_a,
                             gamma_c, nu_c, alpha_c, beta_c,
                             n_points=N_POINTS):
    """Verbatim oracle from the prior repo's stage_4c.py."""
    def student_t_log_prob(y, gamma, nu, alpha, beta):
        df = 2 * alpha
        scale = torch.sqrt(beta * (1 + nu) / (nu * alpha).clamp(min=1e-6))
        dist = torch.distributions.StudentT(df=df, loc=gamma, scale=scale)
        return dist.log_prob(y)

    grid = torch.linspace(-1, 1, n_points, device=gamma_a.device)
    batch_size = gamma_a.shape[0]
    grid = grid.unsqueeze(0).expand(batch_size, -1)

    def expand(t):
        return t.unsqueeze(1).expand(-1, n_points)

    log_prob_a = student_t_log_prob(grid, expand(gamma_a), expand(nu_a), expand(alpha_a), expand(beta_a))
    log_prob_c = student_t_log_prob(grid, expand(gamma_c), expand(nu_c), expand(alpha_c), expand(beta_c))
    log_joint = log_prob_a + log_prob_c
    mode_idx = log_joint.argmax(dim=1)
    return grid[torch.arange(batch_size), mode_idx]


def old_grid():
    return torch.linspace(-1, 1, N_POINTS)


def _controlled_inputs(n, seed):
    """Inputs where the wrap cannot influence the argmax: locations well inside
    the domain, moderate scales, so the joint mode dominates any far tail."""
    g = torch.Generator().manual_seed(seed)
    r = lambda lo, hi: torch.rand(n, generator=g) * (hi - lo) + lo
    a = (r(-0.5, 0.5), r(0.5, 3.0), r(1.5, 4.0), r(0.05, 0.5))
    c = (r(-0.5, 0.5), r(0.5, 3.0), r(1.5, 4.0), r(0.05, 0.5))
    return a, c


def test_hard_gate_exact_equality_on_controlled_inputs():
    for seed in [0, 1, 2, 42, 1234]:
        a, c = _controlled_inputs(500, seed)
        oracle = old_bayesian_fusion_grid(*a, *c)
        ours, _ = fuse([a, c], old_grid())
        assert torch.equal(ours, oracle), (
            f"HARD GATE FAILED (seed {seed}): N=2 unweighted fuse() diverged from "
            f"bayesian_fusion_grid on wrap-free inputs -- the N-way math is wrong. "
            f"{(ours != oracle).sum().item()}/500 samples differ."
        )


def test_hard_gate_weights_of_one_change_nothing():
    a, c = _controlled_inputs(300, 7)
    unweighted, _ = fuse([a, c], old_grid())
    weighted, _ = fuse([a, c], old_grid(), weights=torch.tensor([1.0, 1.0]))
    assert torch.equal(unweighted, weighted)


@pytest.mark.skipif(
    not __import__("pathlib").Path("runs/stage3/naive_average/checkpoints/node_a.pth").exists(),
    reason="stage 3 checkpoints not present",
)
def test_hard_gate_realistic_inputs_disagreements_are_boundary_only():
    """On real model outputs (Stage 3's A and C predicting on red images), the
    corrected fusion must agree with the oracle everywhere except genuine
    circular-boundary cases, which the oracle handles wrongly by design."""
    import random

    import numpy as np

    from beliefmesh.data.digits import RotatedDigitDataset, get_digit7_splits
    from beliefmesh.models.evidential_cnn import EvidentialCNN
    from torch.utils.data import DataLoader

    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    _, eval_idx = get_digit7_splits()
    loader = DataLoader(
        RotatedDigitDataset(filter_type="red", filter_strength=0.8,
                            subset_indices=eval_idx[:256]),
        batch_size=256, shuffle=False)

    def load(p):
        m = EvidentialCNN()
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        return m

    model_a = load("runs/stage3/naive_average/checkpoints/node_a.pth")
    model_c = load("runs/stage3/naive_average/checkpoints/node_c.pth")
    images, _ = next(iter(loader))
    with torch.no_grad():
        a = model_a(images)
        c = model_c(images)

    oracle = old_bayesian_fusion_grid(*a, *c)
    ours, _ = fuse([a, c], old_grid())

    agree = torch.isclose(ours, oracle)
    match_rate = agree.float().mean().item()
    # rotation labels are uniform on the circle, so a real fraction of samples
    # (~15%: within ~25 deg of the boundary) legitimately diverge -- those are
    # the cases the old flat evaluation got WRONG and the correction fixes.
    # The load-bearing assertion is that every single disagreement is such a
    # boundary case; the match rate is a sanity floor, not the gate.
    assert match_rate >= 0.6, f"only {match_rate:.1%} agreement -- too many to be boundary-only"
    print(f"realistic-input agreement: {match_rate:.1%} "
          f"({(~agree).sum().item()} boundary-case divergences)")

    for i in torch.nonzero(~agree).flatten().tolist():
        near_boundary = abs(oracle[i].item()) > 0.9 or abs(ours[i].item()) > 0.9
        opposite_sides = abs(a[0][i].item() - c[0][i].item()) > 1.0
        assert near_boundary or opposite_sides, (
            f"sample {i}: non-boundary disagreement (oracle {oracle[i]:.3f}, "
            f"ours {ours[i]:.3f}, gammas {a[0][i]:.3f}/{c[0][i]:.3f}) -- math error, not wrap"
        )
