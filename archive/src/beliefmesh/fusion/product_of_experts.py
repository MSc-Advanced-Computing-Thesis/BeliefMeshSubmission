"""Product-of-experts belief fusion. Spec Sec 4.

One function for all N >= 2 -- there is no separate pairwise code path; N=2 is
just the sum with two terms (validated against the prior repo's
bayesian_fusion_grid in tests/test_2_pairwise_control.py, the hard gate).

    total(y) = sum_i w_i * log p_i(y)     evaluated on the circular grid
    fused_mode = grid[argmax(total)]

Confidence enters through density shape only (Spec Sec 4.2): a sharp belief's
log-density falls steeply away from its location and pulls the argmax; a
diffuse one barely perturbs the sum. No external certainty scalar is applied.

Circular wraparound (Spec Sec 4.3) is mandatory and lives here, not in the
Student-t: each contributor's log-density is evaluated at the WRAPPED offset
circular_diff(grid, gamma) with a zero-location Student-t. Where wrapping is
inactive this is bit-identical to the prior repo's flat evaluation; where it
is active it removes the spurious twin-mode failure at the angle boundary.

Consensus weights w_i are applied POST-log (Spec Sec 6.2): scaling log p
tempers the density (p^w). Pre-log scaling adds a constant in y and cannot
move the argmax -- verified empirically in tests/test_1_fusion_primitive.py.
"""

from __future__ import annotations

import torch

from beliefmesh.metrics.circular import circular_diff

Belief = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def log_densities(beliefs: list[Belief], grid: torch.Tensor) -> torch.Tensor:
    """Per-contributor Student-t log-densities on the grid, shape (N, B, G).

    Each belief is (gamma, nu, alpha, beta), each of shape (B,). The Student-t
    is the NIG marginal predictive (Spec Sec 3): df 2*alpha, scale
    sqrt(beta*(1+nu)/(nu*alpha)) -- the (nu*alpha) clamp matches the prior
    repo's formula so the N=2 parity gate compares identical arithmetic.
    """
    gamma = torch.stack([b[0] for b in beliefs]).unsqueeze(-1)   # (N, B, 1)
    nu = torch.stack([b[1] for b in beliefs]).unsqueeze(-1)
    alpha = torch.stack([b[2] for b in beliefs]).unsqueeze(-1)
    beta = torch.stack([b[3] for b in beliefs]).unsqueeze(-1)

    diff = circular_diff(grid.view(1, 1, -1), gamma)             # (N, B, G), wrapped
    scale = torch.sqrt(beta * (1 + nu) / (nu * alpha).clamp(min=1e-6))
    dist = torch.distributions.StudentT(df=2 * alpha, loc=torch.zeros_like(scale), scale=scale)
    return dist.log_prob(diff)


def fuse(
    beliefs: list[Belief],
    grid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse N contributors' beliefs into per-sample modes.

    beliefs: N tuples of (gamma, nu, alpha, beta), each tensor shape (B,).
    grid: candidate angles, shape (G,) -- normally circular_grid(G).
    weights: optional per-contributor tempering exponents, shape (N,).
             None means unweighted (all ones) -- Spec Sec 9 step 3.

    Returns (fused_modes (B,), log_densities (N, B, G)). The log-density array
    is returned so consensus (Spec Sec 6.5) can reuse it without rebuilding.
    """
    if len(beliefs) < 2:
        raise ValueError(f"fuse() needs at least 2 contributors, got {len(beliefs)}")

    logp = log_densities(beliefs, grid)                          # (N, B, G)
    if weights is None:
        total = logp.sum(dim=0)                                  # (B, G)
    else:
        total = (weights.view(-1, 1, 1) * logp).sum(dim=0)
    mode_idx = total.argmax(dim=-1)                              # (B,)
    return grid[mode_idx], logp
