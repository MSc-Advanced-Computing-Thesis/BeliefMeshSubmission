"""Consensus / reliability weighting. Spec Sec 6.

Disagreement among N contributors via Jensen-Shannon divergence on the fusion
grid (mixture m = mean of contributor densities). Consensus compounds through
the propagation chain rather than resetting per hop -- see update_consensus.
"""

from __future__ import annotations

import torch


def js_divergence(log_densities: torch.Tensor) -> torch.Tensor:
    """log_densities: (N, G) per-contributor log-density over the fusion grid. Returns scalar D_js in [0, log 2]."""
    raise NotImplementedError


def update_consensus(contributor_consensus: torch.Tensor, d_js: torch.Tensor, prior_consensus: float, rho: float) -> float:
    """Spec Sec 6.2:
    agreement = 1 - d_js / log(2)
    inherited = sum(c_i * c_i) / sum(c_i)   -- consensus-weighted by itself, Spec Sec 6.2
    c_new = agreement * inherited
    returns (1 - rho) * prior_consensus + rho * c_new
    """
    raise NotImplementedError
