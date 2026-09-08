"""Consensus / reliability weighting. Spec Sec 6.

Disagreement among N contributors is the generalised Jensen-Shannon divergence
(information radius): D = (1/N) sum_i KL(p_i || m) with the uniform mixture
m = (1/N) sum_i p_i, computed on the same (N, G) log-density array fusion
already built (Sec 6.5: one extra logsumexp, no rebuild).

NOTE, correcting the spec's Sec 6.2 text: the uniform-weight generalised JS is
bounded above by log(N), not log(2) -- log(2) is the two-distribution special
case. Hence agreement = 1 - D_js / log(N), with agreement = 1 by convention at
N = 1 (a single contributor cannot disagree with itself).

    agreement  = 1 - D_js / log(N)
    inherited  = sum_i c_i^2 / sum_i c_i     (consensus-weighted inheritance)
    c_new      = agreement * inherited
    c_receiver <- (1 - rho) * c_receiver + rho * c_new

Consensus compounds through the propagation chain (inherited term) and is the
tempering exponent w_i applied POST-log inside fuse() (Sec 6.2). It governs
propagation; it is not itself propagated (Sec 5).
"""

from __future__ import annotations

import math

import torch


def js_divergence(log_densities: torch.Tensor) -> torch.Tensor:
    """Generalised JS divergence per sample.

    log_densities: (N, B, G) per-contributor log-densities over the fusion grid
    (unnormalised is fine -- each row is normalised on the grid internally).
    Returns (B,) divergences in [0, log N].
    """
    n = log_densities.shape[0]
    # normalise each contributor's density over the grid (discrete distribution)
    logp = log_densities - torch.logsumexp(log_densities, dim=-1, keepdim=True)  # (N,B,G)
    # uniform mixture in log space: log m = logsumexp_i(logp_i) - log N
    logm = torch.logsumexp(logp, dim=0) - math.log(n)                             # (B,G)
    # D = (1/N) sum_i KL(p_i || m) = (1/N) sum_i sum_g p_i (logp_i - logm)
    kl = (logp.exp() * (logp - logm.unsqueeze(0))).sum(dim=-1)                    # (N,B)
    return kl.mean(dim=0).clamp(min=0.0)                                          # (B,)


def agreement_score(log_densities: torch.Tensor) -> torch.Tensor:
    """agreement = 1 - D_js / log(N), in [0, 1]; ones at N = 1 by convention."""
    n = log_densities.shape[0]
    if n == 1:
        return torch.ones(log_densities.shape[1])
    return (1.0 - js_divergence(log_densities) / math.log(n)).clamp(min=0.0, max=1.0)


def inherited_trust(contributor_consensus: torch.Tensor) -> float:
    """Consensus-weighted mean of contributors' consensus: sum c_i^2 / sum c_i.

    Weighting each c_i by itself matches the power its density was raised to in
    the fusion product; a distrusted contributor is doubly discounted (Sec 6.2).
    """
    total = contributor_consensus.sum()
    if total <= 0:
        return 0.0
    return float((contributor_consensus ** 2).sum() / total)


def update_consensus(current: float, c_new: float, rho: float) -> float:
    """Running update: (1 - rho) * current + rho * c_new. Anchors call this
    with c_new = 1.0 (Sec 6.3: no fusion at anchor timesteps, consensus updates
    toward unity)."""
    return (1.0 - rho) * current + rho * c_new
