"""Unit tests for beliefmesh.fusion.consensus (Spec Sec 6.2 mechanics).

The EXPERIMENTAL ablation (does consensus improve mesh MSE?) lives in the
stage-6 runs; these tests pin the math: JS divergence bounds, the agreement
mapping, consensus-weighted inheritance, and the running update.
"""

from __future__ import annotations

import math

import torch

from beliefmesh.fusion.consensus import (agreement_score, inherited_trust,
                                         js_divergence, update_consensus)
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.fusion.product_of_experts import log_densities

G = 360
DEG = 1.0 / 180.0


def beliefs_at(degrees, nu=4.0, alpha=4.0, beta=0.02):
    return [(torch.tensor([d * DEG]), torch.tensor([nu]),
             torch.tensor([alpha]), torch.tensor([beta])) for d in degrees]


def logp_for(degrees, **kw):
    return log_densities(beliefs_at(degrees, **kw), circular_grid(G))


def test_identical_contributors_have_zero_divergence_full_agreement():
    logp = logp_for([40.0, 40.0, 40.0])
    assert js_divergence(logp).item() < 1e-5
    assert abs(agreement_score(logp).item() - 1.0) < 1e-5


def test_disjoint_sharp_contributors_approach_log_n_bound():
    # three sharp beliefs at mutually distant angles: D_js -> log(3), agreement -> 0
    logp = logp_for([-120.0, 0.0, 120.0], beta=0.005, nu=8.0, alpha=8.0)
    d = js_divergence(logp).item()
    assert d <= math.log(3) + 1e-6
    assert d > 0.9 * math.log(3)
    assert agreement_score(logp).item() < 0.1


def test_agreement_decreases_with_spread():
    tight = agreement_score(logp_for([40.0, 41.0, 42.0])).item()
    medium = agreement_score(logp_for([30.0, 45.0, 60.0])).item()
    wide = agreement_score(logp_for([-60.0, 40.0, 140.0])).item()
    assert tight > medium > wide


def test_agreement_single_contributor_is_one_by_convention():
    logp = logp_for([25.0])
    assert torch.equal(agreement_score(logp), torch.ones(1))


def test_divergence_wraps_on_the_circle():
    # +179 and -179 nearly agree; +90 and -90 are maximally distant. The
    # boundary pair must show far LESS disagreement despite the larger flat
    # difference in raw values.
    near_boundary = js_divergence(logp_for([179.0, -179.0])).item()
    far_apart = js_divergence(logp_for([90.0, -90.0])).item()
    assert near_boundary < 0.2 * far_apart


def test_inherited_trust_is_consensus_weighted():
    # sum c^2 / sum c: a barely-trusted contributor barely drags the inherited
    # value, unlike an arithmetic mean (Spec Sec 6.2 rejection of the mean)
    c = torch.tensor([1.0, 0.01])
    inherited = inherited_trust(c)
    assert abs(inherited - (1.0 + 0.0001) / 1.01) < 1e-6
    assert inherited > 0.9  # arithmetic mean would be ~0.5
    assert inherited_trust(torch.tensor([0.0, 0.0])) == 0.0


def test_update_consensus_ema():
    assert abs(update_consensus(1.0, 0.0, rho=0.2) - 0.8) < 1e-9
    assert abs(update_consensus(0.5, 1.0, rho=0.2) - 0.6) < 1e-9
    # anchors converge toward unity under repeated updates (Sec 6.3)
    c = 0.3
    for _ in range(50):
        c = update_consensus(c, 1.0, rho=0.2)
    assert c > 0.99
