"""Verification gates for the closed-form nig_product fusion, required before
running any nig_product experiment (Christian's explicit instruction, 2026-08):
single-contributor identity, two-identical-contributors agreement, beta_star
strictly increasing under disagreement, and the nu_star/alpha_star guards.
"""

from __future__ import annotations

import pytest

from beliefmesh.fusion.nig_product import fuse_nig_product


def test_single_contributor_returns_unchanged():
    belief = (0.3, 2.0, 3.0, 1.5)
    gamma_star, nu_star, alpha_star, beta_star = fuse_nig_product([belief])
    assert gamma_star == pytest.approx(0.3)
    assert nu_star == pytest.approx(2.0)
    assert alpha_star == pytest.approx(3.0)
    assert beta_star == pytest.approx(1.5)


def test_two_identical_contributors_gamma_matches_common_value():
    belief = (0.5, 1.5, 2.5, 0.8)
    gamma_star, nu_star, alpha_star, beta_star = fuse_nig_product([belief, belief])
    assert gamma_star == pytest.approx(0.5)
    assert nu_star == pytest.approx(3.0)  # sum of nu, both contributors identical


def test_beta_star_increases_with_disagreement():
    nu, alpha, beta = 2.0, 3.0, 1.0
    prev_beta_star = None
    for gap in (0.0, 0.05, 0.1, 0.2, 0.4):
        b1 = (0.0, nu, alpha, beta)
        b2 = (gap, nu, alpha, beta)
        _, _, _, beta_star = fuse_nig_product([b1, b2])
        if prev_beta_star is not None:
            assert beta_star > prev_beta_star, f"beta_star did not increase at gap={gap}"
        prev_beta_star = beta_star
    # at gap=0 (perfect agreement) beta_star must equal the flat sum with no
    # disagreement correction
    b_same = (0.0, nu, alpha, beta)
    _, _, _, beta_star_agree = fuse_nig_product([b_same, b_same])
    assert beta_star_agree == pytest.approx(2 * beta)


def test_disagreement_across_circular_wraparound_still_increases_beta():
    # two contributors near opposite ends of [-1, 1) that are actually CLOSE
    # on the circle (e.g. 0.98 and -0.98 are 0.04 apart, not 1.96 apart) --
    # the flat (non-circular) formula would wildly overstate their
    # disagreement; the circular-safe version must not.
    nu, alpha, beta = 2.0, 3.0, 1.0
    b1 = (0.98, nu, alpha, beta)
    b2 = (-0.98, nu, alpha, beta)
    gamma_star, _, _, beta_star_close = fuse_nig_product([b1, b2])
    # true circular midpoint of 0.98 and -0.98 (0.04 apart) is +/-1.0 (wrapped)
    assert abs(abs(gamma_star) - 1.0) < 1e-6
    # a genuinely disagreeing pair only 0.04 apart in flat terms should have
    # a much smaller beta_star than a pair split by a real half-turn (gap=0.5)
    b3 = (0.0, nu, alpha, beta)
    b4 = (0.5, nu, alpha, beta)
    _, _, _, beta_star_far = fuse_nig_product([b3, b4])
    assert beta_star_close < beta_star_far


def test_nu_star_underflow_guard():
    tiny = 1e-10
    belief = (0.1, tiny, 2.0, 1.0)
    gamma_star, nu_star, alpha_star, beta_star = fuse_nig_product([belief, belief])
    assert nu_star > 0.0
    assert gamma_star == gamma_star  # not NaN
    assert beta_star == beta_star    # not NaN


def test_alpha_star_floor_guard():
    # low weights push n_eff well below 1, which would otherwise drive
    # alpha_star at or below 1 (undefined aleatoric term, beta/(nu*(alpha-1)))
    belief = (0.0, 1.0, 1.01, 1.0)
    gamma_star, nu_star, alpha_star, beta_star = fuse_nig_product(
        [belief, belief], weights=[0.01, 0.01])
    assert alpha_star > 1.0


def test_weighted_matches_unweighted_when_weights_are_all_one():
    beliefs = [(0.1, 1.0, 2.0, 0.5), (-0.2, 2.0, 3.0, 0.8), (0.3, 1.5, 2.5, 0.6)]
    unweighted = fuse_nig_product(beliefs)
    weighted = fuse_nig_product(beliefs, weights=[1.0, 1.0, 1.0])
    assert unweighted == pytest.approx(weighted)
