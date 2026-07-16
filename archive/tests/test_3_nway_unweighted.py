"""Experiment Specification Sec 9, validation order step 3: N-way unweighted fusion.

The corrected mechanism at N > 2, no consensus term. Spec Sec 4.1: "the N-way
rule is the pairwise rule with more terms" -- nothing here is pairwise, and
these tests pin the properties that make the sum-of-logs construction sound at
mesh-scale N (up to the 9-way overlap Stage 6 produces).
"""

from __future__ import annotations

import itertools

import torch

from beliefmesh.fusion.grid import circular_grid
from beliefmesh.fusion.product_of_experts import fuse, log_densities

G = 360
DEG = 1.0 / 180.0


def belief(gamma_deg, nu=4.0, alpha=4.0, beta=0.02, batch=1):
    t = lambda v: torch.full((batch,), float(v))
    return (t(gamma_deg * DEG), t(nu), t(alpha), t(beta))


def test_nway_is_the_sum_with_more_terms():
    # fuse([a,b,c]) must equal argmax of the manually-summed three log-densities:
    # the N-way rule IS the pairwise rule with more terms (Spec Sec 4.1).
    grid = circular_grid(G)
    beliefs = [belief(30.0), belief(45.0, nu=1.0), belief(-10.0, beta=0.5)]
    modes, logp = fuse(beliefs, grid)
    manual = grid[log_densities(beliefs, grid).sum(0).argmax(dim=-1)]
    assert torch.equal(modes, manual)


def test_order_invariance():
    # summation is commutative: any permutation of contributors gives the
    # bit-identical mode.
    grid = circular_grid(G)
    beliefs = [belief(30.0), belief(-100.0, nu=0.5), belief(160.0, beta=0.4),
               belief(-179.0, alpha=2.0), belief(5.0, nu=8.0)]
    reference, _ = fuse(beliefs, grid)
    for perm in itertools.permutations(range(5)):
        modes, _ = fuse([beliefs[i] for i in perm], grid)
        assert torch.equal(modes, reference)


def test_n_identical_beliefs_mode_invariant_in_n():
    # summing N copies of the same log-density scales it by N: argmax unmoved.
    grid = circular_grid(G)
    b = belief(72.0)
    reference, _ = fuse([b, b], grid)
    for n in range(3, 10):
        modes, _ = fuse([b] * n, grid)
        assert torch.equal(modes, reference)
    assert abs(reference.item() * 180.0 - 72.0) <= 0.5  # at the grid candidate nearest gamma


def test_agreeing_majority_beats_sharp_outlier():
    # many-expert reinforcement: eight moderately-sharp beliefs agreeing at +40
    # outweigh one very sharp outlier at -140. This is the N-large behaviour
    # Stage 6D depends on and the spec explicitly declines to assume from N=2
    # (Sec 8, 6D correction) -- here it is verified for the primitive itself.
    grid = circular_grid(G)
    majority = [belief(40.0 + jitter, nu=1.5, alpha=2.5, beta=0.05)
                for jitter in (-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2)]
    outlier = belief(-140.0, nu=9.0, alpha=9.0, beta=0.005)
    modes, _ = fuse(majority + [outlier], grid)
    mode_deg = modes.item() * 180.0
    # The majority side wins -- but note the measured behaviour: the outlier's
    # heavy tail DRAGS the mode ~13 deg off the majority centre (observed:
    # ~27 deg for a majority at 40). Sharp wrong beliefs perturb even when
    # they lose; that residual influence is part of why consensus tempering
    # (Spec Sec 6) exists at all.
    assert abs(mode_deg - 40.0) < 20.0, "majority did not hold the mode"
    assert abs(mode_deg - (-140.0)) > 90.0, "outlier captured the mode"


def test_sharp_outlier_can_win_at_n2():
    # ...and the contrast at N=2: one moderate belief vs the same sharp outlier
    # -- the outlier dominates. Together with the test above this demonstrates
    # that N-way behaviour is genuinely different from pairwise, which is why
    # Stage 6D cannot be extrapolated from Stage 4 (Spec Sec 8).
    grid = circular_grid(G)
    moderate = belief(40.0, nu=1.5, alpha=2.5, beta=0.05)
    outlier = belief(-140.0, nu=9.0, alpha=9.0, beta=0.005)
    modes, _ = fuse([moderate, outlier], grid)
    assert abs(modes.item() * 180.0 - (-140.0)) < 5.0


def test_nway_boundary_wraparound():
    # three beliefs straddling the boundary (+179, -179, +178) fuse to one
    # boundary mode, not an interior compromise.
    grid = circular_grid(G)
    modes, _ = fuse([belief(179.0), belief(-179.0), belief(178.0)], grid)
    assert abs(modes.item() * 180.0) >= 178.0


def test_batched_mesh_scale_shapes():
    # N=9 contributors, batch of 17 cells, G=360: shapes and finiteness at the
    # scale Stage 6's nine-way overlap actually produces.
    grid = circular_grid(G)
    g = torch.Generator().manual_seed(3)
    beliefs = []
    for _ in range(9):
        gamma = torch.rand(17, generator=g) * 2 - 1
        nu = torch.rand(17, generator=g) * 3 + 0.1
        alpha = torch.rand(17, generator=g) * 3 + 1.1
        beta = torch.rand(17, generator=g) * 1 + 0.01
        beliefs.append((gamma, nu, alpha, beta))
    modes, logp = fuse(beliefs, grid)
    assert modes.shape == (17,)
    assert logp.shape == (9, 17, G)
    assert torch.isfinite(logp).all()
    assert ((modes >= -1) & (modes < 1)).all()
