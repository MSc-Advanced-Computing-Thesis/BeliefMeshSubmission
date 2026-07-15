"""Experiment Specification Sec 9, validation order step 1: fusion primitive.

1. Circular wraparound: beliefs at +179 and -179 degrees must fuse to ONE
   shared mode at the boundary, not average to zero / split into twin modes.
2. Pre-log scaling of a density is a no-op on the argmax (Spec Sec 6.2);
   post-log tempering is not. Verified empirically, not asserted.
"""

from __future__ import annotations

import torch

from beliefmesh.fusion.grid import circular_grid
from beliefmesh.fusion.product_of_experts import fuse, log_densities

G = 360
DEG = 1.0 / 180.0  # one degree in normalised units


def belief(gamma_deg: float, nu=4.0, alpha=4.0, beta=0.02):
    """A single sharp belief at gamma_deg degrees, batch size 1."""
    t = lambda v: torch.tensor([float(v)])
    return (t(gamma_deg * DEG), t(nu), t(alpha), t(beta))


def test_wraparound_agreeing_boundary_beliefs_fuse_to_one_boundary_mode():
    # +179 deg and -179 deg describe nearly the same angle. Wrapped fusion must
    # place the mode at the boundary (within ~2 deg of +/-180), NOT near 0
    # (which is what flat averaging of the two locations would give).
    grid = circular_grid(G)
    mode, _ = fuse([belief(179.0), belief(-179.0)], grid)
    mode_deg = mode.item() * 180.0
    assert abs(mode_deg) >= 178.0, f"mode at {mode_deg:.1f} deg -- boundary agreement was split"


def test_flat_evaluation_produces_spurious_twin_modes():
    # Spec Sec 4.3: without wrapping, "the product then exhibits two spurious
    # disjoint modes rather than one sharp shared mode". Demonstrate exactly
    # that: flat evaluation of the +/-179 case has its two best candidates at
    # OPPOSITE ends of the grid index range (disjoint lobes of near-equal
    # height), where wrapped evaluation has them adjacent (see
    # test_single_sharp_mode_not_twin_modes).
    grid = circular_grid(G)
    beliefs = [belief(179.0), belief(-179.0)]
    flat_logp = []
    for g, n, a, b in beliefs:
        scale = torch.sqrt(b * (1 + n) / (n * a).clamp(min=1e-6))
        dist = torch.distributions.StudentT(df=2 * a, loc=g, scale=scale)
        flat_logp.append(dist.log_prob(grid.view(1, -1)))
    total = torch.stack(flat_logp).sum(0)[0]
    top2 = total.topk(2).indices
    raw_index_gap = (top2[0] - top2[1]).abs().item()  # RAW gap: flat has no ring semantics
    assert raw_index_gap > G // 2, (
        f"expected disjoint twin modes under flat evaluation, top-2 only "
        f"{raw_index_gap} indices apart")
    # and the two lobes are near-equal height -- genuinely ambiguous twin modes
    heights = total.topk(2).values
    assert (heights[0] - heights[1]).abs().item() < 1.0


def test_single_sharp_mode_not_twin_modes():
    # the fused log-density for the +/-179 case must have its top-two grid
    # candidates adjacent on the circle (one connected peak), not on opposite
    # sides of the domain.
    grid = circular_grid(G)
    _, logp = fuse([belief(179.0), belief(-179.0)], grid)
    total = logp.sum(0)[0]
    top2 = total.topk(2).indices
    ring_dist = (top2[0] - top2[1]).abs().item()
    ring_dist = min(ring_dist, G - ring_dist)
    assert ring_dist == 1, f"top-two candidates {ring_dist} steps apart -- twin modes"


def test_prelog_scaling_is_noop_on_argmax():
    # scaling the DENSITY before the log adds log(w), constant in y: the argmax
    # must be bit-identical to unweighted for any positive scalings.
    grid = circular_grid(G)
    beliefs = [belief(40.0, nu=3.0), belief(-20.0, nu=0.4, beta=0.3)]
    logp = log_densities(beliefs, grid)
    unweighted_idx = logp.sum(0).argmax(dim=-1)
    for w in ([0.01, 0.99], [5.0, 0.2], [0.5, 0.5]):
        prelog = logp + torch.log(torch.tensor(w)).view(-1, 1, 1)  # log(w * p)
        assert torch.equal(prelog.sum(0).argmax(dim=-1), unweighted_idx)


def test_postlog_tempering_does_move_argmax():
    # the same weights applied POST-log (w * log p) must be able to move the
    # mode -- otherwise consensus tempering (Spec Sec 6.2) would be inert.
    grid = circular_grid(G)
    b1, b2 = belief(40.0), belief(-40.0)  # equally sharp, symmetric
    unweighted_mode, _ = fuse([b1, b2], grid)
    tempered_mode, _ = fuse([b1, b2], grid, weights=torch.tensor([1.0, 0.05]))
    # down-tempering b2 must pull the mode toward b1 (+40 deg)
    assert tempered_mode.item() > unweighted_mode.item()
    assert abs(tempered_mode.item() * 180.0 - 40.0) < 3.0


def test_diffuse_belief_barely_perturbs_sharp_one():
    # Spec Sec 4.2: confidence enters through shape. A very diffuse contributor
    # must not drag the mode meaningfully away from a sharp one's location.
    grid = circular_grid(G)
    sharp = belief(60.0, nu=6.0, alpha=6.0, beta=0.01)
    diffuse = belief(-120.0, nu=0.05, alpha=1.1, beta=2.0)
    mode, _ = fuse([sharp, diffuse], grid)
    assert abs(mode.item() * 180.0 - 60.0) <= 2.0
