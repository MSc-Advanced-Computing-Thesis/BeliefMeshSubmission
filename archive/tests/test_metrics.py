"""Tests for beliefmesh.metrics.circular. Not part of Section 9's numbered validation
order (those are reserved for the fusion primitive) -- this covers the foundational
wraparound arithmetic that circular_diff/circular_mse, nig_loss, and the fusion grid
all end up depending on.

Angles here are normalised (angle_degrees / 180), period 2, matching the prior
experiment's convention -- see circular.py's module docstring for why.
"""

from __future__ import annotations

import torch

from beliefmesh.metrics.circular import circular_diff, circular_mse


def test_ordinary_diff_no_wrap():
    torch.testing.assert_close(circular_diff(torch.tensor(0.3), torch.tensor(0.1)), torch.tensor(0.2))
    torch.testing.assert_close(circular_diff(torch.tensor(0.1), torch.tensor(0.3)), torch.tensor(-0.2))


def test_wraps_near_boundary():
    # the spec's own example (Sec 9 step 1), converted to normalised units:
    # +179 deg and -179 deg are 2 degrees apart, not 358 -- here that's
    # +179/180 and -179/180, which should differ by ~2/180, not ~358/180.
    a = torch.tensor(179.0 / 180.0)
    b = torch.tensor(-179.0 / 180.0)
    torch.testing.assert_close(circular_diff(a, b), torch.tensor(-2.0 / 180.0))
    torch.testing.assert_close(circular_diff(b, a), torch.tensor(2.0 / 180.0))


def test_self_diff_is_zero():
    for angle in [0.0, 0.3, -0.9, 1.0, -1.0]:
        a = torch.tensor(angle)
        assert circular_diff(a, a).item() == 0.0


def test_antipodal_boundary_has_correct_magnitude():
    # exactly half the period apart (equivalent to 180 degrees): round-half-to-even
    # means the sign returned isn't guaranteed, but +1.0 and -1.0 are the same
    # physical angle, so only the magnitude is meaningful to assert here.
    diff = circular_diff(torch.tensor(0.5), torch.tensor(-0.5)).item()
    assert abs(diff) == 1.0


def test_batch_shapes():
    a = torch.tensor([30.0, 179.0, 0.0]) / 180.0
    b = torch.tensor([10.0, -179.0, 180.0]) / 180.0
    diff = circular_diff(a, b)
    assert diff.shape == (3,)
    expected = torch.tensor([20.0, -2.0, -180.0]) / 180.0
    torch.testing.assert_close(diff, expected)


def test_circular_mse_matches_manual_calculation():
    pred = torch.tensor([179.0, 30.0]) / 180.0
    target = torch.tensor([-179.0, 10.0]) / 180.0
    expected = ((-2.0 / 180.0) ** 2 + (20.0 / 180.0) ** 2) / 2
    torch.testing.assert_close(circular_mse(pred, target).item(), expected)


def test_circular_mse_symmetric():
    pred = torch.tensor([179.0, 30.0, -45.0]) / 180.0
    target = torch.tensor([-179.0, 10.0, 100.0]) / 180.0
    torch.testing.assert_close(circular_mse(pred, target), circular_mse(target, pred))


def test_circular_mse_zero_for_identical_inputs():
    x = torch.tensor([12.0, -170.0, 180.0]) / 180.0
    assert circular_mse(x, x).item() == 0.0


def test_gradient_flows_like_ordinary_subtraction_away_from_wrap():
    # round() contributes zero gradient almost everywhere, so away from the wrap
    # boundary circular_mse's gradient should match plain (unwrapped) MSE's gradient
    # exactly -- this matters because nig_loss will backprop through this.
    a = torch.tensor([30.0 / 180.0, 179.0 / 180.0], requires_grad=True)
    b = torch.tensor([10.0, -179.0]) / 180.0
    loss = circular_mse(a, b)
    loss.backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    diff = circular_diff(a.detach(), b)
    expected_grad = 2.0 * diff / diff.numel()
    torch.testing.assert_close(a.grad, expected_grad)
