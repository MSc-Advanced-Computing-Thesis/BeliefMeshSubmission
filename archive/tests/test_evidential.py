"""Tests for beliefmesh.models.evidential (nig_loss, student_t_marginal) and the
EvidentialCNN's output constraints.

The parity test against the prior repo's nig_loss is the important one: the only
intended change in the port is replacing the inline period-2 wrap with
circular_diff, so the two must agree to float precision on identical inputs.
"""

from __future__ import annotations

import pytest
import torch

from beliefmesh.models.evidential import (nig_loss, predictive_uncertainty,
                                          student_t_marginal, training_uncertainty)
from beliefmesh.models.evidential_cnn import EvidentialCNN


def _old_nig_loss(gamma, nu, alpha, beta, y, lam=0.1):
    """Verbatim reference copy of the prior repo's nig_loss (models/cnn.py),
    inline wrap and all. Kept here as the parity oracle -- do not 'fix' it."""
    nu = nu.clamp(min=1e-6)
    omega = 2 * beta * (1 + nu)
    diff = y - gamma
    diff = diff - 2 * torch.round(diff / 2)
    loss_nll = 0.5 * torch.log(torch.tensor(torch.pi) / nu) - alpha * torch.log(omega) + \
               (alpha + 0.5) * torch.log(torch.square(diff) * nu + omega) + \
               torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    loss_reg = torch.abs(diff) * (2 * nu + alpha)
    return (loss_nll + lam * loss_reg).mean()


def _random_nig_params(n, seed):
    g = torch.Generator().manual_seed(seed)
    gamma = torch.rand(n, generator=g) * 2 - 1          # [-1, 1)
    nu = torch.rand(n, generator=g) * 3 + 0.01           # positive
    alpha = torch.rand(n, generator=g) * 3 + 1.01        # > 1
    beta = torch.rand(n, generator=g) * 2 + 0.01         # positive
    y = torch.rand(n, generator=g) * 2 - 1               # [-1, 1)
    return gamma, nu, alpha, beta, y


def test_nig_loss_matches_old_implementation_exactly():
    for seed in [0, 1, 2, 42]:
        gamma, nu, alpha, beta, y = _random_nig_params(64, seed)
        torch.testing.assert_close(
            nig_loss(gamma, nu, alpha, beta, y),
            _old_nig_loss(gamma, nu, alpha, beta, y),
        )


def test_nig_loss_wrap_invariance():
    # same wrapped error magnitude near the boundary and away from it must give
    # the same loss, since the loss depends on diff only via square and abs.
    nu = torch.tensor([1.0])
    alpha = torch.tensor([2.0])
    beta = torch.tensor([0.5])
    near_boundary = nig_loss(torch.tensor([0.99]), nu, alpha, beta, torch.tensor([-0.99]))
    interior = nig_loss(torch.tensor([-0.01]), nu, alpha, beta, torch.tensor([0.01]))
    torch.testing.assert_close(near_boundary, interior)


def test_nig_loss_finite_and_differentiable():
    gamma, nu, alpha, beta, y = _random_nig_params(32, 7)
    gamma, nu, alpha, beta = (t.clone().requires_grad_(True) for t in (gamma, nu, alpha, beta))
    loss = nig_loss(gamma, nu, alpha, beta, y)
    assert torch.isfinite(loss)
    loss.backward()
    for t in (gamma, nu, alpha, beta):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()


def test_nig_loss_penalises_confident_error_more_than_diffuse_error():
    # for an error LARGE relative to the predictive scale, higher evidence
    # (nu, alpha) -> higher loss: confident wrongness costs more than diffuse
    # wrongness. (The condition matters: if the error sits well inside the
    # predictive width, sharpening legitimately improves the NLL instead.)
    gamma = torch.tensor([0.0])
    beta = torch.tensor([0.05])
    y = torch.tensor([0.9])  # ~2.8 sigma for the diffuse belief, ~8 sigma for the confident one
    diffuse = nig_loss(gamma, torch.tensor([0.5]), torch.tensor([1.5]), beta, y)
    confident = nig_loss(gamma, torch.tensor([5.0]), torch.tensor([5.0]), beta, y)
    assert confident > diffuse


def test_student_t_marginal_parameters():
    gamma = torch.tensor([0.2, -0.7])
    nu = torch.tensor([1.5, 0.3])
    alpha = torch.tensor([2.0, 3.5])
    beta = torch.tensor([0.8, 1.2])
    dist = student_t_marginal(gamma, nu, alpha, beta)
    torch.testing.assert_close(dist.df, 2 * alpha)
    torch.testing.assert_close(dist.loc, gamma)
    torch.testing.assert_close(dist.scale, torch.sqrt(beta * (1 + nu) / (nu * alpha)))


def test_student_t_marginal_mode_at_gamma():
    gamma = torch.tensor([0.25])
    dist = student_t_marginal(gamma, torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([0.5]))
    at_mode = dist.log_prob(gamma)
    nearby = dist.log_prob(gamma + 0.1)
    assert at_mode > nearby


def test_cnn_outputs_satisfy_nig_constraints():
    # softplus reparameterisation must guarantee nu > 0, alpha > 1, beta > 0
    # for any input, or nig_loss's logs produce NaNs downstream.
    torch.manual_seed(0)
    model = EvidentialCNN()
    gamma, nu, alpha, beta = model(torch.randn(16, 3, 28, 28))
    assert (nu > 0).all()
    assert (alpha > 1).all()
    assert (beta > 0).all()
    assert torch.isfinite(gamma).all()


def test_cnn_output_feeds_loss_end_to_end():
    torch.manual_seed(0)
    model = EvidentialCNN()
    gamma, nu, alpha, beta = model(torch.randn(8, 3, 28, 28))
    y = torch.rand(8) * 2 - 1
    loss = nig_loss(gamma, nu, alpha, beta, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


# --- training_uncertainty ablation (2026-08) -----------------------------

def test_training_uncertainty_epistemic_matches_predictive_uncertainty_exactly():
    nu = torch.tensor([4.0, 0.5, 10.0])
    alpha = torch.tensor([3.0, 1.2, 8.0])
    beta = torch.tensor([1.0, 0.3, 0.05])
    assert torch.equal(training_uncertainty(nu, alpha, beta, measure="epistemic"),
                       predictive_uncertainty(nu, alpha, beta))


def test_training_uncertainty_aleatoric_has_no_nu_dependence():
    alpha = torch.tensor([3.0])
    beta = torch.tensor([1.0])
    low_nu = training_uncertainty(torch.tensor([0.1]), alpha, beta, measure="aleatoric")
    high_nu = training_uncertainty(torch.tensor([100.0]), alpha, beta, measure="aleatoric")
    assert torch.equal(low_nu, high_nu)
    assert low_nu.item() == 1.0 / (3.0 - 1.0)  # beta / (alpha - 1)


def test_training_uncertainty_total_equals_epistemic_plus_aleatoric():
    nu = torch.tensor([4.0])
    alpha = torch.tensor([3.0])
    beta = torch.tensor([1.0])
    epi = training_uncertainty(nu, alpha, beta, measure="epistemic")
    ale = training_uncertainty(nu, alpha, beta, measure="aleatoric")
    tot = training_uncertainty(nu, alpha, beta, measure="total")
    assert tot.item() == pytest.approx((epi + ale).item())


def test_training_uncertainty_rejects_unknown_measure():
    with pytest.raises(ValueError):
        training_uncertainty(torch.tensor([1.0]), torch.tensor([2.0]),
                             torch.tensor([1.0]), measure="bogus")
