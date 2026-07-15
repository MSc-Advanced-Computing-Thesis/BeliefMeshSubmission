"""NIG loss and distributional utilities. Spec Sec 3.

These operate on (gamma, nu, alpha, beta) tuples regardless of which network
produced them -- EvidentialCNN's fc2 emits the raw four values directly, so
there is no separate learnable head here, just the fixed math: the loss used
to train against a scalar target, and the Student-t marginal predictive that
fusion (Spec Sec 4) consumes.

Epistemic uncertainty is 1/nu, aleatoric is beta/(alpha-1). The marginal
predictive is Student-t with location gamma, df 2*alpha, scale
sqrt(beta*(1+nu)/(nu*alpha)).
"""

from __future__ import annotations

import torch

from beliefmesh.metrics.circular import circular_diff


def nig_loss(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
    lam: float = 0.1,
) -> torch.Tensor:
    """Evidential regression loss: Student-t NLL plus evidence regularisation.

    target is the scalar angle (normalised units, [-1, 1)) - ground truth in
    Stage 0, a fused-mode pseudo-label everywhere else (Spec Sec 5). lam scales
    the evidence regulariser, which penalises confidence (large nu, alpha) in
    proportion to the wrapped error magnitude.
    """
    nu = nu.clamp(min=1e-6)
    omega = 2 * beta * (1 + nu)

    diff = circular_diff(target, gamma)

    loss_nll = 0.5 * torch.log(torch.tensor(torch.pi) / nu) - alpha * torch.log(omega) + \
               (alpha + 0.5) * torch.log(torch.square(diff) * nu + omega) + \
               torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    loss_reg = torch.abs(diff) * (2 * nu + alpha)

    return (loss_nll + lam * loss_reg).mean()


def student_t_marginal(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.distributions.StudentT:
    """Marginal predictive over the target: Student-t with location gamma,
    df 2*alpha, scale sqrt(beta*(1+nu)/(nu*alpha)) (Spec Sec 3).

    Note: StudentT.log_prob does NOT wrap on the circle - it uses a flat
    (y - loc). Fusion (Spec Sec 4.3) must evaluate log-densities at wrapped
    offsets, i.e. at gamma + circular_diff(y, gamma), so that beliefs at +179
    and -179 degrees produce one shared mode rather than two disjoint ones.
    """
    scale = torch.sqrt(beta * (1 + nu) / (nu * alpha))
    return torch.distributions.StudentT(df=2 * alpha, loc=gamma, scale=scale)
