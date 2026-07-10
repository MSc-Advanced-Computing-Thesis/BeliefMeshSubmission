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


def nig_loss(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Evidential regression loss (NLL + evidence regularisation) against a scalar target angle."""
    raise NotImplementedError


def student_t_marginal(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
    """Returns a torch.distributions.StudentT (or equivalent) marginal predictive built from NIG parameters."""
    raise NotImplementedError
