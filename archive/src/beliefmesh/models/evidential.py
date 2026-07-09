"""Normal-Inverse-Gamma evidential regression head. Spec Sec 3.

Head outputs (gamma, nu, alpha, beta) in a single forward pass. Epistemic
uncertainty is 1/nu, aleatoric is beta/(alpha-1). The marginal predictive is
Student-t with location gamma, df 2*alpha, scale sqrt(beta*(1+nu)/(nu*alpha)).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NIGHead(nn.Module):
    """Trainable head mapping backbone features to (gamma, nu, alpha, beta)."""

    def __init__(self, in_features: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (gamma, nu, alpha, beta), each shape (batch,)."""
        raise NotImplementedError


def nig_loss(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Evidential regression loss (NLL + evidence regularisation) against a scalar target angle."""
    raise NotImplementedError


def student_t_marginal(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
    """Returns a torch.distributions.StudentT (or equivalent) marginal predictive built from NIG parameters."""
    raise NotImplementedError
