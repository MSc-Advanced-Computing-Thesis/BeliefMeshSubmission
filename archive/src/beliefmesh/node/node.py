"""Single node: wraps one trainable EvidentialCNN. Spec Sec 3, 5, 6.3.

No frozen/trainable split for this experiment suite -- every node's model is
fully trainable throughout (see evidential_cnn.py's module docstring).

predict() must return the full (gamma, nu, alpha, beta) tuple -- never a
collapsed (pred, uncertainty, certainty) point estimate. Collapsing before
fusion is Defect 1 from the prior implementation; any point-estimate summary
(evidential.predictive_uncertainty) is display/eval-only and never on the
path into fuse().
"""

from __future__ import annotations

from pathlib import Path

import torch

from beliefmesh.models.evidential import nig_loss
from beliefmesh.models.evidential_cnn import EvidentialCNN


class Node:
    def __init__(self, lr: float, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EvidentialCNN().to(self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

    @classmethod
    def from_checkpoint(cls, path: str | Path, lr: float,
                        device: torch.device | None = None) -> "Node":
        node = cls(lr=lr, device=device)
        node.model.load_state_dict(torch.load(path, map_location=node.device))
        return node

    def train_step(self, images: torch.Tensor, targets: torch.Tensor) -> float:
        """One gradient step of the ordinary NIG loss against scalar targets.

        targets may be ground-truth angles or pseudo-labels (a peer's gamma, or
        later a fused mode) -- Spec Sec 5: the receiving node always trains with
        the ordinary NIG loss on a scalar and derives its own uncertainty
        locally. No external weight ever multiplies this loss (Spec Sec 6.1).
        """
        images = images.to(self.device)
        targets = targets.to(self.device)
        self.optimiser.zero_grad()
        gamma, nu, alpha, beta = self.model(images)
        loss = nig_loss(gamma, nu, alpha, beta, targets)
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full NIG belief (gamma, nu, alpha, beta), uncollapsed, no grad."""
        images = images.to(self.device)
        with torch.no_grad():
            return self.model(images)
