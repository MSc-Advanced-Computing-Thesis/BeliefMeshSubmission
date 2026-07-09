"""Single node: frozen backbone + trainable NIG head. Spec Sec 3, 5, 6.3.

predict() must return the full (gamma, nu, alpha, beta) tuple -- never a
collapsed (pred, uncertainty, certainty) point estimate. Collapsing before
fusion is Defect 1 from the prior implementation; any point-estimate summary
belongs in a separate, clearly-named display-only helper, never on the path
into fuse().
"""

from __future__ import annotations

import torch

from beliefmesh.models.backbone import Backbone
from beliefmesh.models.evidential import NIGHead


class Node:
    def __init__(self, backbone: Backbone, head: NIGHead):
        raise NotImplementedError

    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (gamma, nu, alpha, beta) -- full NIG belief, uncollapsed."""
        raise NotImplementedError

    def train_on_pseudo_label(self, image: torch.Tensor, pseudo_label: float) -> float:
        """Ordinary NIG loss against a scalar pseudo-label (fused mode or ground truth). Spec Sec 5: no external weight multiplies this loss -- consensus tempers fusion inputs, not this training step (Spec Sec 6.1)."""
        raise NotImplementedError
