"""Backbone CNN. Spec Sec 3: three conv blocks + two FC layers, frozen after Stage 0 pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """Three convolutional blocks followed by two fully connected layers, producing a feature vector for the NIG head."""

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
