"""Shared circular candidate grid. Spec Sec 4.1: G=360 uniformly spaced candidates over the angle domain."""

from __future__ import annotations

import torch


def circular_grid(g: int) -> torch.Tensor:
    """Returns g candidate angles uniformly spaced over [-180, 180)."""
    raise NotImplementedError
