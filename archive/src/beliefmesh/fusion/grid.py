"""Shared circular candidate grid. Spec Sec 4.1: G uniformly spaced candidates
over the circular angle domain -- G=360 gives one candidate per degree.

Unlike the prior repo's torch.linspace(-1, 1, 200), this grid does NOT
duplicate the endpoint: -1 and +1 are the same physical angle, so the grid
covers [-1, 1) with spacing PERIOD/G. Grid error is quantisation error,
bounded by half the spacing (0.5 degrees at G=360).
"""

from __future__ import annotations

import torch

from beliefmesh.metrics.circular import PERIOD


def circular_grid(g: int, device: torch.device | None = None) -> torch.Tensor:
    """g candidate angles uniformly spaced over [-1, 1), normalised units."""
    return -PERIOD / 2 + PERIOD * torch.arange(g, device=device, dtype=torch.float32) / g
