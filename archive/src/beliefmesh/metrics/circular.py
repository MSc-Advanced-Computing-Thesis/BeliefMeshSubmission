"""Circular angle arithmetic. Spec Sec 2 & 4.3: the ONE place wraparound is defined.

Imported by the training loss, the dataset's angle labels, and the fusion grid
(Spec Sec 4.3: fusion must wrap using "the same wrapping as the circular MSE").
Do not reimplement wraparound anywhere else.
"""

from __future__ import annotations

import torch


PERIOD = 360.0


def circular_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed angular difference a - b, wrapped to [-180, 180)."""
    raw = a - b
    return raw - PERIOD * torch.round(raw / PERIOD)


def circular_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared circular_diff(pred, target)."""
    avg_sq_diffs = torch.mean(torch.square(circular_diff(pred, target)))
    return avg_sq_diffs
