"""Circular angle arithmetic. Spec Sec 2 & 4.3: the ONE place wraparound is defined.

Imported by the training loss, the dataset's angle labels, and the fusion grid
(Spec Sec 4.3: fusion must wrap using "the same wrapping as the circular MSE").
Do not reimplement wraparound anywhere else.

Angles are represented normalised, angle_normalised = angle_degrees / 180, so a
full rotation has period 2 and the domain is [-1, 1). This matches the prior
experiment's convention exactly (same scale the old lr/lam hyperparameters
were tuned against) -- switching to raw degrees here would rescale nig_loss's
squared-error term by ~180^2 for no benefit. Convert to degrees only at
display/reporting boundaries (e.g. angle_normalised * 180), never internally.
"""

from __future__ import annotations

import torch


PERIOD = 2.0


def circular_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed angular difference a - b, wrapped to [-1, 1)."""
    raw = a - b
    return raw - PERIOD * torch.round(raw / PERIOD)


def circular_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared circular_diff(pred, target)."""
    avg_sq_diffs = torch.mean(torch.square(circular_diff(pred, target)))
    return avg_sq_diffs
