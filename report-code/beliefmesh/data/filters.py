"""Colour pipeline: green digit recolouring and red/blue filters. Spec Sec 2.

The digit is recoloured green before filter application because green shares no
dominant channel with either red or blue -- both filters suppress the digit
colour equally, so red and blue filters of equal strength produce symmetric
distributional shifts with no directional bias.

Ported from the prior experiment's data/dataset.py (apply_filter,
make_green_digits) unchanged.
"""

from __future__ import annotations

import torch

_FILTER_MULTIPLIERS = {
    "red": (1.0, 0.2, 0.2),
    "blue": (0.2, 0.2, 1.0),
    "purple": (1.0, 0.2, 1.0),
}


def apply_colour_filter(image: torch.Tensor, filter_type: str, strength: float = 1.0) -> torch.Tensor:
    """Apply a colour filter to a (3, H, W) image or (N, 3, H, W) batch.

    strength in [0, 1] interpolates each channel multiplier between 1.0 (no
    effect) and the filter's full multiplier. filter_type 'neutral' is a no-op.
    """
    if filter_type == "neutral":
        return image

    try:
        multipliers = _FILTER_MULTIPLIERS[filter_type]
    except KeyError:
        raise ValueError(f"Unknown filter type: {filter_type}")

    if image.dim() == 3:
        filtered = torch.stack([
            image[0] * (1 - strength + strength * multipliers[0]),
            image[1] * (1 - strength + strength * multipliers[1]),
            image[2] * (1 - strength + strength * multipliers[2]),
        ])
    else:
        filtered = torch.stack([
            image[:, 0] * (1 - strength + strength * multipliers[0]),
            image[:, 1] * (1 - strength + strength * multipliers[1]),
            image[:, 2] * (1 - strength + strength * multipliers[2]),
        ], dim=1)

    return filtered.clamp(0.0, 1.0)


def make_green_digits(image: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Recolour digit pixels to green in a 3-channel INVERTED image (dark digit
    on white background). Pixels whose channel mean is below threshold are
    treated as digit pixels.
    """
    digit_mask = image.mean(dim=0) < threshold  # (H, W) bool
    result = image.clone()
    result[0][digit_mask] = 0.0   # R
    result[1][digit_mask] = 0.8   # G
    result[2][digit_mask] = 0.0   # B
    return result
