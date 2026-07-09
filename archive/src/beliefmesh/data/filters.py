"""Red/blue colour filters. Spec Sec 2: digit recoloured green pre-filter so red and blue shift symmetrically."""

from __future__ import annotations

import torch


def apply_colour_filter(image: torch.Tensor, filter_colour: str, strength: float) -> torch.Tensor:
    """filter_colour: 'red' or 'blue'. strength in [0, 1]. Image is background/digit-inverted, digit recoloured green, before this is applied."""
    raise NotImplementedError
