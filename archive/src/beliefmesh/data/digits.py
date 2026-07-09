"""Digit-7 rotation regression dataset. Spec Sec 2: fixed train/held-out split under seed 42, 20% held out."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class RotatedDigitDataset(Dataset):
    """Rotated digit-7 images with continuous angle labels in [-180, 180)."""

    def __init__(self, seed: int, holdout_fraction: float, split: str):
        """split: 'train' or 'holdout'. Split must be seed-deterministic per Spec Sec 2."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (image, angle_label)."""
        raise NotImplementedError
