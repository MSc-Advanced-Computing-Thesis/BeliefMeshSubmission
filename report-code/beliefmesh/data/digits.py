"""Digit-7 rotation regression dataset. Spec Sec 2.

MNIST restricted to digit seven (not rotationally symmetric). Fixed train /
held-out split under seed 42 with 20% reserved for evaluation; all evaluation
metrics are computed exclusively on the held-out subset.

Per-item pipeline (ported from the prior experiment's data/dataset.py):
grayscale -> 3 channels, invert (dark digit on white), recolour digit green,
rotate by a freshly sampled uniform angle with white fill, apply colour
filter. Label is the rotation angle in normalised units, angle_degrees / 180,
so it lives in [-1, 1) -- the same space circular_diff/nig_loss operate in.

Note: the rotation angle is re-sampled on every __getitem__ call, so two
passes over the same indices see different angles. This matches the prior
experiment exactly; seed the global `random` module at run start for
reproducibility across runs.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from beliefmesh.data.filters import apply_colour_filter, make_green_digits
from pathlib import Path

# MNIST is downloaded on first use into <repository root>/data/mnist.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "mnist"


def get_digit7_splits(
    root: str | Path = DATA_ROOT,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (train_indices, eval_indices) as positions within the digit-7
    subset of the MNIST training set. Deterministic given the seed: the first
    eval_fraction of a seeded permutation is the held-out set.
    """
    base = datasets.MNIST(root=root, train=True, download=True)
    n_total = int((base.targets == 7).sum().item())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_eval = int(n_total * eval_fraction)
    return perm[n_eval:], perm[:n_eval]


class RotatedDigitDataset(Dataset):
    """Rotated (optionally colour-filtered) digit-7 images with normalised angle labels."""

    def __init__(
        self,
        root: str | Path = DATA_ROOT,
        train: bool = True,
        filter_type: str = "neutral",
        filter_strength: float = 1.0,
        angle_range: tuple[float, float] = (-180.0, 180.0),
        download: bool = True,
        subset_indices: np.ndarray | None = None,
    ):
        self.filter_type = filter_type
        self.filter_strength = filter_strength
        self.angle_range = angle_range

        base = datasets.MNIST(root=root, train=train, download=download)
        all_images = base.data[base.targets == 7]  # (N, 28, 28) uint8
        if subset_indices is not None:
            self.images = all_images[np.asarray(subset_indices)]
        else:
            self.images = all_images

        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.to_tensor(self.images[idx])  # white digit on black

        # invert: dark digit on white background
        image = TF.invert(image)

        # recolour digit green before rotation (Spec Sec 2)
        image = make_green_digits(image)

        # rotate with white fill
        angle = random.uniform(*self.angle_range)
        image = TF.rotate(image, angle, fill=1.0)

        # colour filter over everything
        image = apply_colour_filter(image, self.filter_type, self.filter_strength)

        angle_normalised = angle / 180.0
        return image, torch.tensor(angle_normalised, dtype=torch.float32)
