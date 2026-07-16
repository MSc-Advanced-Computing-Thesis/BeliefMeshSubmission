"""Spatial grid environment for Stage 6. Spec Sec 7.

A single digit-7 image serves as the visual input for every cell (Spec Sec 2
decision: Stage 6 retains the single-digit-per-cell design). At each timestep
a cell's image is the digit rotated by that cell's fixed pre-drawn angle, with
the cell's current environment value applied as a colour filter.

Rotation angles are precomputed for every (step, row, col) under seed 42, so
cell inputs are deterministic -- which also allows per-step render caching:
every node covering a cell sees the identical image, so it is rendered once.

env_value semantics (ported from the prior repo's apply_filter_from_env_value):
  [0, 1]:   blue (0) -> purple (0.5) -> red (1)
  [1, 1.5]: red -> white
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

from beliefmesh.data.filters import make_green_digits


def apply_filter_from_env_value(image: torch.Tensor, env_value: float) -> torch.Tensor:
    """Channel multipliers as a continuous function of the environment value."""
    v = max(0.0, float(env_value))
    if v <= 1.0:
        r_w = 0.8 * v + 0.2
        g_w = 0.2
        b_w = 1.0 - 0.8 * v
    else:
        t = min((v - 1.0) / 0.5, 1.0)
        r_w = 1.0
        g_w = 0.2 + 0.8 * t
        b_w = 0.2 + 0.8 * t

    if image.dim() == 3:
        filtered = torch.stack([image[0] * r_w, image[1] * g_w, image[2] * b_w])
    else:
        filtered = torch.stack(
            [image[:, 0] * r_w, image[:, 1] * g_w, image[:, 2] * b_w], dim=1)
    return filtered.clamp(0.0, 1.0)


class GridEnvironment:
    """Renders per-cell inputs for a (T, H, W) environment-value tensor."""

    def __init__(self, grid_size: int, environment_grids: np.ndarray,
                 mnist_root: str = "data/mnist", rotation_seed: int = 42):
        self.grid_size = grid_size
        self.environment_grids = environment_grids

        base = datasets.MNIST(root=mnist_root, train=True, download=True)
        digit7 = base.data[base.targets == 7][0]  # the single fixed seven
        to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        # invert + green recolour once; rotation/filter applied per cell
        self.base_image = make_green_digits(TF.invert(to_tensor(digit7)))

        rng = np.random.default_rng(rotation_seed)
        self.cell_rotations = rng.uniform(
            -180, 180, size=(len(environment_grids), grid_size, grid_size))

        self._cache_step: int | None = None
        self._cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_cell_input(self, row: int, col: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(image, normalised_angle_label) for one cell at one timestep. Cached
        per step: deterministic given (step, row, col), and every covering node
        sees the identical rendering."""
        if step != self._cache_step:
            self._cache_step = step
            self._cache = {}
        key = (row, col)
        if key not in self._cache:
            angle = self.cell_rotations[step, row, col]
            env_value = self.environment_grids[step, row, col]
            image = TF.rotate(self.base_image, float(angle), fill=1.0)
            image = apply_filter_from_env_value(image, float(env_value))
            self._cache[key] = (image, torch.tensor(angle / 180.0, dtype=torch.float32))
        return self._cache[key]

    def get_batch_for_cells(self, cells, step: int):
        """(images, targets, keys) lists for a list of (row, col) cells."""
        images, targets, keys = [], [], []
        for row, col in cells:
            img, target = self.get_cell_input(row, col, step)
            images.append(img)
            targets.append(target)
            keys.append((row, col))
        return images, targets, keys

    def get_multiple_rotations(self, cell, step: int, n: int, rng: np.random.Generator):
        """n freshly-rotated samples of the digit under the cell's current
        filter -- used for hop-0 wearable ground-truth training."""
        row, col = cell
        env_value = self.environment_grids[step, row, col]
        images, targets = [], []
        for _ in range(n):
            angle = float(rng.uniform(-180, 180))
            image = TF.rotate(self.base_image, angle, fill=1.0)
            images.append(apply_filter_from_env_value(image, env_value))
            targets.append(torch.tensor(angle / 180.0, dtype=torch.float32))
        return images, targets
