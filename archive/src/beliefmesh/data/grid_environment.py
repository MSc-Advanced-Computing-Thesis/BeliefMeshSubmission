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
                 mnist_root: str = "data/mnist", rotation_seed: int = 42,
                 offset_field: np.ndarray | None = None,
                 excluded_rotation_ranges: list[tuple[float, float]] | None = None):
        """offset_field: optional per-cell label offsets in DEGREES (CHECKLIST
        Phase 13, mapping-conflict experiments). Shape (H, W) for a static
        field, or (T, H, W) for a time-varying one (e.g. a hard-edged block
        appearing mid-run -- an obstruction suddenly changing the local
        mapping). The image shows rotation theta; the true label becomes
        theta + offset(cell[, step]). Identical visual inputs then demand
        different answers at different locations -- a location-dependent
        mapping that no shared global function can fit.

        excluded_rotation_ranges: optional list of (lo, hi) degree ranges to
        exclude from rotation sampling (both training via
        get_multiple_rotations AND evaluation via cell_rotations). Diagnosed
        empirically (analyze_excursion_frames.py): this specific base digit
        becomes visually self-ambiguous at ~120-150 deg and near the +-180
        deg wrap boundary, producing large, confident errors independent of
        the fusion/training mechanism. A disclosed, principled exclusion of
        a known blind spot -- not tuned per-run, applied consistently to
        every mode/arm that uses this environment instance."""
        self.grid_size = grid_size
        self.environment_grids = environment_grids
        self.offset_field = offset_field
        self.excluded_rotation_ranges = excluded_rotation_ranges

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
        self._rotation_rng = rng  # reused below for excluded-range rejection sampling
        shape = (len(environment_grids), grid_size, grid_size)
        if excluded_rotation_ranges:
            self.cell_rotations = np.array(
                [self._sample_rotation(rng) for _ in range(int(np.prod(shape)))]
            ).reshape(shape)
        else:
            self.cell_rotations = rng.uniform(-180, 180, size=shape)

        self._cache_step: int | None = None
        self._cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _sample_rotation(self, rng: np.random.Generator) -> float:
        """Uniform(-180, 180), rejecting excluded_rotation_ranges if set."""
        if not self.excluded_rotation_ranges:
            return float(rng.uniform(-180, 180))
        while True:
            angle = float(rng.uniform(-180, 180))
            if not any(lo <= angle <= hi for lo, hi in self.excluded_rotation_ranges):
                return angle

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
            self._cache[key] = (image, self._label(angle, row, col, step))
        return self._cache[key]

    def _label(self, angle_deg: float, row: int, col: int, step: int) -> torch.Tensor:
        """Normalised label, including the cell's mapping offset if configured.
        Wraps on the circle (same period-2 convention as circular_diff)."""
        total = angle_deg / 180.0
        if self.offset_field is not None:
            field = (self.offset_field[step] if self.offset_field.ndim == 3
                     else self.offset_field)
            total = total + field[row, col] / 180.0
        total = ((total + 1.0) % 2.0) - 1.0
        return torch.tensor(total, dtype=torch.float32)

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
            angle = self._sample_rotation(rng)
            image = TF.rotate(self.base_image, angle, fill=1.0)
            images.append(apply_filter_from_env_value(image, env_value))
            targets.append(self._label(angle, row, col, step))
        return images, targets
