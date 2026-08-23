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

_TO_TENSOR = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def load_digit_instances(indices, mnist_root: str = "data/mnist") -> list[torch.Tensor]:
    """Processed (invert + green-recolour) digit-seven images at the given
    positions within the MNIST digit-7 subset. Shared helper (2026-08) so
    GridEnvironment's per_cell_digits path and Mesh's per-node-instance path
    use the identical preprocessing pipeline."""
    base = datasets.MNIST(root=mnist_root, train=True, download=True)
    sevens = base.data[base.targets == 7]
    return [make_green_digits(TF.invert(_TO_TENSOR(sevens[i]))) for i in indices]


def sample_digit_instances(n: int, mnist_root: str = "data/mnist", seed: int = 42) -> list[torch.Tensor]:
    """n distinct instances drawn without replacement from the TRAINING
    partition (beliefmesh.data.digits.get_digit7_splits, same fixed
    seed=42/20%-holdout convention as pretraining -- no held-out image is
    ever used here)."""
    from beliefmesh.data.digits import get_digit7_splits
    train_idx, _ = get_digit7_splits(root=mnist_root)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(train_idx, size=n, replace=(len(train_idx) < n))
    return load_digit_instances(chosen, mnist_root)


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
                 excluded_rotation_ranges: list[tuple[float, float]] | None = None,
                 per_cell_digits: bool = False, digit_seed: int = 42,
                 noise_field: np.ndarray | None = None, noise_seed: int = 42,
                 apply_colour_filter: bool = True):
        """apply_colour_filter (2026-08): if False, apply_filter_from_env_value
        is never called -- every cell renders as the plain white-background,
        green-digit image (identical to Stage 0 pretraining's
        RotatedDigitDataset(filter_type="neutral") pipeline), regardless of
        `environment_grids`' values. Default True preserves the exact prior
        behaviour of every experiment this session -- Stage 6's rendering
        path had NO genuine no-filter option before this: unlike
        RotatedDigitDataset's filter_type="neutral" no-op, GridEnvironment
        unconditionally called apply_filter_from_env_value, so every
        "offset world" script that passed a uniform 0.5 colour grid (meant
        to signal "no colour variation") was in fact rendering a constant
        muted purple background (r_w=0.6, g_w=0.2, b_w=0.6 at v=0.5) --
        visually different from the plain-white images the pretrained
        checkpoint was actually trained on. This flag is the genuine fix,
        not a cosmetic one.

        noise_field (2026-08, spatially-varying-noise evidence-asymmetry
        diagnostic): optional (H, W) array of per-cell Gaussian jitter STD
        (normalised units, same scale as the label domain), applied ONLY to
        the direct ground-truth TRAINING label returned by
        get_multiple_rotations() -- the label get_cell_input() returns
        (consumed by evaluation and by propagation's image rendering) is
        completely unaffected, so reported MSE is always measured against
        the true angle, never the jittered training signal. None (default)
        reproduces every prior experiment's behaviour exactly (no jitter).

        per_cell_digits (2026-08, evidence-asymmetry diagnostic): if True,
        each cell is assigned its OWN digit-seven instance -- drawn without
        replacement from the training partition (beliefmesh.data.digits.
        get_digit7_splits, same fixed seed=42/20%-holdout convention used
        everywhere else in this codebase, so no held-out image is ever used
        here) -- fixed for the whole run, instead of every cell sharing the
        single fixed instance the default (False) behaviour uses. The
        assignment is stored on THIS GridEnvironment object, which every
        node/the whole mesh shares a single reference to, so every node
        querying a given cell sees the identical instance -- fusion's
        "common referent" requirement is satisfied by construction, not by
        any extra synchronisation. Default False reproduces every prior
        experiment's behaviour exactly (single shared image, no change).

        offset_field: optional per-cell label offsets in DEGREES (CHECKLIST
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
        self.per_cell_digits = per_cell_digits
        self.noise_field = noise_field
        self._noise_rng = np.random.default_rng(noise_seed)
        self.apply_colour_filter = apply_colour_filter

        base = datasets.MNIST(root=mnist_root, train=True, download=True)
        to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        # invert + green recolour once per instance; rotation/filter applied per cell
        if per_cell_digits:
            from beliefmesh.data.digits import get_digit7_splits
            train_idx, _ = get_digit7_splits(root=mnist_root)  # default seed=42, 20% holdout
            sevens = base.data[base.targets == 7]
            n_cells = grid_size * grid_size
            digit_rng = np.random.default_rng(digit_seed)
            chosen = digit_rng.choice(train_idx, size=n_cells,
                                      replace=(len(train_idx) < n_cells))
            self.cell_base_images: dict[tuple[int, int], torch.Tensor] = {}
            i = 0
            for row in range(grid_size):
                for col in range(grid_size):
                    self.cell_base_images[(row, col)] = make_green_digits(
                        TF.invert(to_tensor(sevens[chosen[i]])))
                    i += 1
            self.base_image = None
        else:
            digit7 = base.data[base.targets == 7][0]  # the single fixed seven
            self.base_image = make_green_digits(TF.invert(to_tensor(digit7)))
            self.cell_base_images = None

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

    def render_with_image(self, base_image: torch.Tensor, row: int, col: int, step: int) -> torch.Tensor:
        """Render a cell's current (rotation, colour-filter) state using an
        EXTERNAL base image instead of this environment's own shared/per-cell
        instance -- 2026-08 per-node-instance evidence-asymmetry diagnostic.
        Same rotate-then-filter pipeline as get_cell_input(), just with the
        base image supplied by the caller. No caching (the whole point is
        that different callers pass different images for the same cell)."""
        angle = self.cell_rotations[step, row, col]
        image = TF.rotate(base_image, float(angle), fill=1.0)
        if not self.apply_colour_filter:
            return image
        env_value = self.environment_grids[step, row, col]
        return apply_filter_from_env_value(image, float(env_value))

    def get_batch_for_cells_with_image(self, base_image: torch.Tensor, cells, step: int):
        """get_batch_for_cells(), but every cell is rendered from the given
        EXTERNAL base image rather than this environment's own instance --
        used for a node's own training-view rendering when it has its own
        fixed digit instance (per-node-instance construction)."""
        images, targets, keys = [], [], []
        for row, col in cells:
            images.append(self.render_with_image(base_image, row, col, step))
            angle = self.cell_rotations[step, row, col]
            targets.append(self._label(angle, row, col, step))
            keys.append((row, col))
        return images, targets, keys

    def get_multiple_rotations_with_image(self, base_image: torch.Tensor, cell, step: int,
                                          n: int, rng: np.random.Generator):
        """get_multiple_rotations(), but every sample is rendered from the
        given EXTERNAL base image -- used for hop-0 ground-truth training
        when the training node has its own fixed digit instance."""
        row, col = cell
        env_value = self.environment_grids[step, row, col]
        images, targets = [], []
        for _ in range(n):
            angle = self._sample_rotation(rng)
            image = TF.rotate(base_image, angle, fill=1.0)
            if self.apply_colour_filter:
                image = apply_filter_from_env_value(image, env_value)
            images.append(image)
            targets.append(self._label(angle, row, col, step))
        return images, targets

    def _base_image_for(self, row: int, col: int) -> torch.Tensor:
        """The digit instance a given cell renders from -- the single shared
        image (default), or this cell's own assigned instance if
        per_cell_digits=True. Every node querying this cell gets the same
        object back, always."""
        return self.cell_base_images[(row, col)] if self.per_cell_digits else self.base_image

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
            image = TF.rotate(self._base_image_for(row, col), float(angle), fill=1.0)
            if self.apply_colour_filter:
                env_value = self.environment_grids[step, row, col]
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
        base_image = self._base_image_for(row, col)
        images, targets = [], []
        noise_std = float(self.noise_field[row, col]) if self.noise_field is not None else 0.0
        for _ in range(n):
            angle = self._sample_rotation(rng)
            image = TF.rotate(base_image, angle, fill=1.0)
            if self.apply_colour_filter:
                image = apply_filter_from_env_value(image, env_value)
            images.append(image)
            label = self._label(angle, row, col, step)
            if noise_std > 0:
                jitter = float(self._noise_rng.normal(0.0, noise_std))
                label = torch.tensor(((label.item() + jitter + 1.0) % 2.0) - 1.0, dtype=torch.float32)
            targets.append(label)
        return images, targets
