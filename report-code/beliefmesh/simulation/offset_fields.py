"""Offset-world field construction and wearable path generation.

The offset world (report Section 4.2.2) adds a spatially and temporally varying
label offset to the rotation task. build_dynamic_offset_field() is the field
used throughout Chapter 5 (the reference environment, "field0"); the four
alternative environments of Section 5.4 and Appendix D are built by
field_variants.build_field(), which generalises the same construction.

build_random_wander_path() is the unguided wearable policy of Section 4.3.
analytic_floors() / dynamic_analytic_floors() give the global-model and
per-node lower bounds implied by a field's spatial variance.
"""

from __future__ import annotations

import numpy as np

from beliefmesh.node.mesh import fov_cells

OFFSET_MAX_DEG = 60.0   # linear gradient across columns: -60 .. +60 degrees
BASE_ERROR = 0.0123     # stage 0 baseline holdout MSE (clean-task error floor)


EVENT_STEP = 195          # midpoint: the "house" event fires here
EVENT_OFFSET_DEG = 45.0   # hard-edged additional offset inside the block
EVENT_BLOCK = (slice(5, 11), slice(5, 11))  # rows 5-10, cols 5-10


def build_offset_field(grid_size: int, total_steps: int, event: bool = True,
                       event_step: int | None = None) -> np.ndarray:
    """(T, H, W): smooth column gradient throughout, plus a HARD-EDGED block
    of extra offset appearing at EVENT_STEP -- an obstruction (e.g. collapsed
    structure) abruptly changing the local mapping mid-deployment. Tests
    adaptation to sudden discontinuous change, not just gentle gradients."""
    cols = np.linspace(-OFFSET_MAX_DEG, OFFSET_MAX_DEG, grid_size)
    base = np.tile(cols, (grid_size, 1))                    # (H, W)
    field = np.tile(base, (total_steps, 1, 1))              # (T, H, W)
    if event:
        start = EVENT_STEP if event_step is None else event_step
        field[start:, EVENT_BLOCK[0], EVENT_BLOCK[1]] += EVENT_OFFSET_DEG
    return field


def build_dynamic_offset_field(grid_size: int, total_steps: int,
                               n_keyframes: int = 4,
                               background_max_deg: float = 60.0,
                               block_pos: tuple[float, float] | None = None,
                               block_radius: float | None = None,
                               wake_length: float | None = None,
                               wake_half_angle_deg: float = 22.0,
                               wake_amplitude: float = 50.0,
                               wake_period: int = 110,
                               control_margin: float = 4.0,
                               control_spacing: float = 13.0,
                               seed: int = 7) -> np.ndarray:
    """(T, H, W) offset field with two genuinely dynamic components:

    BACKGROUND -- built the SAME way as the project's dynamic colour
    environments (environment/V5/generate_environment.py): a handful of
    control points carry randomised keyframe values, linearly interpolated
    over time, RBF-interpolated across the grid at each step. Smooth,
    organic, continuously evolving large-scale drift -- not a synthetic
    linear ramp.

    WAKE (the debris shadow) -- a triangular wedge of turbulent extra offset
    extending downstream of a small FIXED blocking object (e.g. a house).
    The downstream direction sweeps slowly over the run, so the wedge
    visibly moves around the fixed object as flow direction changes -- a
    real wind/water shadow, not a static patch. Inside the wedge the extra
    offset oscillates fast (turbulence downstream of an obstruction is
    chaotic, not a scaled copy of the ambient flow); outside it, only the
    smooth background applies.

    SCALING (added for the 10x10-node environment test): block_pos,
    block_radius and wake_length default to None and, when not given
    explicitly, scale proportionally to grid_size relative to the original
    22-cell design (14.0, 8.0)/1.5/10.0 -- so a bigger grid gets a
    proportionally-sized wake, not a tiny fixed-size one lost in more empty
    space. control_margin/control_spacing default to the values that
    reproduce the original 4-corner control-point layout EXACTLY at
    grid_size=22 (verified bit-for-bit), but scale the NUMBER of control
    points up for bigger grids at roughly the same spacing -- more
    independent spatial structure per unit area, not the same 4 points
    stretched thinner (which would make the field smoother, not equally
    dynamic, as the grid grows -- the opposite of what "add more new
    environment at the same scale" requires).
    """
    from scipy.interpolate import RBFInterpolator

    REFERENCE_GRID = 22.0
    scale = grid_size / REFERENCE_GRID
    if block_pos is None:
        block_pos = (14.0 * scale, 8.0 * scale)
    if block_radius is None:
        block_radius = 1.5 * scale
    if wake_length is None:
        wake_length = 10.0 * scale

    rng = np.random.default_rng(seed)
    n_per_axis = max(2, round((grid_size - 2 * control_margin) / control_spacing) + 1)
    axis_positions = np.linspace(control_margin, grid_size - 1 - control_margin, n_per_axis)
    control_positions = np.array([[r, c] for r in axis_positions for c in axis_positions],
                                 dtype=float)
    keyframe_values = rng.uniform(-background_max_deg, background_max_deg,
                                  size=(n_keyframes, len(control_positions)))
    # i.i.d. draws per keyframe don't guarantee consistent spatial contrast --
    # one keyframe can land with all 4 control points close together by pure
    # chance, flattening the field for the whole segment it governs (this bit
    # us: the last keyframe's spread was 13.6 vs 24.8-43.6 for the other
    # three, silently killing spatial dynamism for the entire final third of
    # the run). Normalise every keyframe to the same spread so no segment can
    # randomly go quiet, while keeping each keyframe's own mean level and the
    # random sign/pattern across control points.
    target_spread = 0.45 * background_max_deg
    row_mean = keyframe_values.mean(axis=1, keepdims=True)
    row_std = np.maximum(keyframe_values.std(axis=1, keepdims=True), 1e-6)
    keyframe_values = row_mean + (keyframe_values - row_mean) / row_std * target_spread
    keyframe_values = np.clip(keyframe_values, -background_max_deg, background_max_deg)
    steps_between = total_steps / (n_keyframes - 1)
    grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    field = np.zeros((total_steps, grid_size, grid_size))
    for step in range(total_steps):
        frame_float = step / steps_between
        k0 = int(frame_float)
        k1 = min(k0 + 1, n_keyframes - 1)
        alpha = frame_float - k0
        vals = (1 - alpha) * keyframe_values[k0] + alpha * keyframe_values[k1]
        rbf = RBFInterpolator(control_positions, vals, kernel="thin_plate_spline", smoothing=1.0)
        field[step] = np.clip(rbf(grid_points).reshape(grid_size, grid_size),
                              -background_max_deg, background_max_deg)

    r_idx, c_idx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    dr, dc = r_idx - block_pos[0], c_idx - block_pos[1]
    dist = np.sqrt(dr ** 2 + dc ** 2)
    angle_to_cell = np.arctan2(dr, dc)

    t = np.arange(total_steps)
    # slow single sweep (~0.4 cycles over the run), not a rapid back-and-forth
    theta = np.deg2rad(70) * np.sin(2 * np.pi * t / (total_steps * 2.5))
    half_angle = np.deg2rad(wake_half_angle_deg)

    for step in range(total_steps):
        ang_diff = np.abs(((angle_to_cell - theta[step]) + np.pi) % (2 * np.pi) - np.pi)
        in_wake = (dist >= block_radius) & (dist <= block_radius + wake_length) & (ang_diff <= half_angle)
        turbulence = wake_amplitude * np.sin(2 * np.pi * step / wake_period)
        field[step][in_wake] += turbulence
    return field


def build_chaotic_offset_field(grid_size: int, total_steps: int,
                               n_keyframes: int = 30,
                               background_max_deg: float = 100.0,
                               control_margin: float = 4.0,
                               control_spacing: float = 13.0,
                               rbf_smoothing: float = 0.1,
                               seed: int = 7) -> np.ndarray:
    """Deliberately harder variant of build_dynamic_offset_field: measured on
    the original field, adjacent cells differed by ~2-3 deg on average and a
    fixed cell drifted ~0.06 deg/step -- smooth enough that a single
    continuously-retrained shared model (fedavg_global) could stay
    approximately valid over a wide neighbourhood and a long time window,
    which is what let it beat fusion/consensus under sparse coverage despite
    having no per-region specialisation at all (verified empirically:
    fedavg_global's error DID concentrate in the rare high-offset cells,
    corr(|offset|, MSE)=0.48, 3.6x worse in the top-10% vs bottom-10% offset
    cells -- it just didn't matter much for the mean, because 90% of the map
    was easy).

    IMPORTANT, counterintuitive finding from tuning this (validated via
    dynamic_analytic_floors before spending any GPU time): denser control
    points (shorter spatial correlation length) make the fusion-vs-global
    theoretical GAP SMALLER, not bigger -- pushing disagreement to a scale
    shorter than the FOV (7 cells) hurts a per-node model's own internal
    consistency just as much as it hurts the global model, so the ratio
    between them barely moves (an early attempt at spacing=3.5 gave ratio
    1.47x, actually worse than the original field's 4.97x). What actually
    widens the gap: keep the correlation length >= the original's (spacing
    13.0, matching its 4-corner control-point layout, i.e. each node's own
    FOV stays internally coherent) but sharpen the transitions BETWEEN
    regions (rbf_smoothing 1.0 -> 0.1, so nearby corners blend far less into
    each other) and raise the amplitude (60 -> 100 deg, still safely under
    the 180 deg wrap ceiling). Measured result: global floor 0.0133->0.0674,
    per-node floor 0.0027->0.0076, ratio 4.97x->8.89x -- both the absolute
    gap (~5.6x bigger) and the ratio (~1.8x bigger) improved together.

    TEMPORAL: ~8x more keyframes (30 vs 4) over the same total_steps, so each
    keyframe governs ~13 steps instead of ~130 -- a region's offset drifts
    ~2.4 deg/step here vs ~0.06 deg/step in the original (near-frozen in
    time), giving real staleness pressure within a single hop-propagation
    window instead of over dozens of steps.

    No wake component: the point here is a uniformly turbulent field, not
    one turbulent patch on an otherwise-calm background.
    """
    from scipy.interpolate import RBFInterpolator

    rng = np.random.default_rng(seed)
    n_per_axis = max(2, round((grid_size - 2 * control_margin) / control_spacing) + 1)
    axis_positions = np.linspace(control_margin, grid_size - 1 - control_margin, n_per_axis)
    control_positions = np.array([[r, c] for r in axis_positions for c in axis_positions],
                                 dtype=float)
    keyframe_values = rng.uniform(-background_max_deg, background_max_deg,
                                  size=(n_keyframes, len(control_positions)))
    target_spread = 0.55 * background_max_deg
    row_mean = keyframe_values.mean(axis=1, keepdims=True)
    row_std = np.maximum(keyframe_values.std(axis=1, keepdims=True), 1e-6)
    keyframe_values = row_mean + (keyframe_values - row_mean) / row_std * target_spread
    keyframe_values = np.clip(keyframe_values, -background_max_deg, background_max_deg)
    steps_between = total_steps / (n_keyframes - 1)
    grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    field = np.zeros((total_steps, grid_size, grid_size))
    for step in range(total_steps):
        frame_float = step / steps_between
        k0 = int(frame_float)
        k1 = min(k0 + 1, n_keyframes - 1)
        alpha = frame_float - k0
        vals = (1 - alpha) * keyframe_values[k0] + alpha * keyframe_values[k1]
        rbf = RBFInterpolator(control_positions, vals, kernel="thin_plate_spline",
                              smoothing=rbf_smoothing)
        field[step] = np.clip(rbf(grid_points).reshape(grid_size, grid_size),
                              -background_max_deg, background_max_deg)
    return field


def build_random_wander_path(total_steps: int, grid_size: int, seed: int,
                             step_mean: float = 0.35, step_std: float = 0.20,
                             momentum: float = 0.85) -> np.ndarray:
    """Smoothed, momentum-driven random walk with boundary reflection --
    matches the measured step statistics of the original wearable_path.npy
    (mean step ~0.35, std ~0.20). Random start position each call (seeded).
    Used for independent multi-wearable coverage (THESIS_EXPERIMENTS_SUMMARY
    Stage 4.3-4.4): simpler than any uncertainty-driven routing policy tried,
    and it outperformed all of them."""
    rng = np.random.default_rng(seed)
    lo, hi = 0.1, grid_size - 1.1
    pos = rng.uniform(lo + 2, hi - 2, size=2)
    velocity = rng.normal(0, step_mean, size=2)
    path = [pos.copy()]
    for _ in range(total_steps - 1):
        velocity = momentum * velocity + (1 - momentum) * rng.normal(0, step_mean, size=2)
        speed = np.linalg.norm(velocity)
        target_speed = max(rng.normal(step_mean, step_std), 0.02)
        if speed > 1e-6:
            velocity = velocity / speed * target_speed
        pos = pos + velocity
        for axis in range(2):
            if pos[axis] < lo:
                pos[axis] = lo + (lo - pos[axis])
                velocity[axis] *= -1
            elif pos[axis] > hi:
                pos[axis] = hi - (pos[axis] - hi)
                velocity[axis] *= -1
        path.append(pos.copy())
    return np.array(path)


def analytic_floors(field_deg: np.ndarray, node_centres, fov_size, grid_size):
    field_n = field_deg / 180.0
    global_floor = float(field_n.var())
    fov_vars = []
    for cx, cy in node_centres:
        cells = fov_cells(int(cx), int(cy), fov_size, grid_size)
        vals = np.array([field_n[r, c] for r, c in cells])
        fov_vars.append(vals.var())
    return global_floor, float(np.mean(fov_vars))


def dynamic_analytic_floors(field: np.ndarray, node_centres, fov_size, grid_size, window: int = 50):
    """Mean instantaneous floors over the last `window` steps of a (T,H,W)
    field -- the theoretical minimum for an OMNISCIENT model that knows the
    best constant at each instant. A real continually-trained global model,
    which only observes the anchor's current cell, will generally do worse
    than this: it lags the fastest-drifting regions everywhere it isn't
    currently anchored."""
    gfs, nfs = [], []
    for t in range(len(field) - window, len(field)):
        gf, nf = analytic_floors(field[t], node_centres, fov_size, grid_size)
        gfs.append(gf); nfs.append(nf)
    return float(np.mean(gfs)), float(np.mean(nfs))
