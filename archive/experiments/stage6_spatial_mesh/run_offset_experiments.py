# Mapping-conflict (offset) experiments. CHECKLIST Phase 13, PRE-REGISTERED.
#
# The colour axis proved circumventable: one condition-invariant function fits
# every region, so global parameter averaging wins. These runs replace colour
# heterogeneity with a smooth spatially-varying LABEL OFFSET: the image shows
# rotation theta, the true label is theta + offset(cell), colour held uniform.
# The mapping itself is now location-dependent.
#
# ANALYTIC FLOORS (computed from the actual field and recorded before/with
# results):
#   global floor  = spatial variance of the normalised offset field (+ base
#                   task error): a single shared function cannot output
#                   different answers for identical inputs at different
#                   locations; its best case is the mean offset.
#   per-node floor = mean within-FOV variance of the field (+ base error):
#                   a per-node model can absorb its local mean offset but not
#                   within-FOV variation.
# PREDICTIONS: fedavg_global pinned at >= global floor; fusion (and gossip,
# included as the spatially-adaptive weight-space comparator) approach the
# per-node floor, far below the global floor. If fedavg_global BEATS its
# floor, position information is leaking -- debug before claiming anything.
#
# Run from the project root: python -u experiments/stage6_spatial_mesh/run_offset_experiments.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.evaluation import git_commit
from stage6_spatial_mesh.runner import run_mesh_experiment

from beliefmesh.config import load_config
from beliefmesh.node.mesh import fov_cells

ENV_DIR = Path("experiments/stage6_spatial_mesh/environment_v2")
BASELINE_CHECKPOINT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
RUN_ROOT = Path("runs/stage6/offset_world/static/main_arms")
OFFSET_MAX_DEG = 60.0   # linear gradient across columns: -60 .. +60 degrees
BASE_ERROR = 0.0123     # stage 0 baseline holdout MSE (clean-task error floor)
ARMS = ["frozen", "fedavg_global", "fedavg", "fusion"]


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
                               block_pos: tuple[float, float] = (14.0, 8.0),
                               block_radius: float = 1.5,
                               wake_length: float = 10.0,
                               wake_half_angle_deg: float = 22.0,
                               wake_amplitude: float = 50.0,
                               wake_period: int = 110,
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
    """
    from scipy.interpolate import RBFInterpolator

    rng = np.random.default_rng(seed)
    control_positions = np.array([
        [4, 4], [4, grid_size - 5], [grid_size - 5, 4], [grid_size - 5, grid_size - 5],
    ], dtype=float)
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


def main(arms: list[str]):
    cfg = load_config()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    grids = np.load(ENV_DIR / "environment_grids.npy")
    grid_size, total_steps = grids.shape[1], grids.shape[0]
    # colour held UNIFORM and STATIC: mid-value everywhere, every timestep --
    # the offset field is the only heterogeneity in the experiment
    all_grids = np.full((total_steps, grid_size, grid_size), 0.5)
    wearable_paths = [np.load(ENV_DIR / "wearable_path.npy")]
    node_centres = np.load(ENV_DIR / "node_centres.npy")

    field = build_offset_field(grid_size, total_steps)
    pre_gf, pre_ff = analytic_floors(field[0], node_centres, 7, grid_size)
    post_gf, post_ff = analytic_floors(field[-1], node_centres, 7, grid_size)
    floors = {
        "offset_field": (f"linear column gradient +/-{OFFSET_MAX_DEG} deg; hard-edged "
                         f"+{EVENT_OFFSET_DEG} deg block rows/cols 5-10 from step {EVENT_STEP}"),
        "event_step": EVENT_STEP,
        "base_error_reference": BASE_ERROR,
        "pre_event": {"global_floor": pre_gf, "global_floor_plus_base": pre_gf + BASE_ERROR,
                      "per_node_floor": pre_ff, "per_node_floor_plus_base": pre_ff + BASE_ERROR},
        "post_event": {"global_floor": post_gf, "global_floor_plus_base": post_gf + BASE_ERROR,
                       "per_node_floor": post_ff, "per_node_floor_plus_base": post_ff + BASE_ERROR},
    }
    global_floor, fov_floor = post_gf, post_ff  # last-50-steps metric is post-event
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RUN_ROOT / "analytic_floors.yaml", "w") as f:
        yaml.safe_dump({"git_commit": git_commit(), **floors}, f, sort_keys=False)
    print("ANALYTIC FLOORS (pre-registered):")
    print(yaml.safe_dump(floors, sort_keys=False))

    results = {}
    for arm in arms:
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        results[arm] = run_mesh_experiment(
            cfg, condition=f"offset_{arm}", run_dir=RUN_ROOT / arm,
            all_grids=all_grids, wearable_paths=wearable_paths,
            node_centres=node_centres, fov_size=7, mode=arm,
            baseline_checkpoint=BASELINE_CHECKPOINT,
            n_wearable_samples=1, n_train_repeats=10,
            title=f"Offset mapping-conflict ({arm})",
            offset_field=field,
            extra_manifest={"experiment": "mapping-conflict, CHECKLIST Phase 13",
                            "analytic_floors": floors},
        )

    print("\n=== OFFSET EXPERIMENT SUMMARY ===")
    print(f"global floor (field only / +base): {global_floor:.4f} / {global_floor + BASE_ERROR:.4f}")
    print(f"per-node floor (field only / +base): {fov_floor:.4f} / {fov_floor + BASE_ERROR:.4f}")
    for arm, res in results.items():
        print(f"{arm}: measured {res['mean_mse_last_50']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    main(parser.parse_args().arms)
