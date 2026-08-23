# Independent offset-field realisations (2026-08).
#
# WHY THIS EXISTS. build_dynamic_offset_field() defaults to seed=7 and no
# comparator ever passes one, so every multi-seed replication in this project
# varies cell rotations, wearable paths and (in the heterogeneity runs) device
# layout -- but never the spatial structure of the task. Every reported result
# is therefore conditional on ONE field realisation. This module generates
# independent fields so that conditionality can be lifted.
#
# WHY A NEW MODULE RATHER THAN EDITING THE ORIGINAL. build_dynamic_offset_field
# is called by ~25 scripts and every reported run depends on its exact output.
# It is left untouched. build_offset_field_general() below is a strict
# generalisation -- parameterising what the original hard-codes (bearing sweep
# amplitude/period/centre/phase, turbulence phase, and an arbitrary NUMBER of
# wedges) -- and reproduce_original() asserts bit-for-bit equality with the
# original at the original settings. That assertion is the guarantee that the
# new fields differ from the old one only in the ways intended.
#
# Run: python -u experiments/stage6_spatial_mesh/offset_field_variants.py
# (self-test: verifies the bit-for-bit reproduction, prints nothing else)

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REFERENCE_GRID = 22.0


def wrap_deg(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap degrees into [-180, 180) -- the same circular convention the label
    pipeline uses (GridEnvironment._label wraps normalised angle into [-1,1)
    with period 2, i.e. 360 degrees). Applied to the composed field so that
    summing multiple wedges can never emit an out-of-range 'angle'."""
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


@dataclass
class Wedge:
    """One obstruction and the turbulent wedge swept behind it.

    block_pos/block_radius/length are in CELLS (grid coordinates, (row, col)).
    The wedge occupies cells whose distance from block_pos lies in
    [block_radius, block_radius + length] and whose bearing from block_pos is
    within half_angle_deg of the current sweep bearing.

    bearing(t) = bearing_centre_deg
                 + bearing_amp_deg * sin(2 pi t / bearing_period + bearing_phase)
    turbulence(t) = amplitude * sin(2 pi t / period + phase)

    The original build_dynamic_offset_field hard-codes bearing_centre_deg=0,
    bearing_amp_deg=70, bearing_period=total_steps*2.5, and both phases=0.
    """
    block_pos: tuple[float, float]
    block_radius: float = 1.5
    length: float = 10.0
    half_angle_deg: float = 22.0
    amplitude: float = 50.0
    period: int = 110
    phase: float = 0.0
    bearing_centre_deg: float = 0.0
    bearing_amp_deg: float = 70.0
    bearing_period: float | None = None      # None -> total_steps * 2.5
    bearing_phase: float = 0.0

    def scaled(self, scale: float) -> "Wedge":
        """Proportionally rescale cell-valued geometry for a non-22 grid, the
        same way the original scales block_pos/radius/length."""
        return Wedge(block_pos=(self.block_pos[0] * scale, self.block_pos[1] * scale),
                     block_radius=self.block_radius * scale,
                     length=self.length * scale,
                     half_angle_deg=self.half_angle_deg, amplitude=self.amplitude,
                     period=self.period, phase=self.phase,
                     bearing_centre_deg=self.bearing_centre_deg,
                     bearing_amp_deg=self.bearing_amp_deg,
                     bearing_period=self.bearing_period,
                     bearing_phase=self.bearing_phase)


def _background(grid_size: int, total_steps: int, n_keyframes: int,
                background_max_deg: float, control_margin: float,
                control_spacing: float, seed: int) -> np.ndarray:
    """Identical arithmetic to build_dynamic_offset_field's background block."""
    from scipy.interpolate import RBFInterpolator

    rng = np.random.default_rng(seed)
    n_per_axis = max(2, round((grid_size - 2 * control_margin) / control_spacing) + 1)
    axis_positions = np.linspace(control_margin, grid_size - 1 - control_margin, n_per_axis)
    control_positions = np.array([[r, c] for r in axis_positions for c in axis_positions],
                                 dtype=float)
    keyframe_values = rng.uniform(-background_max_deg, background_max_deg,
                                  size=(n_keyframes, len(control_positions)))
    target_spread = 0.45 * background_max_deg
    row_mean = keyframe_values.mean(axis=1, keepdims=True)
    row_std = np.maximum(keyframe_values.std(axis=1, keepdims=True), 1e-6)
    keyframe_values = row_mean + (keyframe_values - row_mean) / row_std * target_spread
    keyframe_values = np.clip(keyframe_values, -background_max_deg, background_max_deg)
    steps_between = total_steps / (n_keyframes - 1)
    grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    out = np.zeros((total_steps, grid_size, grid_size))
    for step in range(total_steps):
        frame_float = step / steps_between
        k0 = int(frame_float)
        k1 = min(k0 + 1, n_keyframes - 1)
        alpha = frame_float - k0
        vals = (1 - alpha) * keyframe_values[k0] + alpha * keyframe_values[k1]
        rbf = RBFInterpolator(control_positions, vals, kernel="thin_plate_spline",
                              smoothing=1.0)
        out[step] = np.clip(rbf(grid_points).reshape(grid_size, grid_size),
                            -background_max_deg, background_max_deg)
    return out


def build_offset_field_general(grid_size: int, total_steps: int,
                               wedges: list[Wedge],
                               n_keyframes: int = 4,
                               background_max_deg: float = 60.0,
                               control_margin: float = 4.0,
                               control_spacing: float = 13.0,
                               seed: int = 7,
                               wrap: bool = True,
                               return_masks: bool = False):
    """(T, H, W) offset field in degrees: smooth RBF background + N swept
    turbulent wedges, summed. With wrap=True the composed field is wrapped
    into [-180, 180) so multiple overlapping wedges can never produce an
    invalid angle.

    NOTE on the wrap: it is a genuine no-op whenever
    background_max_deg + sum(|amplitude|) < 180, which holds for every
    configuration used here (60 + 50 = 110 single-wedge, 60 + 50 + 50 = 160
    dual-wedge). It is applied for correctness-by-construction, not because
    it currently bites -- see report_wrap_headroom().
    """
    scale = grid_size / REFERENCE_GRID
    wedges = [w.scaled(scale) for w in wedges]

    out = _background(grid_size, total_steps, n_keyframes, background_max_deg,
                      control_margin, control_spacing, seed)

    r_idx, c_idx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    t = np.arange(total_steps)
    # per-wedge masks, shape (n_wedges, T, H, W) -- callers reduce with
    # .any(axis=0) for coverage, or .sum(axis=0) >= 2 for the overlap region
    masks = np.zeros((len(wedges), total_steps, grid_size, grid_size), dtype=bool)

    for wi, w in enumerate(wedges):
        dr, dc = r_idx - w.block_pos[0], c_idx - w.block_pos[1]
        dist = np.sqrt(dr ** 2 + dc ** 2)
        angle_to_cell = np.arctan2(dr, dc)
        bearing_period = w.bearing_period if w.bearing_period is not None else total_steps * 2.5
        theta = (np.deg2rad(w.bearing_centre_deg)
                 + np.deg2rad(w.bearing_amp_deg)
                 * np.sin(2 * np.pi * t / bearing_period + w.bearing_phase))
        half_angle = np.deg2rad(w.half_angle_deg)
        for step in range(total_steps):
            ang_diff = np.abs(((angle_to_cell - theta[step]) + np.pi) % (2 * np.pi) - np.pi)
            in_wake = ((dist >= w.block_radius)
                       & (dist <= w.block_radius + w.length)
                       & (ang_diff <= half_angle))
            turbulence = w.amplitude * np.sin(2 * np.pi * step / w.period + w.phase)
            out[step][in_wake] += turbulence
            masks[wi, step] = in_wake

    if wrap:
        out = wrap_deg(out)
    return (out, masks) if return_masks else out


# --- the five field realisations -------------------------------------------
# FIELD_0 is the existing/original field (seed 7). The four new ones vary
# obstruction position, bearing sweep (centre, amplitude, period) and
# turbulence phase, so they differ in SPATIAL STRUCTURE and not merely in the
# background noise draw. Background seeds are distinct too.

ORIGINAL_WEDGE = Wedge(block_pos=(14.0, 8.0))   # all other fields default-matched

FIELDS: dict[str, dict] = {
    "field0_original": dict(
        seed=7,
        wedges=[ORIGINAL_WEDGE],
        # wrap disabled ONLY here, so this field is bit-for-bit identical to
        # the one every reported run used. The wrap is a mathematical no-op at
        # these amplitudes but is not a floating-point no-op (it perturbs by
        # ~3e-14 deg), and there is no reason to carry even that against the
        # already-reported baseline.
        wrap=False,
        note="existing field -- reproduces build_dynamic_offset_field(22,390) exactly",
    ),
    "field1_seed11": dict(
        seed=11,
        wedges=[Wedge(block_pos=(7.0, 13.0), bearing_centre_deg=140.0,
                      bearing_amp_deg=55.0, bearing_period=390 * 1.7,
                      phase=np.pi / 2)],
        note="obstruction upper-right, wedge sweeps about 140 deg, turbulence quarter-phase",
    ),
    "field2_seed23": dict(
        seed=23,
        wedges=[Wedge(block_pos=(16.0, 15.0), bearing_centre_deg=-95.0,
                      bearing_amp_deg=85.0, bearing_period=390 * 3.1,
                      phase=np.pi)],
        note="obstruction lower-right, wide slow sweep about -95 deg, turbulence anti-phase",
    ),
    "field3_seed37": dict(
        seed=37,
        wedges=[Wedge(block_pos=(9.0, 5.0), bearing_centre_deg=35.0,
                      bearing_amp_deg=65.0, bearing_period=390 * 2.2,
                      phase=3 * np.pi / 2)],
        note="obstruction left-centre, sweep about +35 deg, turbulence three-quarter phase",
    ),
    # Two obstructions in a SHARED FLOW. Both wedges run broadly downstream in
    # the same direction -- they are wakes behind two objects in one current,
    # not two opposed jets, so their bearings must not be antiparallel. They
    # share a sweep centre (45 deg), amplitude and period; block B's wedge is
    # aimed 10 deg off A's and lags it slightly in phase, representing the
    # local flow turning marginally sooner at one object than the other.
    # Block B sits ~7.1 cells downstream of A along the mean flow bearing
    # (atan2(+5,+5) = +45 deg), so B stands IN A's wake and the two wakes
    # merge downstream of B -- which is where the offsets sum.
    "field4_seed51_dual": dict(
        seed=51,
        wedges=[Wedge(block_pos=(7.0, 6.0), bearing_centre_deg=45.0,
                      bearing_amp_deg=35.0, bearing_period=390 * 2.0,
                      bearing_phase=0.0, phase=0.0),
                Wedge(block_pos=(12.0, 11.0), bearing_centre_deg=55.0,
                      bearing_amp_deg=35.0, bearing_period=390 * 2.0,
                      bearing_phase=0.35, phase=np.pi / 3)],
        note="two obstructions in one flow -- co-directional wakes merging downstream",
    ),
}


def build_field(name: str, grid_size: int = 22, total_steps: int = 390,
                return_masks: bool = False):
    spec = FIELDS[name]
    return build_offset_field_general(grid_size, total_steps, wedges=spec["wedges"],
                                      seed=spec["seed"],
                                      wrap=spec.get("wrap", True),
                                      return_masks=return_masks)


def wrap_headroom(name: str, background_max_deg: float = 60.0) -> float:
    """Worst-case |offset| this field's construction can emit, before wrapping:
    background clip + every wedge's amplitude. If this is < 180 the wrap can
    never trigger, so no +-180 discontinuity is reachable by construction."""
    return background_max_deg + sum(abs(w.amplitude) for w in FIELDS[name]["wedges"])


def reproduce_original(grid_size: int = 22, total_steps: int = 390) -> None:
    """Assert the generalisation reproduces build_dynamic_offset_field exactly
    at the original settings. If this ever fails, the new fields are not
    comparable with the reported ones and nothing downstream should be trusted."""
    from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field
    ref = build_dynamic_offset_field(grid_size, total_steps)
    got = build_field("field0_original", grid_size, total_steps)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.array_equal(got, ref), (
        f"generalised generator diverges from the original: "
        f"max abs diff {np.abs(got - ref).max():.3e}")
    # and confirm the wrap itself is inert at these amplitudes
    wrapped = build_offset_field_general(grid_size, total_steps, wedges=[ORIGINAL_WEDGE],
                                         seed=7, wrap=True)
    assert np.abs(wrapped - ref).max() < 1e-12, "wrap perturbs the field materially"


if __name__ == "__main__":
    reproduce_original()
    print("OK: build_offset_field_general reproduces build_dynamic_offset_field "
          "bit-for-bit at the original settings (22x22, 390 steps, seed 7)")
    for nm in FIELDS:
        print(f"   {nm:22s} wrap headroom {wrap_headroom(nm):6.1f} deg "
              f"({'wrap unreachable' if wrap_headroom(nm) < 180 else 'WRAP CAN TRIGGER'})")
