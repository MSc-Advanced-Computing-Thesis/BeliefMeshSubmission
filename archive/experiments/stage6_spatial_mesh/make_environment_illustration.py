# Thesis figure: colour world vs offset world, side by side, rendered from
# the real GridEnvironment code path (no hand-drawn/schematic content) --
# each cell exactly as a node would receive it: digit rotated to that
# cell's angle, filtered by that cell's environmental conditions. Same
# timestep/seed/EXCLUDED_RANGES/node-scale environment_v2 configuration used
# by every reported comparator run this session.
#
# Run: python -u experiments/stage6_spatial_mesh/make_environment_illustration.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

from beliefmesh.data.grid_environment import GridEnvironment

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
OUT = Path("figures/environment_illustration.pdf")
SEED = 42
STEP = 0
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
# BOTH panels show the SAME grid region, so corresponding cells carry the
# identical base rotation (both environments draw cell_rotations from the
# same rotation_seed, so same region => same angles, cell for cell). The
# panels then differ in exactly one respect each, which is the point of the
# figure: the colour world applies a colour filter on top of that shared
# rotation while the label stays the rotation itself; the offset world
# leaves the appearance plain and instead adds the annotated offset to the
# label. Region chosen for spatially diverse red/blue colour (13 cells
# >0.6, 12 cells <0.0, spanning -0.58 to 1.0) AND a smooth offset gradient.
REGION_R0, REGION_C0 = 11, 7
REGION_N = 6
# Rendered at 120 deg (vs the 60 deg default) so the offsets are large
# enough to read at print size: +2 to +58 deg across the window, still
# smooth (max cell-to-cell jump ~11 deg).
OFFSET_MAX_DEG = 120.0


def build_environments():
    colour_grids = np.load(ENV / "environment_grids.npy")   # (T,G,G), real colour field
    T, G = colour_grids.shape[0], colour_grids.shape[1]
    uniform_grids = np.full((T, G, G), 0.5)                 # unused numerically once filter is off
    offset_field = build_dynamic_offset_field(G, T, background_max_deg=OFFSET_MAX_DEG)

    colour_env = GridEnvironment(G, colour_grids, rotation_seed=SEED,
                                 excluded_rotation_ranges=EXCLUDED_RANGES)
    offset_env = GridEnvironment(G, uniform_grids, rotation_seed=SEED,
                                 offset_field=offset_field,
                                 excluded_rotation_ranges=EXCLUDED_RANGES,
                                 apply_colour_filter=False)
    # Display-only environment used solely to render the TARGET orientation
    # as a ghost outline. The real offset world never renders this: its
    # image is rotated by theta alone (grid_environment.py get_cell_input)
    # and the offset enters only through _label(). Rotating by theta+offset
    # here is a visualisation of the answer, not of any node's input --
    # which is exactly the quantity the offset world varies.
    ghost_env = GridEnvironment(G, uniform_grids, rotation_seed=SEED,
                                excluded_rotation_ranges=EXCLUDED_RANGES,
                                apply_colour_filter=False)
    ghost_env.cell_rotations = ghost_env.cell_rotations + offset_field
    return colour_env, offset_env, ghost_env, offset_field, G


def _ink(tile: np.ndarray) -> np.ndarray:
    """Per-pixel digit coverage: 0 on the white background, ~1 on the ink."""
    return np.clip((1.0 - tile.min(axis=2)) * 1.6, 0.0, 1.0)


def ghost_rgba(tile: np.ndarray, rgb=(0.55, 0.57, 0.62), alpha=0.42) -> np.ndarray:
    """Faint grey silhouette of a rendered tile -- the TARGET orientation,
    drawn underneath the input so the input stays the dominant mark."""
    out = np.zeros((*tile.shape[:2], 4), dtype=float)
    out[..., 0], out[..., 1], out[..., 2] = rgb
    out[..., 3] = _ink(tile) * alpha
    return out


def input_rgba(tile: np.ndarray) -> np.ndarray:
    """The rendered input with its white background made transparent, so it
    can be composited over the ghost without hiding it."""
    out = np.zeros((*tile.shape[:2], 4), dtype=float)
    out[..., :3] = tile
    out[..., 3] = _ink(tile)
    return out


def render_region(env: GridEnvironment, step: int, r0: int, c0: int, n: int):
    """(n, n) array of (28,28,3) numpy images, exactly as get_cell_input
    renders them -- no separate/simplified rendering path."""
    tiles = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            img, _ = env.get_cell_input(r0 + i, c0 + j, step)
            tiles[i, j] = img.permute(1, 2, 0).numpy()
    return tiles


def main():
    colour_env, offset_env, ghost_env, offset_field, G = build_environments()
    colour_tiles = render_region(colour_env, STEP, REGION_R0, REGION_C0, REGION_N)
    offset_tiles = render_region(offset_env, STEP, REGION_R0, REGION_C0, REGION_N)
    ghost_tiles = render_region(ghost_env, STEP, REGION_R0, REGION_C0, REGION_N)
    offset_vals = offset_field[STEP, REGION_R0:REGION_R0 + REGION_N,
                               REGION_C0:REGION_C0 + REGION_N]
    rotations = colour_env.cell_rotations[STEP, REGION_R0:REGION_R0 + REGION_N,
                                          REGION_C0:REGION_C0 + REGION_N]
    assert np.allclose(rotations, offset_env.cell_rotations[
        STEP, REGION_R0:REGION_R0 + REGION_N, REGION_C0:REGION_C0 + REGION_N])

    print(f"timestep={STEP}  seed={SEED}  full_grid={G}x{G}  "
          f"region=rows[{REGION_R0}:{REGION_R0+REGION_N}] cols[{REGION_C0}:{REGION_C0+REGION_N}]")
    print(f"offset background_max_deg={OFFSET_MAX_DEG:g}")
    print("shared base rotation of the digit in each cell (degrees, identical in both panels):")
    for row in rotations:
        print("  " + "  ".join(f"{v:+7.1f}" for v in row))
    print("offset added to the label in the offset world (degrees):")
    for row in offset_vals:
        print("  " + "  ".join(f"{v:+6.1f}" for v in row))

    fig = plt.figure(figsize=(6.3, 3.5))
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12)

    for panel_idx, (title, tiles, annotate) in enumerate([
        ("Colour world", colour_tiles, False),
        ("Offset world", offset_tiles, True),
    ]):
        inner = gridspec.GridSpecFromSubplotSpec(
            REGION_N, REGION_N, subplot_spec=outer[panel_idx], wspace=0.03, hspace=0.03)
        for i in range(REGION_N):
            for j in range(REGION_N):
                ax = fig.add_subplot(inner[i, j])
                if annotate:
                    # target orientation (theta + offset) UNDER the input, so
                    # the green input remains the dominant mark
                    ax.set_facecolor("white")
                    ax.imshow(ghost_rgba(ghost_tiles[i, j]), interpolation="nearest")
                    ax.imshow(input_rgba(tiles[i, j]), interpolation="nearest")
                else:
                    ax.imshow(tiles[i, j], interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.5)
                    spine.set_color("#999999")
                if annotate:
                    # corner-anchored so it never sits on top of either digit
                    ax.text(0.04, 0.04, f"{offset_vals[i, j]:+.0f}°",
                           transform=ax.transAxes, ha="left", va="bottom",
                           fontsize=5.5, color="#B03030", fontweight="bold",
                           bbox=dict(boxstyle="square,pad=0.08", facecolor="white",
                                     edgecolor="none", alpha=0.7))
        # panel title spanning the 6x6 block
        title_ax = fig.add_subplot(outer[panel_idx])
        title_ax.set_title(title, fontsize=10, pad=10)
        title_ax.axis("off")

    fig.suptitle("")
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
