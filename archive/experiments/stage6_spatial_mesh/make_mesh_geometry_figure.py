# Thesis method-chapter figure (Sec 4.3): deployed mesh geometry and the
# resulting per-cell coverage. Both panels are generated from the ACTUAL
# placement and FOV code -- node_centres.npy as loaded by every comparator,
# and beliefmesh.node.mesh.fov_cells (including its edge clamps) -- never a
# reconstruction of the geometry.
#
# The left panel is deliberately unpainted: geometry only, no environment or
# offset field, since a coloured background would imply the placement relates
# to the environment when it does not.
#
# In THIS mesh coverage takes only the values {1,2,3,4,6,9} -- it factorises as
# (row factor) x (col factor) with each factor in {1,2,3} -- so 5, 7 and 8 do
# not occur. The scale is nonetheless DISCRETE OVER THE FULL RANGE 1..9, one
# equally spaced band per level, NOT one band per occurring level.
#
# The reason is the other figure. The Section 5.7 node-loss coverage maps
# (make_block2_figures.fig_57_maps) plot the same quantity for a PARTIALLY
# FAILED mesh, which does produce 5, 7 and 8. Binning those onto occurring-only
# levels would silently report a 5-covered cell as a 6 and an 8 as a 9 -- an
# error no reader could detect. Sharing the full 1..9 scale costs this figure
# nothing: it simply leaves three bands unused, and a cell with 4 covering
# nodes renders identically in both figures. THE TWO MUST STAY IN STEP.
#
# Run: python -u experiments/stage6_spatial_mesh/make_mesh_geometry_figure.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beliefmesh.node.mesh import fov_cells

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
OUT_DIR = Path("figures")
FOV = 7
# FOVs outlined for two ADJACENT nodes (centres 3 apart, the stride). Adjacent
# rather than distant deliberately: two separated FOVs would show the 7x7
# extent but nothing else, whereas neighbours at one stride overlap in a 4-cell
# band, and the doubled fill makes visible the overlap that the right panel
# then quantifies. Specified by centre coordinate, not list index, so the
# choice survives any reordering of node_centres.npy.
HIGHLIGHT_CENTRES = [(9, 9), (12, 9)]      # (col, row)

# Two hues from Figure 3.1's violet/teal/orange, so the two figures read as one
# family. teal+orange chosen over violet+teal: the two single regions separate
# far more strongly (composited distance 39.3 vs 20.5 at alpha=0.28), their
# intersection composites to an olive that is unmistakably a THIRD colour
# rather than a darker shade of either parent, and blue-green/orange is the
# standard colour-vision-deficiency-safe pairing whereas violet/teal is not.
FOV_COLOURS = ["#1B9E9E", "#E6821B"]       # teal, orange
FOV_ALPHA = 0.28                            # see the alpha comparison in the
                                            # commit message: 0.22 leaves the
                                            # overlap too faint, 0.35 starts to
                                            # bury the gridlines.
NODE_NEUTRAL = "#8c8c8c"
GRID_COLOUR, GRID_LW = "#bdbdbd", 0.5


def build():
    centres = np.load(ENV / "node_centres.npy")
    grids = np.load(ENV / "environment_grids.npy")
    G = grids.shape[1]
    cov = np.zeros((G, G), dtype=int)
    for cx, cy in centres:                      # cx -> column, cy -> row
        for r, c in fov_cells(int(cx), int(cy), FOV, G):
            cov[r, c] += 1
    return centres, cov, G


def draw(centres, cov, G, annotate: bool, out: Path):
    # FIXED levels 1..9, not the occurring ones. The node-loss coverage maps
    # (make_block2_figures.fig_57_maps) plot the same quantity and DO exercise
    # 5, 7 and 8; binning those onto occurring-only levels would misreport the
    # count in a way a reader cannot detect. Sharing the full scale costs this
    # figure nothing -- it simply leaves three bands unused. The two must stay
    # in step.
    levels = list(range(1, 10))
    # discrete: one equal-width band per level
    base = plt.get_cmap("YlGnBu")
    colours = [base(0.12 + 0.80 * i / (len(levels) - 1)) for i in range(len(levels))]
    cmap = ListedColormap(colours)
    idx = np.searchsorted(levels, cov)          # level -> band index
    norm = BoundaryNorm(np.arange(len(levels) + 1) - 0.5, len(levels))

    fig = plt.figure(figsize=(6.3, 3.15))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.045], wspace=0.18)
    axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1]); cax = fig.add_subplot(gs[2])

    # ---- left: geometry only -------------------------------------------
    axL.set_xlim(-0.5, G - 0.5); axL.set_ylim(G - 0.5, -0.5)
    axL.set_aspect("equal")
    for k in range(G + 1):
        axL.axhline(k - 0.5, color=GRID_COLOUR, lw=GRID_LW, zorder=1)
        axL.axvline(k - 0.5, color=GRID_COLOUR, lw=GRID_LW, zorder=1)
    for hc, col in zip(HIGHLIGHT_CENTRES, FOV_COLOURS):
        cells = fov_cells(int(hc[0]), int(hc[1]), FOV, G)   # real cells, incl. clamps
        rs = [c[0] for c in cells]; cs = [c[1] for c in cells]
        axL.add_patch(Rectangle((min(cs) - 0.5, min(rs) - 0.5),
                                max(cs) - min(cs) + 1, max(rs) - min(rs) + 1,
                                facecolor=col, alpha=FOV_ALPHA,
                                edgecolor=col, lw=1.3, zorder=2))
    # the 34 non-highlighted nodes stay neutral so region ownership is unambiguous
    owners = {tuple(int(v) for v in hc) for hc in HIGHLIGHT_CENTRES}
    others = np.array([c for c in centres
                       if tuple(int(v) for v in c) not in owners])
    axL.scatter(others[:, 0], others[:, 1], s=13, c=NODE_NEUTRAL,
                edgecolors="white", linewidths=0.5, zorder=3)
    for hc, col in zip(HIGHLIGHT_CENTRES, FOV_COLOURS):
        axL.scatter([hc[0]], [hc[1]], s=28, c=col,
                    edgecolors="white", linewidths=0.8, zorder=4)
    axL.set_xticks([0, 7, 14, 21]); axL.set_yticks([0, 7, 14, 21])
    axL.tick_params(labelsize=7.5, length=2, pad=1.5)
    axL.set_xlabel("cell column", fontsize=8)
    axL.set_ylabel("cell row", fontsize=8)
    for s in axL.spines.values():
        s.set_color("#999999"); s.set_linewidth(0.6)
    axL.set_title("Node placement", fontsize=10, pad=6)

    # ---- right: coverage ------------------------------------------------
    axR.imshow(idx, cmap=cmap, norm=norm, interpolation="nearest")
    axR.set_aspect("equal")
    # node_centres[:,0] is the COLUMN (cx), [:,1] the ROW (cy) -- same
    # convention as make_block2_figures.fig_57_maps
    if not annotate:
        axR.plot(centres[:, 0], centres[:, 1], "o", ms=3.6, mfc="white",
                 mec="#22252a", mew=0.8, ls="none", zorder=5)
    axR.set_xticks([0, 7, 14, 21]); axR.set_yticks([0, 7, 14, 21])
    axR.tick_params(labelsize=7.5, length=2, pad=1.5)
    axR.set_xlabel("cell column", fontsize=8)
    for s in axR.spines.values():
        s.set_color("#999999"); s.set_linewidth(0.6)
    axR.set_title("Cell coverage", fontsize=10, pad=6)
    if annotate:
        for r in range(G):
            for c in range(G):
                rgb = colours[idx[r, c]][:3]
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                axR.text(c, r, str(cov[r, c]), ha="center", va="center",
                         fontsize=3.1,
                         color="white" if lum < 0.55 else "#333333")

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      ticks=np.arange(len(levels)))
    cb.ax.set_yticklabels([str(v) for v in levels], fontsize=8)
    cb.set_label("nodes covering cell", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)

    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return levels


def main():
    OUT_DIR.mkdir(exist_ok=True)
    centres, cov, G = build()
    levels = draw(centres, cov, G, True, OUT_DIR / "mesh_geometry_annotated.pdf")
    draw(centres, cov, G, False, OUT_DIR / "mesh_geometry.pdf")

    print(f"grid {G}x{G} = {G*G} cells | {len(centres)} nodes | FOV {FOV}x{FOV} | "
          f"stride {int(np.diff(np.unique(centres[:,0]))[0])}")
    present = [tuple(c) for c in centres.astype(int).tolist()]
    for hc in HIGHLIGHT_CENTRES:
        assert hc in present, f"highlighted centre {hc} is not an actual node"
    print(f"highlighted FOVs at centres {HIGHLIGHT_CENTRES} (col,row) -- "
          f"adjacent, {abs(HIGHLIGHT_CENTRES[0][0]-HIGHLIGHT_CENTRES[1][0])} apart")
    print()
    print("coverage distribution")
    print(f"  {'nodes covering':>14s} {'cells':>7s} {'% of grid':>10s}")
    for v in levels:
        n = int((cov == v).sum())
        print(f"  {v:>14d} {n:>7d} {100*n/(G*G):>9.1f}%")
    print(f"  {'TOTAL':>14s} {G*G:>7d} {100.0:>9.1f}%")
    print()
    print(f"mean coverage {cov.mean():.4f} | median {np.median(cov):.1f} | "
          f"min {cov.min()} | max {cov.max()}")
    print(f"cells with >=2 nodes (fusible): {int((cov>=2).sum())} "
          f"({100*(cov>=2).mean():.1f}%)")
    print(f"unreachable levels in [1,9]: {[v for v in range(1,10) if v not in levels]}")
    print(f"\nsaved {OUT_DIR/'mesh_geometry.png'} and {OUT_DIR/'mesh_geometry_annotated.png'}")


if __name__ == "__main__":
    main()
