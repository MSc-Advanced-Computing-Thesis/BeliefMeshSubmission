# V4 geometry figure. The colour field is imported from the RUN script, not
# reimplemented, so the figure cannot drift from what was actually executed.
#
# Run: python -u experiments/section5_2/make_colour3_v4_geometry.py

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import experiment_three_node_colour_world as V3
from beliefmesh.node.mesh import fov_cells

G, FOV, DPI = V3.G, V3.FOV, 200
A, B, C = V3.A_CENTRE, V3.B_CENTRE, V3.C_CENTRE
a, b, c = V3.regions()
FIELD = V3.colour_field()[0]          # (G, G), constant in time
VA, VB = V3.VA, V3.VB
MID = (VA + VB) / 2.0


def rect_of(cells, ax, colour, ls, lw, z=4):
    rs = [x[0] for x in cells]; cs = [x[1] for x in cells]
    ax.add_patch(Rectangle((min(cs) - 0.5, min(rs) - 0.5),
                           max(cs) - min(cs) + 1, max(rs) - min(rs) + 1,
                           fill=False, edgecolor=colour, ls=ls, lw=lw, zorder=z))


def main():
    # The data box is 12 columns x 10 rows, so with imshow(aspect="equal") the
    # AXES box must be 1.2:1 or matplotlib pads the difference as dead space
    # inside the axes -- which is what made the old margins so large.
    FIGH, AXW, AXL, AXB = 2.85, 0.42, 0.077, 0.145
    AXH = (AXW * 6.3 / 1.2) / FIGH
    fig, ax = plt.subplots(figsize=(6.3, FIGH))
    fig.subplots_adjust(left=AXL, right=AXL + AXW, bottom=AXB, top=AXB + AXH)
    im = ax.imshow(FIELD, cmap="RdBu_r", vmin=0.0, vmax=1.0, origin="upper",
                   interpolation="nearest", zorder=0)

    for ctr, lab in ((A, "A"), (B, "B"), (C, "C")):
        rect_of(fov_cells(*ctr, FOV, G), ax, "#3a3f46", "-", 0.9, z=3)
        ax.text(ctr[0], ctr[1], lab, ha="center", va="center", fontsize=8,
                color="white", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.14", fc="#22252a", ec="none"))
    rect_of(a & b, ax, "#111317", "-", 2.2)
    rect_of(c & (a | b), ax, "#1b7a3e", "--", 2.2)
    rect_of(c - a - b, ax, "#6b3fa0", ":", 2.4)

    for cells in (a - b - c, b - a - c):
        arr = np.array(sorted(cells))
        ax.plot(arr[:, 1], arr[:, 0], marker=".", ls="none", ms=1.9,
                color="white", mec="#22252a", mew=0.2, zorder=6)

    # crossover still COMPUTED for the printed report and the caption, but no
    # longer drawn: the gradient already shows where the field turns over
    shared = c & (a | b)
    cols = sorted({cc for _, cc in shared})
    per = {cc: FIELD[r, cc] for r, cc in shared}
    xs = [k for k in cols if per[k] > MID]
    if xs and len(xs) < len(cols):
        i = cols.index(xs[-1])
        x0 = cols[i] + (per[cols[i]] - MID) / (per[cols[i]] - per[cols[i + 1]])
        ax.annotate("crossover\ncol %.2f" % x0, xy=(x0, 4.0),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=6.6, color="#111317", va="top", ha="left", zorder=9)

    ax.set_xlim(4.5, 16.5); ax.set_ylim(15.5, 5.5)
    ax.set_xticks([6, 9, 12, 15]); ax.set_yticks([6, 9, 12, 15])
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("cell column", fontsize=8)
    ax.set_ylabel("cell row", fontsize=8)

    # colourbar spans exactly the axes vertical extent
    cax = fig.add_axes([AXL + AXW + 0.017, AXB, 0.016, AXH])
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, VB, MID, VA, 1.0])
    cb.ax.set_yticklabels(["0.0", "%.2f B" % VB, "%.2f mid" % MID,
                           "%.2f A" % VA, "1.0"], fontsize=6.4)
    cb.ax.set_title("filter value", fontsize=7, pad=4)
    cb.outline.set_visible(False)

    nA = sum(1 for r, cc in shared if FIELD[r, cc] > MID)
    handles = [
        plt.Line2D([], [], color="#111317", ls="-", lw=2.2,
                   label="A$\\cap$B: excluded from both"),
        plt.Line2D([], [], color="#1b7a3e", ls="--", lw=2.2,
                   label="C's supervision (28 cells)"),
        plt.Line2D([], [], color="#6b3fa0", ls=":", lw=2.4,
                   label="C exclusive: unsupervised"),
        plt.Line2D([], [], color="#3a3f46", ls="-", lw=0.9, label="field of view"),
        plt.Line2D([], [], marker=".", ls="none", ms=6, color="white",
                   mec="#22252a", mew=0.5, label="anchor wearable confined")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(AXL + AXW + 0.125, AXB + AXH / 2.0),
               fontsize=6.9, frameon=False, handlelength=2.4, labelspacing=0.75)
    p = FIG_DIR / "fig_5_5_three_node_geometry.png"
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print("  wrote %s" % p)
    print("  per-column v across C's supervised region:")
    print("    " + "  ".join("c%d=%.3f" % (k, per[k]) for k in cols))
    print("  A trains %.2f, B trains %.2f, midpoint %.3f, crossover col %.2f"
          % (VA, VB, MID, x0))


if __name__ == "__main__":
    main()
