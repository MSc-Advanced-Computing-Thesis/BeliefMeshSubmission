# DIAGNOSTIC (not a chapter figure): per-cell epistemic uncertainty reported by
# each anchor in the v3 three-node colour world, seed 42, mean over the last 50
# timesteps.
#
# The question: is each anchor's confidence spatially structured the way the
# construction intends -- A more confident on the red side of the crossover,
# B on the blue side -- or does the r(nu, error) correlation arise from
# something other than spatial specialisation?
#
# Slot mapping: cell_beliefs_steps stores covering beliefs in ASCENDING node id
# (runner.py:336-339). Node order here is [A, B, C] = ids 0, 1, 2, so per cell
# the slot holding A is the rank of id 0 among that cell's coverers, and
# likewise for B.
#
# Run: python -u experiments/section5_2/make_colour3_v3_epistemic_maps.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import run_colour_three_node_v3 as V3
from beliefmesh.node.mesh import fov_cells

G, FOV, LAST, DPI = V3.G, V3.FOV, 50, 200
A, B, C = V3.A_CENTRE, V3.B_CENTRE, V3.C_CENTRE
a, b, c = V3.regions()
FIELD = V3.colour_field()[0]
VA, VB = V3.VA, V3.VB
MID = (VA + VB) / 2.0
ARM = "nig_product"
RUN = Path("runs/chapter5_v2/s5_3_colour_three_node_v3/%s_seed42" % ARM)


def epistemic_maps():
    bel = np.load(RUN / "cell_beliefs_steps.npy").astype(np.float64)[-LAST:]
    out = {0: np.full((G, G), np.nan), 1: np.full((G, G), np.nan)}
    for r in range(G):
        for cc in range(G):
            ids = [i for i, s in enumerate((a, b, c)) if (r, cc) in s]
            for want in (0, 1):
                if want not in ids:
                    continue
                j = ids.index(want)
                nu, al, be = (bel[:, r, cc, j, k] for k in (1, 2, 3))
                ok = np.isfinite(nu) & (al > 1.0)
                if ok.any():
                    out[want][r, cc] = float(np.mean(
                        be[ok] / (np.maximum(nu[ok], 1e-6)
                                  * np.maximum(al[ok] - 1.0, 1e-6))))
    return out[0], out[1]


def crossover_col():
    shared = c & (a | b)
    cols = sorted({cc for _, cc in shared})
    per = {cc: FIELD[r, cc] for r, cc in shared}
    xs = [k for k in cols if per[k] > MID]
    i = cols.index(xs[-1])
    return cols[i] + (per[cols[i]] - MID) / (per[cols[i]] - per[cols[i + 1]])


def main():
    epiA, epiB = epistemic_maps()
    x0 = crossover_col()
    vmax = np.nanpercentile(np.concatenate([epiA[np.isfinite(epiA)],
                                            epiB[np.isfinite(epiB)]]), 99)
    vmin = np.nanmin([np.nanmin(epiA), np.nanmin(epiB)])

    FIGH, AXW, AXL, AXB = 2.85, 0.335, 0.070, 0.155
    AXH = (AXW * 6.3 / 1.2) / FIGH
    fig = plt.figure(figsize=(6.3, FIGH))
    axes = [fig.add_axes([AXL, AXB, AXW, AXH]),
            fig.add_axes([AXL + AXW + 0.075, AXB, AXW, AXH])]
    for ax, M, lab, ctr in ((axes[0], epiA, "anchor A (trains at %.2f)" % VA, A),
                            (axes[1], epiB, "anchor B (trains at %.2f)" % VB, B)):
        im = ax.imshow(np.ma.masked_invalid(M), cmap="magma_r", vmin=vmin,
                       vmax=vmax, origin="upper", interpolation="nearest")
        # student's supervised region and the crossover
        sh = c & (a | b)
        rs = [x[0] for x in sh]; cs = [x[1] for x in sh]
        ax.add_patch(Rectangle((min(cs) - 0.5, min(rs) - 0.5),
                               max(cs) - min(cs) + 1, max(rs) - min(rs) + 1,
                               fill=False, edgecolor="#1b7a3e", ls="--", lw=1.8,
                               zorder=5))
        ax.axvline(x0, color="#111317", lw=1.4, ls=(0, (5, 2)), zorder=6)
        ax.plot([ctr[0]], [ctr[1]], marker="o", ms=6, mfc="white",
                mec="#22252a", mew=1.0, zorder=7)
        ax.set_xlim(4.5, 16.5); ax.set_ylim(15.5, 5.5)
        ax.set_xticks([6, 9, 12, 15]); ax.set_yticks([6, 9, 12, 15])
        ax.tick_params(labelsize=7)
        ax.set_xlabel("cell column", fontsize=8)
        ax.set_title(lab, fontsize=8.5, pad=4)
    axes[0].set_ylabel("cell row", fontsize=8)
    axes[1].set_yticklabels([])

    cax = fig.add_axes([AXL + 2 * AXW + 0.092, AXB, 0.016, AXH])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("epistemic", fontsize=7, pad=4)
    cb.ax.tick_params(labelsize=6.6)
    cb.outline.set_visible(False)

    p = Path("figures") / "diag_colour3_v3_epistemic_maps.png"
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print("  wrote %s   (DIAGNOSTIC, %s, seed 42, last-%d mean)" % (p, ARM, LAST))

    # ---- quantify ----
    # A left/right split cannot be computed for A: its FOV ends at column 11
    # and the crossover is at 11.20, so A covers NO student cell on the blue
    # side. The direct test of spatial specialisation is instead whether each
    # anchor's confidence degrades as the local filter value departs from the
    # value it trained on.
    print("\nSPATIAL SPECIALISATION TEST -- over each anchor's OWN covered cells")
    print("  does epistemic rise as |local filter - own training value| grows?")
    print("  %-10s %6s %26s %14s" % ("anchor", "cells", "r(|v - v_train|, epistemic)",
                                     "epistemic range"))
    for lab, M, own, vtrain in (("A", epiA, a, VA), ("B", epiB, b, VB)):
        cells = [(r, cc) for r, cc in sorted(own) if np.isfinite(M[r, cc])]
        d = np.array([abs(FIELD[r, cc] - vtrain) for r, cc in cells])
        e = np.array([M[r, cc] for r, cc in cells])
        r_ = float(np.corrcoef(d, e)[0, 1]) if d.std() > 0 else float("nan")
        print("  %-10s %6d %26s %14s"
              % (lab, len(cells), "%+.3f" % r_, "%.5f - %.5f" % (e.min(), e.max())))
    print("  POSITIVE = confidence falls away from its own training condition,")
    print("  which is what spatial specialisation predicts.")

    print("\nBY COLUMN, over the student's supervised region (28 cells):")
    sh = sorted(c & (a | b))
    cols = sorted({cc for _, cc in sh})
    print("  %-12s" % "column" + "".join("%9d" % k for k in cols))
    print("  %-12s" % "filter v" + "".join("%9.3f" % FIELD[sh[0][0], k] for k in cols))
    for lab, M in (("A epistemic", epiA), ("B epistemic", epiB)):
        row = "  %-12s" % lab
        for k in cols:
            v = [M[r, cc] for r, cc in sh if cc == k and np.isfinite(M[r, cc])]
            row += "%9s" % ("%.5f" % np.mean(v) if v else "--")
        print(row)
    print("  ('--' = that anchor does not cover the column)")
    print("\ncrossover column %.2f;  A covers cols %d-%d, B covers cols %d-%d"
          % (x0, min(cc for _, cc in a), max(cc for _, cc in a),
             min(cc for _, cc in b), max(cc for _, cc in b)))
    print("  whole-FOV mean epistemic: A %.5f, B %.5f" % (np.nanmean(epiA), np.nanmean(epiB)))


if __name__ == "__main__":
    main()
