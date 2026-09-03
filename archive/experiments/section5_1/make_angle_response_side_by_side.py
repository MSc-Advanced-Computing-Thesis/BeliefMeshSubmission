# Section 5.1 angle response, SIDE BY SIDE. Halves the height of the stacked
# version by placing the two panels left/right instead of top/bottom.
#
#   left  : held-out population, 15-degree bins
#   right : the mesh digit-7 instance at half-degree resolution
#
# Shared x-label beneath both. Independent y-ranges and independent right-hand
# axes: the two are not comparable in level.
#
# TWO LEGENDS, not one. The marks differ between panels -- a 15-degree bin
# mean (INK, o-) is not the same object as a single render's error curve
# (GREEN, -) -- so a consolidated legend needs group headings to say which
# entry applies where, which position already states. Same reasoning as the
# stacked version's own note.
#
# PNG only, 6.3 in wide, dpi 200.
#
# Run: python -u experiments/section5_1/make_angle_response_side_by_side.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse import read_dump
from make_figures import BLUE, FIGS, GREY, INK, RED, WIDTH, load_csv
from make_angle_response_stacked import (DIAGNOSED, ELEVATED, GREEN,
                                         PRECAUTIONARY, SHADE_DIAG,
                                         SHADE_ELEV, SHADE_PREC)

HEIGHT = 2.5
NAME = "section5_1_angle_response_side_by_side"

LEG = dict(fontsize=5.9, frameon=False, loc="upper center", ncol=2,
           handlelength=1.5, handletextpad=0.5, columnspacing=1.0,
           labelspacing=0.35, borderpad=0.0)


def main(height: float = HEIGHT, name: str = NAME):
    b = load_csv(Path("runs/section5_1/f_angle_bins/error_by_true_angle.csv"))
    pop = read_dump(Path("runs/section5_1/r1a/per_sample_baseline.csv"))
    j = read_dump(Path("runs/section5_1/j_single_instance/per_render.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height))
    # legends below the panels, so the space above is reclaimed for the axes
    fig.subplots_adjust(left=0.072, right=0.918, bottom=0.395, top=0.905,
                        wspace=0.62)

    # ── left: population, 15-degree bins ─────────────────────────────────
    ax = axes[0]
    ax.axvspan(*ELEVATED, color=SHADE_ELEV, lw=0, zorder=0)
    ax.scatter(pop["true_angle_deg"], np.abs(pop["circular_error_deg"]), s=1.6,
               color=GREY, alpha=0.24, lw=0, zorder=2)
    ax.plot(b["bin_centre_deg"], b["mean_abs_error_deg"], "o-", color=INK,
            ms=2.6, lw=1.1, zorder=4)
    ax.plot(b["bin_centre_deg"], b["p90_abs_error_deg"], "^--", color=RED,
            ms=2.4, lw=0.9, zorder=3)
    ax.set_ylabel("abs circular error (deg)", fontsize=7.5)
    axt = ax.twinx()
    axt.plot(b["bin_centre_deg"], b["mean_epistemic"], "s:", color=BLUE, ms=2.4,
             lw=1.0, zorder=5)
    axt.set_ylabel("mean epistemic", color=BLUE, fontsize=7.5)
    axt.tick_params(axis="y", labelcolor=BLUE, labelsize=7)
    ax.set_title("held-out population", fontsize=8.5, pad=3)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=GREY, ms=2.2, alpha=0.5,
               label="held-out sample"),
        Line2D([], [], marker="o", ls="-", color=INK, ms=2.6, lw=1.1,
               label=r"bin mean, 15$\degree$"),
        Line2D([], [], marker="^", ls="--", color=RED, ms=2.4, lw=0.9,
               label="p90"),
        Line2D([], [], marker="s", ls=":", color=BLUE, ms=2.4, lw=1.0,
               label="mean epistemic"),
        Patch(facecolor=SHADE_ELEV, lw=0, label="elevated band"),
    ], bbox_to_anchor=(0.5, -0.36), **LEG)

    # ── right: mesh instance, half-degree ────────────────────────────────
    ax = axes[1]
    for lo, hi in DIAGNOSED:
        ax.axvspan(lo, hi, color=SHADE_DIAG, lw=0, zorder=0)
    for lo, hi in PRECAUTIONARY:
        ax.axvspan(lo, hi, facecolor=SHADE_PREC, edgecolor=SHADE_DIAG,
                   hatch="///", lw=0.0, zorder=0)
    order = np.argsort(j["true_angle_deg"])
    ang = j["true_angle_deg"][order]
    err = np.abs(j["circular_error_deg"])[order]
    ax.plot(ang, err, "-", color=GREEN, lw=0.8, zorder=4)
    ax.set_ylabel("abs circular error (deg)", fontsize=7.5)
    axt = ax.twinx()
    axt.plot(ang, j["epistemic"][order], ":", color=BLUE, lw=1.3, zorder=6)
    axt.set_ylabel("epistemic", color=BLUE, fontsize=7.5)
    axt.tick_params(axis="y", labelcolor=BLUE, labelsize=7)
    ax.set_title("mesh instance, 0.5$\\degree$", fontsize=8.5, pad=3)
    ax.legend(handles=[
        Line2D([], [], ls="-", color=GREEN, lw=1.1, label=r"error, 0.5$\degree$"),
        Line2D([], [], ls=":", color=BLUE, lw=1.3, label="epistemic"),
        Patch(facecolor=SHADE_DIAG, lw=0, label="excluded: diagnosed"),
        Patch(facecolor=SHADE_PREC, edgecolor=SHADE_DIAG, hatch="///", lw=0.0,
              label="excluded: precautionary"),
    ], bbox_to_anchor=(0.5, -0.36), **LEG)

    for a in axes:
        a.set_xlim(-180, 180)
        a.set_xticks(range(-180, 181, 90))
        a.tick_params(labelsize=7)
    # one shared x-label beneath both panels
    fig.text(0.495, 0.268, "true angle (deg)", ha="center", fontsize=8)

    FIGS.mkdir(parents=True, exist_ok=True)
    png = FIGS / (name + ".png")
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("  wrote %s   %.2f x %.2f in at dpi 200 -> %d x %d px"
          % (png, WIDTH, height, round(WIDTH * 200), round(height * 200)))

    # panel width in inches, and what the failure window occupies
    l, r = 0.072, 0.938
    panel_w = (r - l - 0.62 * (r - l) / 2.62) * WIDTH / 2.0
    print("  right panel width ~%.2f in -> %.2f px per degree at dpi 200"
          % (panel_w, panel_w * 200 / 360.0))
    print("  a 5-degree feature therefore spans ~%.1f px"
          % (5.0 * panel_w * 200 / 360.0))


if __name__ == "__main__":
    main()
