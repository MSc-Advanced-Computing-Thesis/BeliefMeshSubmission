# Section 5.1 combined angle-response figure, stacked: population (F) above the
# single mesh instance (J). Both panels keep their full content -- sample
# scatter, error curves, and the epistemic-uncertainty twin axis, which is the
# result the figure exists to show.
#
# Built ENTIRELY from dumped CSVs (runs/section5_1/f_angle_bins/ and
# runs/section5_1/j_single_instance/per_render.csv). No model, no evaluation.
#
# Resolution differs by panel on purpose:
#   TOP     15-degree bins -- 1,253 held-out samples spread over the circle,
#           so ~52 samples per bin; that is the resolution the data supports.
#   BOTTOM  native half-degree, NOT binned. The instance's failure window is
#           ~5 degrees wide (+126.5 to +131.5) and 15-degree binning smears it
#           into a single bar.
#
# NO p90 ON THE BOTTOM PANEL. At half-degree spacing there is one render per
# angle, so any p90 there has to come from a rolling window -- and a rolling
# p90 is a local envelope across ONE instance at nearby angles, whereas the top
# panel's p90 is a population tail across ~52 DIFFERENT instances at similar
# angles. Sharing the name invites a comparison the quantities do not support,
# and the 5-degree window needed to make it a real order statistic widened the
# apparent feature from 5 degrees to 9, blunting the one thing this panel
# exists to show. The raw error curve and its maximum carry the argument, in
# prose. With no second p90 to disambiguate against, the top panel's entry is
# simply "p90".
#
# SHADING DIFFERS BY PANEL, and the two are different KINDS of object:
#   TOP     a MEASURED finding -- the band this analysis identifies as
#           anomalous (+90 to +135), where mean absolute error runs 3.20x and
#           mean epistemic uncertainty 11.8x the rest of the circle. The mesh
#           exclusion ranges are deliberately NOT drawn here: they were never
#           applied to this population, in training or evaluation.
#   BOTTOM  a DESIGN DECISION -- the rotation ranges the mesh experiments
#           actually excluded from sampling. Diagnosed (120-150) and
#           precautionary (+/-165-180) stay visually distinct so the figure
#           does not imply equal evidential status.
# The legend labels say which is which.
#
# Independent y-ranges throughout: the instance sits in the TRAINING partition
# and the population is held out, so their error LEVELS are not comparable.
#
# PNG only (Christian's instruction), 6.3 in wide, dpi 200.
#
# Run: python -u experiments/section5_1/make_angle_response_stacked.py

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyse import read_dump
from figure_sweeps import BLUE, FIGS, GREY, INK, RED, WIDTH, load_csv

# The bottom panel's raw error curve gets its OWN colour rather than reusing
# the top panel's ink: they are different quantities (a 15-degree bin mean vs
# a single render's error) and sharing a colour implied they were the same
# mark at two resolutions.
GREEN = "#1b7a3e"

HEIGHT = 5.0
NAME = "section5_1_angle_response"

# Top panel: the empirically elevated band (this analysis's own finding).
ELEVATED = (90.0, 135.0)
SHADE_ELEV = "#f4cfcf"

# Bottom panel: what the mesh runs actually excluded.
DIAGNOSED = [(120.0, 150.0)]
PRECAUTIONARY = [(165.0, 180.0), (-180.0, -165.0)]
SHADE_DIAG = "#c9ced6"
SHADE_PREC = "#eceef1"


# One legend per panel, vertically centred beside its own panel: with the
# marks differing between panels, a shared legend had to be grouped and
# labelled to say which applied where, which a per-panel legend states by
# position alone.
LEGEND_KW = dict(fontsize=6.8, frameon=False, loc="center left",
                 bbox_to_anchor=(1.135, 0.5), handlelength=1.9,
                 handletextpad=0.6, labelspacing=0.65, borderpad=0.0)


def main(height: float = HEIGHT, name: str = NAME):
    b = load_csv(Path(ART + "/5.1_baseline_model_and_calibration/f_angle_bins/error_by_true_angle.csv"))
    pop = read_dump(Path(ART + "/5.1_baseline_model_and_calibration/r1a_held_out_evaluation/per_sample_baseline.csv"))
    j = read_dump(Path(ART + "/5.1_baseline_model_and_calibration/j_single_instance/per_render.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, height), sharex=True)
    # Right margin pulled in for the two per-panel legends, which sit outside
    # the axes beyond the twin-axis tick labels and y-label.
    fig.subplots_adjust(left=0.090, right=0.700, bottom=0.085, top=0.985,
                        hspace=0.11)

    # ── top: population, 15-degree bins, measured elevated band in red ────
    ax = axes[0]
    ax.axvspan(*ELEVATED, color=SHADE_ELEV, lw=0, zorder=0)
    # Small and faint: these are context for the binned curves, not a mark to
    # be read individually, and at s=4/alpha=0.45 they competed with the mean.
    ax.scatter(pop["true_angle_deg"], np.abs(pop["circular_error_deg"]), s=2.2,
               color=GREY, alpha=0.28, lw=0, zorder=2)
    ax.plot(b["bin_centre_deg"], b["mean_abs_error_deg"], "o-", color=INK,
            ms=3.5, lw=1.3, zorder=4)
    ax.plot(b["bin_centre_deg"], b["p90_abs_error_deg"], "^--", color=RED,
            ms=3, lw=1.0, zorder=3)
    ax.set_ylabel("absolute circular error (deg)", fontsize=8)
    axt = ax.twinx()
    axt.plot(b["bin_centre_deg"], b["mean_epistemic"], "s:", color=BLUE, ms=3,
             lw=1.1, zorder=5)
    axt.set_ylabel("mean epistemic uncertainty", color=BLUE, fontsize=8)
    axt.tick_params(axis="y", labelcolor=BLUE, labelsize=8)
    axt.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=GREY, ms=2.6, alpha=0.5,
               label="held-out sample"),
        Line2D([], [], marker="o", ls="-", color=INK, ms=3.5, lw=1.3,
               label=r"bin mean, 15$\degree$"),
        Line2D([], [], marker="^", ls="--", color=RED, ms=3, lw=1.0,
               label="p90"),
        Line2D([], [], marker="s", ls=":", color=BLUE, ms=3, lw=1.1,
               label="mean epistemic"),
        Patch(facecolor=SHADE_ELEV, lw=0,
              label="elevated error band\n(measured)"),
    ], **LEGEND_KW)

    # ── bottom: mesh instance, half-degree, mesh exclusions shaded ────────
    ax = axes[1]
    for lo, hi in DIAGNOSED:
        ax.axvspan(lo, hi, color=SHADE_DIAG, lw=0, zorder=0)
    for lo, hi in PRECAUTIONARY:
        ax.axvspan(lo, hi, facecolor=SHADE_PREC, edgecolor=SHADE_DIAG,
                   hatch="///", lw=0.0, zorder=0)
    order = np.argsort(j["true_angle_deg"])
    ang = j["true_angle_deg"][order]
    err = np.abs(j["circular_error_deg"])[order]
    # No scatter: at half-degree spacing the 720 render points fall entirely
    # under the error line and never show. The line IS the renders.
    ax.plot(ang, err, "-", color=GREEN, lw=0.9, zorder=4)
    ax.set_ylabel("absolute circular error (deg)", fontsize=8)
    axt = ax.twinx()
    # The two curves very nearly coincide through the failure window -- that
    # coincidence IS the result -- so the epistemic trace is drawn heavier and
    # on top, otherwise it disappears under the error curve exactly where it
    # matters most.
    axt.plot(ang, j["epistemic"][order], ":", color=BLUE, lw=1.5, zorder=6)
    axt.set_ylabel("epistemic uncertainty", color=BLUE, fontsize=8)
    axt.tick_params(axis="y", labelcolor=BLUE, labelsize=8)
    axt.legend(handles=[
        Line2D([], [], ls="-", color=GREEN, lw=1.3,
               label=r"error, 0.5$\degree$"),
        Line2D([], [], ls=":", color=BLUE, lw=1.5, label="epistemic"),
        Patch(facecolor=SHADE_DIAG, lw=0,
              label="excluded from mesh\nruns: diagnosed"),
        Patch(facecolor=SHADE_PREC, edgecolor=SHADE_DIAG, hatch="///", lw=0.0,
              label="excluded from mesh\nruns: precautionary"),
    ], **LEGEND_KW)

    for a in axes:
        a.set_xlim(-180, 180)
        a.set_xticks(range(-180, 181, 45))
        a.tick_params(labelsize=8)
    axes[1].set_xlabel("true angle (deg)", fontsize=8)

    FIGS.mkdir(parents=True, exist_ok=True)
    png = FIGS / (name + ".png")
    # No bbox_inches="tight": it crops to ink and silently changes the on-page
    # size, which would defeat checking legibility at the declared dimensions.
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("  wrote " + str(png))
    return png


if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else HEIGHT
    main(h)
