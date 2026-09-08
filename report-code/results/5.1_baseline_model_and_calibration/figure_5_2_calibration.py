# Section 5.1 calibration figure: held-out error distribution beside Student-t
# interval coverage.
#
# Built ENTIRELY from dumped CSVs (runs/section5_1/r1a/). No model, no
# evaluation. The error-by-uncertainty-decile panel is deliberately absent --
# those numbers are quoted in prose.
#
# The right panel is drawn with an EQUAL aspect ratio and identical x/y limits,
# so the identity diagonal sits at a true 45 degrees and any departure from it
# is read at face value rather than exaggerated or flattened by unequal scaling.
# That squares the axes box, which is why the panel widths differ.
#
# PNG only (Christian's instruction), 6.3 in wide, dpi 200.
#
# Run: python -u experiments/section5_1/make_calibration_figure.py

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
import yaml

from analyse import read_dump
from figure_sweeps import FIGS, GREY, INK, RED, WIDTH, load_csv

HEIGHT = 2.7
NAME = "fig_5_2_calibration"
LIM = (0.0, 1.0)       # the fine grid runs 0.05-0.95
MARKED = [0.50, 0.80, 0.90, 0.95]   # the four originally reported levels


def main(height: float = HEIGHT, name: str = NAME):
    d = read_dump(Path(ART + "/5.1_baseline_model_and_calibration/r1a_held_out_evaluation/per_sample_baseline.csv"))
    cov = load_csv(Path(ART + "/5.1_baseline_model_and_calibration/r1a_held_out_evaluation/interval_coverage_fine.csv"))
    # Headline error statistics come from the 20-SEED protocol, not from the
    # single pinned pass the histogram draws. The thesis quotes the 20-seed
    # numbers, so the marked lines must be those; the histogram illustrates the
    # shape of the distribution from one representative pass. The +/- sd bands
    # make the difference explicit rather than implying the lines are
    # properties of the plotted bars.
    ref = yaml.safe_load(open(ART + "/5.1_baseline_model_and_calibration/r1a_held_out_evaluation/summary.yaml"))["reference_passes"]
    mae, mae_sd = ref["mean_abs_error_deg"]["mean"], ref["mean_abs_error_deg"]["sd"]
    rms, rms_sd = ref["rms_error_deg"]["mean"], ref["rms_error_deg"]["sd"]

    # The right panel's axes box is squared by the equal aspect, so it needs
    # less horizontal room than the histogram; the width ratio hands the slack
    # to the left panel instead of leaving it as dead margin.
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.20, top=0.96,
                        wspace=0.26)

    # ── left: held-out absolute error distribution ────────────────────────
    err = np.abs(d["circular_error_deg"])
    ax = axes[0]
    ax.hist(err, bins=np.arange(0, 185, 5), color=GREY, edgecolor="white",
            linewidth=0.4)
    ax.set_yscale("log")
    ax.axvspan(mae - mae_sd, mae + mae_sd, color=RED, alpha=0.13, lw=0)
    ax.axvline(mae, color=RED, lw=1.1, ls="-",
               label=r"mean abs %.2f $\pm$ %.2f$\degree$" % (mae, mae_sd))
    ax.axvspan(rms - rms_sd, rms + rms_sd, color=INK, alpha=0.10, lw=0)
    ax.axvline(rms, color=INK, lw=1.1, ls="--",
               label=r"RMS %.2f $\pm$ %.2f$\degree$" % (rms, rms_sd))
    ax.set_xlabel("absolute circular error (deg)", fontsize=8)
    ax.set_ylabel("held-out samples", fontsize=8)
    ax.legend(fontsize=7, frameon=False)

    # ── right: Student-t interval coverage vs nominal ─────────────────────
    ax = axes[1]
    ax.plot(LIM, LIM, ls=":", color=GREY, lw=1.0, zorder=1,
            label="perfect calibration")
    order = np.argsort(cov["nominal"])
    nom, emp = cov["nominal"][order], cov["empirical"][order]
    ax.plot(nom, emp, "-", color=INK, lw=1.3, zorder=3, label="empirical")
    # Markers only at the four originally reported levels; the rest of the grid
    # is carried by the line so the curve stays readable.
    keep = np.isin(np.round(nom, 2), MARKED)
    ax.plot(nom[keep], emp[keep], "o", color=INK, ms=4.5, zorder=4)
    # No vacuous region is shaded: the largest share of intervals spanning the
    # whole circle is 0.8% (at nominal 0.95), below the 1% threshold in
    # run_coverage_grid.py, so every level plotted is a real measurement.
    assert cov["frac_interval_covers_circle"].max() <= 0.01
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect("equal", adjustable="box")
    # Regular 0.2 ticks rather than one per nominal level: adjacent grid levels
    # are 0.05 apart and their labels would overprint. The four reported levels
    # are identified by their markers, not by tick position.
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("nominal coverage", fontsize=8)
    ax.set_ylabel("empirical coverage", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    for a in axes:
        a.tick_params(labelsize=8)

    FIGS.mkdir(parents=True, exist_ok=True)
    png = FIGS / (name + ".png")
    # No bbox_inches="tight": it crops to ink and silently changes the on-page
    # size, defeating a legibility check at the declared dimensions.
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("  wrote " + str(png))
    return png


if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else HEIGHT
    main(h)
