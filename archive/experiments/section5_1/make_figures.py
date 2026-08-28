# Section 5.1 thesis figures. Generated ENTIRELY from the dumped artefacts
# under runs/section5_1/ -- no model is loaded and no evaluation is run here,
# so a figure can never disagree with the numbers in the tables.
#
# Thesis convention: 6.3-inch width (LaTeX text width) at dpi=200, into
# figures/. PNG only -- no PDF companions (Christian's instruction).
#
# Run: python -u experiments/section5_1/make_figures.py

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import read_dump

RUNS = Path("runs/section5_1")
FIGS = Path("figures")
WIDTH = 6.3
RED, BLUE = "#c1121f", "#1f5fc1"
GREY, INK = "#8a8f98", "#22252a"


def load_csv(path):
    with open(path, newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array(vals, dtype=np.float64)
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def save(fig, name):
    # PNG only, per Christian: no PDF companions for these figures.
    FIGS.mkdir(parents=True, exist_ok=True)
    png = FIGS / (name + ".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote " + str(png))


def fig1_error_distribution():
    d = read_dump(RUNS / "r1a" / "per_sample_baseline.csv")
    dec = load_csv(RUNS / "r1a" / "rms_by_uncertainty_decile.csv")
    err = np.abs(d["circular_error_deg"])
    mean_e = float(np.mean(err))
    rms_e = float(np.sqrt(np.mean(err ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    ax = axes[0]
    ax.hist(err, bins=np.arange(0, 185, 5), color=GREY, edgecolor="white",
            linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("absolute circular error (deg)")
    ax.set_ylabel("held-out samples")
    ax.set_title("Error distribution", fontsize=9)
    ax.axvline(mean_e, color=RED, lw=1.1, ls="-",
               label="mean %.1f deg" % mean_e)
    ax.axvline(rms_e, color=INK, lw=1.1, ls="--",
               label="RMS %.1f deg" % rms_e)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1]
    ax.plot(dec["decile"], dec["rms_error_deg"], "o-", color=INK, ms=3.5,
            lw=1.2, label="RMS")
    ax.plot(dec["decile"], dec["mean_abs_error_deg"], "s--", color=GREY, ms=3,
            lw=1.0, label="mean abs")
    ax.set_xlabel("predictive-uncertainty decile")
    ax.set_ylabel("error (deg)")
    ax.set_xticks(range(1, 11))
    ax.set_title("Error by uncertainty decile", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    for a in axes:
        a.tick_params(labelsize=7)
        a.xaxis.label.set_size(8)
        a.yaxis.label.set_size(8)
    fig.tight_layout()
    save(fig, "section5_1_error_distribution")


def fig2_error_by_angle():
    d = read_dump(RUNS / "r1a" / "per_sample_baseline.csv")
    b = load_csv(RUNS / "f_angle_bins" / "error_by_true_angle.csv")
    excluded = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]

    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    for i, (lo, hi) in enumerate(excluded):
        ax.axvspan(lo, hi, color="#d8dce2", alpha=0.55, lw=0, zorder=0,
                   label="mesh-excluded range" if i == 0 else None)
    ax.scatter(d["true_angle_deg"], np.abs(d["circular_error_deg"]), s=4,
               color=GREY, alpha=0.45, lw=0, zorder=2, label="held-out sample")
    ax.plot(b["bin_centre_deg"], b["mean_abs_error_deg"], "o-", color=INK,
            ms=3.5, lw=1.3, zorder=4, label="bin mean")
    ax.plot(b["bin_centre_deg"], b["p90_abs_error_deg"], "^--", color=RED,
            ms=3, lw=1.0, zorder=3, label="bin p90")
    ax.set_xlabel("true angle (deg)")
    ax.set_ylabel("absolute circular error (deg)")
    ax.set_xlim(-180, 180)
    ax.set_xticks(range(-180, 181, 45))
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)

    ax2 = ax.twinx()
    ax2.plot(b["bin_centre_deg"], b["mean_epistemic"], "s:", color=BLUE, ms=3,
             lw=1.1, zorder=5, label="bin mean epistemic")
    ax2.set_ylabel("mean epistemic uncertainty", color=BLUE, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=BLUE, labelsize=7)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper left",
              ncol=2)
    fig.tight_layout()
    save(fig, "section5_1_error_by_true_angle")


def _band(ax, x, mean, sd, colour, label, marker="o", ls="-"):
    ax.plot(x, mean, marker + ls, color=colour, ms=3, lw=1.2, label=label)
    ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.18, lw=0)


def fig3_colour_sweep():
    s = load_csv(RUNS / "r1b" / "sweep.csv")
    ft = s["filter_type"].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharex=True)
    panels = [("circular_mse_norm", "circular MSE (normalised)", "MSE"),
              ("mean_epistemic", "mean epistemic", "Epistemic"),
              ("mean_aleatoric", "mean aleatoric", "Aleatoric")]
    for ax, (key, ylab, title) in zip(axes, panels):
        for name, colour in (("red", RED), ("blue", BLUE)):
            m = ft == name
            order = np.argsort(s["w"][m])
            _band(ax, s["w"][m][order], s[key + "_mean"][m][order],
                  s[key + "_sd"][m][order], colour, name)
        ax.set_xlabel("blend weight w")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "section5_1_colour_sweep")


def fig4_offset_sweep():
    s = load_csv(RUNS / "r1c" / "sweep.csv")
    order = np.argsort(s["offset_deg"])
    x = s["offset_deg"][order]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharex=True)

    _band(axes[0], x, s["circular_mse_norm_mean"][order],
          s["circular_mse_norm_sd"][order], INK, "MSE")
    axes[0].set_ylabel("circular MSE (normalised)")
    axes[0].set_title("MSE", fontsize=9)

    _band(axes[1], x, s["mean_epistemic_mean"][order],
          s["mean_epistemic_sd"][order], BLUE, "epistemic")
    _band(axes[1], x, s["mean_aleatoric_mean"][order],
          s["mean_aleatoric_sd"][order], RED, "aleatoric", marker="s", ls="--")
    axes[1].set_ylabel("uncertainty")
    axes[1].set_title("Uncertainty components", fontsize=9)
    axes[1].legend(fontsize=6.5, frameon=False)

    _band(axes[2], x, s["coverage_90_mean"][order], s["coverage_90_sd"][order],
          INK, "empirical")
    axes[2].axhline(0.90, color=RED, lw=1.0, ls=":", label="nominal 90%")
    axes[2].set_ylabel("90% interval coverage")
    axes[2].set_title("Coverage", fontsize=9)
    axes[2].set_ylim(0, 1.02)
    axes[2].legend(fontsize=6.5, frameon=False)

    for ax in axes:
        ax.set_xlabel("target offset (deg)")
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    fig.tight_layout()
    save(fig, "section5_1_offset_sweep")


def fig5_width_table():
    s = load_csv(RUNS / "r1d" / "width_table.csv")
    order = np.argsort(s["n_params"])

    def fmt(m, sd, p=4):
        return ("%." + str(p) + "f +/- %." + str(p) + "f") % (m, sd)

    header = ["variant", "params", "circular MSE", "RMS err\n(deg)",
              "mean abs err\n(deg)", "r\n(epi vs |err|)", "90%\ncoverage"]
    col_widths = [0.10, 0.11, 0.20, 0.16, 0.16, 0.15, 0.16]
    body = []
    for i in order:
        body.append([
            str(s["variant"][i]),
            "{:,}".format(int(s["n_params"][i])),
            fmt(s["circular_mse_norm_mean"][i], s["circular_mse_norm_sd"][i], 5),
            fmt(s["rms_error_deg_mean"][i], s["rms_error_deg_sd"][i], 2),
            fmt(s["mean_abs_error_deg_mean"][i], s["mean_abs_error_deg_sd"][i], 2),
            fmt(s["manifest_convention_pearson_r_mean"][i],
                s["manifest_convention_pearson_r_sd"][i], 3),
            fmt(s["coverage_90_mean"][i], s["coverage_90_sd"][i], 4),
        ])
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=header, cellLoc="center",
                   loc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.9)
    tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#c9ced6")
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(INK)
    fig.tight_layout()
    save(fig, "section5_1_width_variants")


if __name__ == "__main__":
    print("Section 5.1 figures:")
    fig1_error_distribution()
    fig2_error_by_angle()
    fig3_colour_sweep()
    fig4_offset_sweep()
    fig5_width_table()
