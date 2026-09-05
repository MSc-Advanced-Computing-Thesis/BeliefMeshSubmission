# Matched routing under BOTH fusion rules -- diagnostic figure and table.
#
# Four arms: Product fusion and Average fusion, each with random and
# uncertainty-guided wearable routing. Grouped by RULE so the comparison the
# eye makes first is guided against random within a rule.
#
# ENCODING. Two visual channels only:
#   colour -> fusion rule      (matching make_fusion_illustration.py)
#   hatch  -> routing policy   (plain = random, hatched = guided)
# The accuracy panel's whole-run / last-50 split is carried by POSITION and a
# label beneath each pair, not by a third channel. An earlier version used
# alpha for it, which drew the last-50 bars faint -- and those are the ones
# carrying the section's result.
#
# THE POINT OF THE THIRD PANEL. Routing widens the interval under BOTH rules.
# Under Product fusion that moves hw:RMS from below the empirical 90%-coverage
# band toward it; under Average fusion it starts above the band and moves
# further away. Both must be legible on one shared axis, which is why the
# y-range is checked against the Product bars rather than left to autoscale.
#
# Style, dimensions and naming follow make_ablation_outputs.fig_routing.
# PNG only, 6.3 in wide, dpi 200. Writes a NEW filename; the two-bar figure
# survives.
#
# Run: python -u experiments/section5_2/make_routing_both_rules.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from averaged_readout import load_run, metrics
from calibration_band import BAND
from calibration_band import draw as _cal_band

LAST = 50
BASE = "runs/chapter5_v2"
FIGS = Path("figures")
OUT = Path("runs/chapter5_v2")
W, DPI = 6.3, 200
INK, RED, GREY = "#22252a", "#c1121f", "#8a8f98"
# same two colours the fusion-behaviours figure uses for the two rules
PROD_COLOR, AVG_COLOR = "#0F6E56", "#E6821B"

ARMS = [
    # tick labels stay SHORT -- "uncertainty guided" is wider than the bar
    # spacing a 4-bar panel allows and collided with its neighbour. The bottom
    # legend spells the policy out in full.
    ("Product fusion", "random", "s5_9_routing_matched", "random", PROD_COLOR, ""),
    ("Product fusion", "guided", "s5_9_routing_matched",
     "uncertainty_guided", PROD_COLOR, "///"),
    ("Average fusion", "random", "s5_9_routing_matched_avgfusion", "random",
     AVG_COLOR, ""),
    ("Average fusion", "guided", "s5_9_routing_matched_avgfusion",
     "uncertainty_guided", AVG_COLOR, "///"),
]
# grouped: two rules, two policies inside each
X = np.array([0.0, 0.82, 2.38, 3.20])


def arm(root, name):
    M = []
    for p in sorted(glob.glob("%s/%s/%s_seed*/manifest.yaml" % (BASE, root, name))):
        d = Path(os.path.dirname(p))
        sd = int(yaml.safe_load(open(p))["env_seed"])
        M.append(metrics(load_run(d, sd), last=LAST)["averaged"])
    if not M:
        raise SystemExit("no runs for %s/%s" % (root, name))
    ms = lambda k: (st.mean([m[k] for m in M]),
                    st.stdev([m[k] for m in M]) if len(M) > 1 else 0.0)
    return dict(n=len(M), whole=ms("whole"), last50=ms("last50"),
                cov=ms("cov"), ratio=ms("ratio"))


A = [(rule, pol, col, hat, arm(root, name))
     for rule, pol, root, name, col, hat in ARMS]

fig, axes = plt.subplots(1, 3, figsize=(W, 2.9))
fig.subplots_adjust(left=0.098, right=0.995, bottom=0.30, top=0.86, wspace=0.44)

# ---- accuracy: eight bars, metric by POSITION not by alpha ---------------
# Within each rule: [whole-run random, whole-run guided] then a gap then
# [last-50 random, last-50 guided], each pair labelled beneath. Colour still
# carries the rule and hatch still carries the routing policy, so no third
# visual channel is needed and the last-50 bars -- which carry the section's
# result -- are drawn at the same weight as the whole-run ones.
ax = axes[0]
GC = np.array([0.0, 1.62])          # rule-group centres
PAIR_OFF, BAR_OFF, BW = 0.40, 0.115, 0.21
for gi, (rule_i, pol_i) in enumerate(((0, 1), (2, 3))):
    for mi, key in enumerate(("whole", "last50")):
        pc = GC[gi] + (mi - 0.5) * 2 * PAIR_OFF
        for bi, ai in enumerate((rule_i, pol_i)):
            a = A[ai]
            ax.bar(pc + (bi - 0.5) * 2 * BAR_OFF, a[4][key][0], BW,
                   yerr=a[4][key][1], capsize=2.2,
                   color=a[2], hatch=a[3], edgecolor="white", lw=0.5)
ax.set_ylabel("cell-space MSE", fontsize=8)
ax.set_title("accuracy", fontsize=9, pad=11)
ax.set_ylim(0, max(a[4]["whole"][0] + a[4]["whole"][1] for a in A) * 1.16)
ax.set_xlim(GC[0] - PAIR_OFF - 0.35, GC[-1] + PAIR_OFF + 0.35)
ax.set_xticks([])
for gi in range(2):
    for mi, lab in enumerate(("whole-run", "last-50")):
        ax.annotate(lab, xy=(GC[gi] + (mi - 0.5) * 2 * PAIR_OFF, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -4), textcoords="offset points",
                    ha="center", va="top", fontsize=5.6)

# ---- coverage and hw:RMS ------------------------------------------------
for ax_, key, lab, title in ((axes[1], "cov", "90% coverage", "coverage"),
                             (axes[2], "ratio", "hw : RMS", "hw : RMS")):
    ax_.bar(X, [a[4][key][0] for a in A], 0.62,
            yerr=[a[4][key][1] for a in A], capsize=2.5,
            color=[a[2] for a in A], hatch=[a[3] for a in A],
            edgecolor="white", lw=0.5)
    if key == "cov":
        ax_.axhline(0.90, color=RED, ls=":", lw=1.0)
        ax_.set_ylim(0, 1.12)
    else:
        _cal_band(ax_, on_top=True)
        top = max(a[4]["ratio"][0] + a[4]["ratio"][1] for a in A) * 1.10
        ax_.set_ylim(0, top)
    ax_.set_ylabel(lab, fontsize=8)
    ax_.set_title(title, fontsize=9, pad=11)

# One tick per RULE, not per bar. At this panel width the two policy labels
# are wider than the bar spacing and collide however they are abbreviated;
# routing is carried by the hatch and named in the legend.
for ax_ in axes[1:]:
    ax_.set_xticks([X[:2].mean(), X[2:].mean()])
    ax_.set_xticklabels(["Product\nfusion", "Average\nfusion"], fontsize=7.4)
    ax_.tick_params(labelsize=8, length=0)
    ax_.set_xlim(X[0] - 0.62, X[-1] + 0.62)
# the accuracy panel has its own pair labels, so its rule row sits lower
for gi, rule in enumerate(("Product\nfusion", "Average\nfusion")):
    axes[0].annotate(rule, xy=(GC[gi], 0), xycoords=("data", "axes fraction"),
                     xytext=(0, -14), textcoords="offset points",
                     ha="center", va="top", fontsize=7.4)

# The dotted line and the shaded band are DIFFERENT objects in different
# panels -- nominal 90% coverage, and the empirical hw:RMS reference. One
# combined entry invited reading the band as a coverage threshold.
# ncol fills COLUMN-major, so the order below is chosen to put rules in the
# first column, routing in the second and the two references in the third
# rather than mixing a reference with a routing policy.
fig.legend(handles=[Patch(facecolor=PROD_COLOR, ec="white", label="Product fusion"),
                    Patch(facecolor=AVG_COLOR, ec="white", label="Average fusion"),
                    Patch(facecolor=GREY, ec="white", label="random routing"),
                    Patch(facecolor=GREY, ec="white", hatch="///",
                          label="uncertainty-guided routing"),
                    Line2D([], [], color=RED, ls=":", lw=1.0,
                           label="90% nominal coverage (centre panel)"),
                    Patch(facecolor=RED, alpha=0.16, ec="none",
                          label="empirical 90%% hw:RMS band %.2f-%.2f (right panel)"
                                % BAND)],
           loc="lower center", ncol=3, fontsize=6.2, frameon=False,
           handlelength=1.5, columnspacing=1.0, bbox_to_anchor=(0.5, -0.005))

FIGS.mkdir(exist_ok=True)
p = FIGS / "ch5_5_9_routing_both_rules.png"
fig.savefig(p, dpi=DPI)
plt.close(fig)
print("wrote %s" % p)

# ---- compression check on the third panel --------------------------------
top = max(a[4]["ratio"][0] + a[4]["ratio"][1] for a in A) * 1.10
pr, pg = A[0][4]["ratio"][0], A[1][4]["ratio"][0]
# ---- accuracy-panel legibility, measured rather than eyeballed -----------
bb = axes[0].get_window_extent()
span = axes[0].get_xlim()[1] - axes[0].get_xlim()[0]
# get_window_extent is in CANVAS pixels (fig.dpi, default 100); the PNG is
# written at DPI. Rescale, or the reported widths understate the real file.
scale = DPI / fig.dpi
px_per_unit = bb.width * scale / span
bb_w = bb.width * scale
print("\nACCURACY-PANEL LEGIBILITY (8 bars, dpi %d)" % DPI)
print("  panel width           %.0f px (%.2f in)" % (bb_w, bb_w / DPI))
print("  bar width             %.1f px" % (BW * px_per_unit))
print("  gap within a pair     %.1f px" % ((2 * BAR_OFF - BW) * px_per_unit))
print("  gap between pairs     %.1f px" % ((2 * PAIR_OFF - 2 * BAR_OFF - BW) * px_per_unit))
print("  gap between rules     %.1f px"
      % (((GC[1] - GC[0]) - 2 * PAIR_OFF - 2 * BAR_OFF - BW) * px_per_unit))

print("\nTHIRD-PANEL COMPRESSION CHECK  (y-axis 0 to %.2f)" % top)
print("  Product random %.3f -> %.1f%% of axis height" % (pr, 100 * pr / top))
print("  Product guided %.3f -> %.1f%% of axis height" % (pg, 100 * pg / top))
print("  gap between the two Product bars: %.3f = %.1f%% of axis height"
      % (pg - pr, 100 * (pg - pr) / top))
print("  calibration band %.2f-%.2f sits at %.1f%%-%.1f%% of axis height"
      % (BAND[0], BAND[1], 100 * BAND[0] / top, 100 * BAND[1] / top))

# ---- table, same shape as the existing one ------------------------------
print("\n" + "=" * 104)
print("MATCHED ROUTING UNDER BOTH FUSION RULES")
print("cell space, AVERAGED NIG readout, last-50 window, 5 seeds, mean +/- sd")
print("=" * 104)
print("%-16s %-20s %3s %-21s %-21s %-15s %-15s"
      % ("fusion rule", "routing", "n", "whole-run MSE", "last-50 MSE",
         "90% coverage", "hw : RMS"))
f = lambda t, p=5: "%.*f +/- %.*f" % (p, t[0], p, t[1])
for rule, pol, col, hat, r in A:
    print("%-16s %-20s %3d %-21s %-21s %-15s %-15s"
          % (rule, pol.replace("\n", " "), r["n"], f(r["whole"]), f(r["last50"]),
             f(r["cov"], 3), f(r["ratio"], 3)))

csv = OUT / "table_5_9_routing_both_rules.csv"
with open(csv, "w", encoding="utf8") as fh:
    fh.write("fusion_rule,routing,n_seeds,whole_run_mse_mean,whole_run_mse_sd,"
             "last50_mse_mean,last50_mse_sd,coverage90_mean,coverage90_sd,"
             "hw_rms_mean,hw_rms_sd\n")
    for rule, pol, col, hat, r in A:
        fh.write("%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n"
                 % (rule, pol.replace("\n", " "), r["n"], r["whole"][0], r["whole"][1],
                    r["last50"][0], r["last50"][1], r["cov"][0], r["cov"][1],
                    r["ratio"][0], r["ratio"][1]))
print("  wrote %s" % csv)
