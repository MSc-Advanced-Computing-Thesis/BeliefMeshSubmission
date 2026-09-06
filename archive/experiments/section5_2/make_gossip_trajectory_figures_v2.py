# Section 5.4 parameter-exchange trajectories, v2.
#
#   ch5_5_4_gossip_trajectories_shared.png  -- two panels (field0, field3) on a
#       SHARED right-hand offset axis, so the two environments' field amplitudes
#       are directly comparable
#   ch5_5_4_gossip_trajectories_all.png     -- all five environments, appendix
#
# The shared offset axis is set from the smoothed traces of ALL FIVE fields
# (5.67 to 32.13 deg), rounded out to 5-35, so the same axis serves both
# figures and nothing is clipped in either.
#
# LAYOUT for the five-panel version: 3 rows x 2 columns, not 1x5 or 2x3.
# Each panel carries a left MSE axis AND a right twin offset axis, and the
# two-panel figure already needed wspace=0.62 to keep those from colliding at
# two columns. Three columns at 6.3 in would leave ~1.7 in per panel and make
# that collision unavoidable; one row of five is worse still at ~1.1 in.
# Two columns keeps panels at ~2.5 in and only the outer edges need labels.
#
# Run: python -u experiments/section5_2/make_gossip_trajectory_figures_v2.py

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from arm_names import name as arm_name
from averaged_readout import load_run
from stage6_spatial_mesh.offset_field_variants import build_field

W, DPI, G, T = 6.3, 200, 22, 390
SMOOTH = 15
INK, RED, BLUE, GREEN = "#22252a", "#c1121f", "#1f5fc1", "#1b7a3e"
OFFC = "#9aa0a8"
ROOT = Path("runs/chapter5_v2/s5_5_gossip_multifield")
FIELDS = ["field0_original", "field1_seed11", "field2_seed23",
          "field3_seed37", "field4_seed51_dual"]
TITLE = {"field0_original": "field0 -- parity holds",
         "field1_seed11": "field1", "field2_seed23": "field2",
         "field3_seed37": "field3 -- gossip leads",
         "field4_seed51_dual": "field4 -- dual obstruction"}
MODES = [("fusion", INK, "-"), ("gossip_uniform", RED, "--"),
         ("gossip_weighted", BLUE, "-."), ("fedavg_global", GREEN, ":")]
OFF_LIM = (5.0, 35.0)          # shared, covers all five smoothed traces
# Legend order is independent of draw order: the arms are plotted with
# Product fusion first so it sits under the others, but listed last.
LEGEND_ORDER = ["fedavg_global", "gossip_uniform", "gossip_weighted", "fusion"]


def ordered(ax):
    h, l = ax.get_legend_handles_labels()
    by = dict(zip(l, h))
    labs = [arm_name(m) for m in LEGEND_ORDER]
    return [by[x] for x in labs], labs

sm = lambda y: np.convolve(y, np.ones(SMOOTH) / SMOOTH, mode="valid")


def trace(field, mode):
    per_seed = []
    for p in sorted(glob.glob(str(ROOT / field / ("gossip_cmp_%s*" % mode)
                                  / "manifest.yaml"))):
        nm = os.path.basename(os.path.dirname(p))
        if nm.replace("gossip_cmp_", "").rsplit("_seed", 1)[0] != mode:
            continue
        d = Path(os.path.dirname(p))
        sd = int(yaml.safe_load(open(p))["env_seed"])
        with np.errstate(invalid="ignore"):
            per_seed.append(np.nanmean(load_run(d, sd)["averaged"][0], axis=(1, 2)))
    return np.nanmean(np.stack(per_seed), axis=0)


def offset_trace(field):
    return np.abs(np.asarray(build_field(field, G, T))).mean(axis=(1, 2))


def panel(ax, field, *, ylab, rlab, xlab):
    o = sm(offset_trace(field))
    axt = ax.twinx()
    axt.plot(np.arange(len(o)) + SMOOTH // 2, o, "--", color=OFFC, lw=0.9, zorder=0)
    axt.set_ylim(*OFF_LIM)
    axt.tick_params(axis="y", labelsize=6.8, labelcolor="#7c828a")
    if rlab:
        axt.set_ylabel("mean |offset| (deg)", fontsize=7.2, color="#7c828a",
                       labelpad=2)
    else:
        axt.set_yticklabels([])
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    for mode, col, ls in MODES:
        s = sm(trace(field, mode))
        ax.plot(np.arange(len(s)) + SMOOTH // 2, s, ls, color=col, lw=1.2,
                label=arm_name(mode), zorder=3)
    ax.set_yscale("log")
    # field4 spans well under a decade, and matplotlib then labels the log
    # MINOR ticks in scientific notation -- which rendered as a column of
    # repeated "x10^-2". Keep the minor ticks as unlabelled marks.
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0),
                                          numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(TITLE[field], fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    if ylab:
        ax.set_ylabel("cell-space MSE", fontsize=8)
    if xlab:
        ax.set_xlabel("timestep", fontsize=8)
    return o


Path("figures").mkdir(exist_ok=True)
clip = []

# ---------------------------------------------------------------- 2 panels
fig, axes = plt.subplots(1, 2, figsize=(W, 2.9))
fig.subplots_adjust(left=0.082, right=0.918, bottom=0.255, top=0.885, wspace=0.62)
for ax, f in zip(axes, ["field0_original", "field3_seed37"]):
    o = panel(ax, f, ylab=True, rlab=True, xlab=True)
    if o.min() < OFF_LIM[0] or o.max() > OFF_LIM[1]:
        clip.append(f)
h, l = ordered(axes[0])
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7,
           handlelength=2.0, columnspacing=1.4, bbox_to_anchor=(0.5, 0.035))
p2 = Path("figures/ch5_5_4_gossip_trajectories_shared.png")
fig.savefig(p2, dpi=DPI); plt.close(fig)
print("wrote %s" % p2)

# ---------------------------------------------------------------- 5 panels
H5 = 5.6
fig, axes = plt.subplots(3, 2, figsize=(W, H5))
fig.subplots_adjust(left=0.088, right=0.912, bottom=0.115, top=0.945,
                    wspace=0.60, hspace=0.55)
flat = axes.ravel()
for i, f in enumerate(FIELDS):
    ax = flat[i]
    o = panel(ax, f, ylab=(i % 2 == 0), rlab=(i % 2 == 1), xlab=(i >= 3))
    if o.min() < OFF_LIM[0] or o.max() > OFF_LIM[1]:
        clip.append(f)
flat[5].axis("off")          # 5 environments in a 6-cell grid
h, l = ordered(flat[0])
flat[5].legend(h, l, loc="center", frameon=False, fontsize=7.5,
               handlelength=2.2, labelspacing=0.9)
p5 = Path("figures/ch5_5_4_gossip_trajectories_all.png")
fig.savefig(p5, dpi=DPI); plt.close(fig)
print("wrote %s (%.1f x %.1f in)" % (p5, W, H5))

print("\nSHARED OFFSET AXIS: %.0f to %.0f deg" % OFF_LIM)
print("  smoothed extremes across all five fields: 5.67 to 32.13 deg")
print("  fields clipped by this range: %s" % (sorted(set(clip)) or "none"))
