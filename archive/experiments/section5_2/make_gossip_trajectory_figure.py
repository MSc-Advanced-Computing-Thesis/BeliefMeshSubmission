# Section 5.4 parameter-exchange trajectories: cell-space MSE against timestep.
#
# Two panels, one per environment: field0_original (where last-50 parity holds)
# and field3_seed37 (where the gossip arms lead by ~20%), so the difference
# between them is visible rather than asserted.
#
# Treatment matches the peer-supervision figure (make_block2_figures.fig_52):
# log MSE axis, 15-step moving average, and the environment's mean |offset| as
# a thin dashed grey trace on a right-hand axis. Here the offset is averaged
# over the WHOLE field, not a single node's FOV, because every arm covers the
# whole grid.
#
# Also prints the diagnostics the section needs: whether the belief-exchange /
# gossip gap is constant or concentrated, and whether it tracks the field's
# level or its rate of change.
#
# Run: python -u experiments/section5_2/make_gossip_trajectory_figure.py

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
INK, RED, BLUE, GREY = "#22252a", "#c1121f", "#1f5fc1", "#8a8f98"
GREEN = "#1b7a3e"
ROOT = Path("runs/chapter5_v2/s5_5_gossip_multifield")
PANELS = [("field0_original", "field0 -- parity holds"),
          ("field3_seed37", "field3 -- gossip leads")]
MODES = [("fusion", INK, "-"), ("gossip_uniform", RED, "--"),
         ("gossip_weighted", BLUE, "-."), ("fedavg_global", GREEN, ":")]


def trace(field, mode):
    """Per-timestep cell-space MSE under the averaged readout, seed-averaged."""
    per_seed = []
    for p in sorted(glob.glob(str(ROOT / field / ("gossip_cmp_%s*" % mode)
                                  / "manifest.yaml"))):
        nm = os.path.basename(os.path.dirname(p))
        if nm.replace("gossip_cmp_", "").rsplit("_seed", 1)[0] != mode:
            continue
        d = Path(os.path.dirname(p))
        sd = int(yaml.safe_load(open(p))["env_seed"])
        e2 = load_run(d, sd)["averaged"][0]          # (T, G, G)
        with np.errstate(invalid="ignore"):
            per_seed.append(np.nanmean(e2, axis=(1, 2)))
    return np.nanmean(np.stack(per_seed), axis=0), len(per_seed)


def offset_trace(field):
    f = np.asarray(build_field(field, G, T))
    return np.abs(f).mean(axis=(1, 2))


sm = lambda y: np.convolve(y, np.ones(SMOOTH) / SMOOTH, mode="valid")

fig, axes = plt.subplots(1, 2, figsize=(W, 2.9))
# wspace has to clear BOTH a right-hand twin-axis label on the left panel
# and the left-hand y-label of the right panel, so it is wider than usual
fig.subplots_adjust(left=0.082, right=0.918, bottom=0.30, top=0.885, wspace=0.62)

DIAG = {}
for ax, (field, title) in zip(axes, PANELS):
    off = offset_trace(field)
    axt = ax.twinx()
    o = sm(off)
    axt.plot(np.arange(len(o)) + SMOOTH // 2, o, "--", color="#9aa0a8", lw=0.9,
             zorder=0)
    axt.tick_params(axis="y", labelsize=7, labelcolor="#7c828a")
    axt.set_ylabel("mean |offset| (deg)", fontsize=7.2, color="#7c828a",
                   labelpad=2)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    curves = {}
    for mode, col, ls in MODES:
        y, n = trace(field, mode)
        curves[mode] = y
        s = sm(y)
        ax.plot(np.arange(len(s)) + SMOOTH // 2, s, ls, color=col, lw=1.25,
                label=arm_name(mode), zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("timestep", fontsize=8)
    ax.set_ylabel("cell-space MSE", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    DIAG[field] = (curves, off)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7,
           handlelength=2.0, columnspacing=1.4, bbox_to_anchor=(0.5, -0.01))
Path("figures").mkdir(exist_ok=True)
p = Path("figures/ch5_5_4_gossip_trajectories.png")
fig.savefig(p, dpi=DPI)
plt.close(fig)
print("wrote %s" % p)

# ---- diagnostics -------------------------------------------------------
print("\n" + "=" * 96)
print("IS THE GAP CONSTANT, OR CONCENTRATED?")
print("gap(t) = mean gossip MSE - belief-exchange MSE, positive = fusion better")
print("=" * 96)
for field, _ in PANELS:
    curves, off = DIAG[field]
    gos = np.nanmean(np.stack([curves["gossip_uniform"],
                               curves["gossip_weighted"]]), axis=0)
    gap = gos - curves["fusion"]
    rel = gap / curves["fusion"]
    q = [(0, 97), (97, 195), (195, 292), (292, 390)]
    print("\n--- %s" % field)
    print("  %-14s %14s %14s" % ("window", "mean gap", "gap / fusion"))
    for a, b in q:
        print("  steps %3d-%3d %+14.5f %13.1f%%"
              % (a, b, np.nanmean(gap[a:b]), 100 * np.nanmean(rel[a:b])))
    print("  whole run    %+14.5f %13.1f%%"
          % (np.nanmean(gap), 100 * np.nanmean(rel)))
    sign = np.sign(gap[np.isfinite(gap)])
    print("  timesteps where fusion is ahead: %d of %d (%.0f%%)"
          % (int((sign > 0).sum()), sign.size, 100 * (sign > 0).mean()))

print("\n" + "=" * 96)
print("DOES THE GAP TRACK THE FIELD?")
print("=" * 96)
print("  %-20s %16s %18s" % ("field", "r(gap, |offset|)", "r(gap, d|offset|/dt)"))
for field, _ in PANELS:
    curves, off = DIAG[field]
    gos = np.nanmean(np.stack([curves["gossip_uniform"],
                               curves["gossip_weighted"]]), axis=0)
    gap = gos - curves["fusion"]
    rate = np.abs(np.gradient(off))
    m = np.isfinite(gap)
    r1 = np.corrcoef(gap[m], off[m])[0, 1]
    r2 = np.corrcoef(gap[m], rate[m])[0, 1]
    print("  %-20s %+16.3f %+18.3f" % (field, r1, r2))
print("\n  |offset| is the field's LEVEL; d|offset|/dt its RATE OF CHANGE.")
