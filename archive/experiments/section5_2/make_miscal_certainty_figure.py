# Miscalibrated-wearable certainty DROP map, seed 42, last-50 mean.
#
# Single panel: baseline certainty MINUS miscalibrated certainty, so positive
# (warm) = certainty LOST under miscalibration. Diverging scale centred at zero
# and set by the range of the difference itself.
#
# WHY NOT TWO ABSOLUTE PANELS. Measured: per-cell certainty spans 0.911-0.997
# across the two runs, a range set by a handful of low-coverage periphery cells
# that have nothing to do with the wearable. The effect to be shown is ~0.007,
# under 10% of that span, so on a shared absolute scale the travelled region
# saturates identically in both panels. Filled path markers made it worse by
# covering the cells they annotate.
#
# HONEST CAVEAT, printed by this script. The near-path mean drop (+0.0074) is
# 3.4x the far-field mean (+0.0022), but the single LARGEST drop sits 8 cells
# from the path in the low-coverage corner. The spatial signal is real in the
# mean and noisy per cell; the darkest cell is not the affected region.
#
# Run: python -u experiments/section5_2/make_miscal_certainty_figure.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from averaged_readout import averaged_params

LAST, G = 50, 22
W, DPI = 6.3, 200
RED, GREY = "#c1121f", "#8a8f98"
FIGS = Path("figures")
MIS = Path("runs/chapter5_v2/s5_miscal_wearable/miscal_seed42")
BASE = Path("runs/chapter5_v2/s531_b2/nig_product_sampled_seed42")
BAD_W, NEAR, FAR = 2, 2.0, 6.0


def cert_map(d):
    bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(d / "cell_ncov_steps.npy").astype(int)
    _, nu, al, be = averaged_params(bel, nc)
    epi = be / (np.maximum(nu, 1e-6) * np.maximum(al - 1.0, 1e-6))
    return np.nanmean((1.0 / (1.0 + np.minimum(epi, 10.0)))[-LAST:], axis=0)


DROP = cert_map(BASE) - cert_map(MIS)      # positive = certainty lost
paths = np.load(MIS / "realised_wearable_paths.npy")
bad = np.unique(paths[BAD_W].reshape(-1, 2), axis=0).astype(float)
rr, cc = np.mgrid[0:G, 0:G]
allc = np.stack([rr.ravel(), cc.ravel()], 1).astype(float)
D = np.linalg.norm(allc[:, None, :] - bad[None, :, :], axis=2).min(1).reshape(G, G)

v = float(np.nanmax(np.abs(DROP)))
near, far = float(np.nanmean(DROP[D < NEAR])), float(np.nanmean(DROP[D >= FAR]))
i = np.unravel_index(np.nanargmax(DROP), DROP.shape)

print("CERTAINTY DROP (baseline - miscalibrated), seed 42, last-50 mean")
print("  scale: +/-%.5f, symmetric, set by the difference's own range" % v)
print("  near-path (<%.0f cells) mean drop %+.5f = %.0f%% of half-range"
      % (NEAR, near, 100 * near / v))
print("  far-field (>=%.0f cells) mean drop %+.5f = %.0f%% of half-range"
      % (FAR, far, 100 * far / v))
print("  near / far ratio %.1fx" % (near / far))
print("  LARGEST single-cell drop %+.5f at %s -- %.1f cells from the path"
      % (DROP[i], i, D[i]))

fig, ax = plt.subplots(figsize=(W, 3.4))
fig.subplots_adjust(left=0.105, right=0.80, bottom=0.145, top=0.915)
im = ax.imshow(DROP, cmap="RdBu_r", vmin=-v, vmax=v, origin="upper",
               interpolation="nearest")

# other two wearables: faint, so they are present but do not compete
for wi in range(paths.shape[0]):
    if wi == BAD_W:
        continue
    u = np.unique(paths[wi].reshape(-1, 2), axis=0)
    ax.plot(u[:, 1], u[:, 0], "s", ms=3.0, mfc="none", mec=GREY, mew=0.5,
            ls="none", alpha=0.45, zorder=3)
# miscalibrated wearable: OUTLINE only, so the cell beneath stays readable
u = np.unique(paths[BAD_W].reshape(-1, 2), axis=0)
ax.plot(u[:, 1], u[:, 0], "s", ms=4.2, mfc="none", mec=RED, mew=0.9,
        ls="none", zorder=5)

ax.set_title("certainty lost under miscalibration (seed 42, last-50 mean)",
             fontsize=9)
ax.set_xticks([0, 7, 14, 21]); ax.set_yticks([0, 7, 14, 21])
ax.tick_params(labelsize=7.5)
ax.set_xlabel("cell column", fontsize=8)
ax.set_ylabel("cell row", fontsize=8)

cax = fig.add_axes([0.815, 0.145, 0.028, 0.77])
cb = fig.colorbar(im, cax=cax)
cb.set_label("baseline $-$ miscalibrated certainty", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
cb.ax.axhline(0.0, color="#444444", lw=0.6)

fig.legend(handles=[Line2D([], [], marker="s", ls="none", ms=4.2, mfc="none",
                           mec=RED, mew=0.9,
                           label="miscalibrated wearable (+20 deg)"),
                    Line2D([], [], marker="s", ls="none", ms=3.0, mfc="none",
                           mec=GREY, mew=0.5, alpha=0.7,
                           label="other wearables")],
           loc="lower center", ncol=2, frameon=False, fontsize=7,
           bbox_to_anchor=(0.45, -0.008))

FIGS.mkdir(exist_ok=True)
p = FIGS / "ch5_5_miscal_certainty_drop.png"
fig.savefig(p, dpi=DPI); plt.close(fig)
print("\nwrote %s" % p)
