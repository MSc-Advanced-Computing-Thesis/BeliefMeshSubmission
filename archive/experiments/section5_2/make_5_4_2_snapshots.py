# 5.4.2 nine-panel snapshot grid: offset field, fused certainty, fused interval
# half-width, at three timesteps of ONE run (dynamic offset world, seed 42).
#
# Rows are quantities, columns are timesteps. Colour scales are FIXED across
# the three columns within each row, so what changes between columns is the
# data, not the normalisation.
#
# Readout is the ADOPTED averaged-NIG convention (averaged_readout.py): per
# cell, all covering beliefs fused with weights 1/N. Certainty uses the
# pipeline's own definition and clamp (runner.py); half-width is the 90%
# Student-t predictive interval, in degrees.
#
# PNG only, 6.3 in wide, dpi 200, no footer.
#
# Run: python -u experiments/section5_2/make_5_4_2_snapshots.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from averaged_readout import averaged_params, half_width
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

RUN = Path("runs/chapter5_v2/s5_4_2_dynamic/fusion_random_seed42")
FIGS = Path("figures")
NAME = "ch5_5_4_2_dynamic_snapshots"
WIDTH, DPI = 6.3, 200
LAST = 50

# Fixed per-row scales -- stated in the run log so the caption can name them.
#
# The two uncertainty rows share ONE reading: dark = the mesh is unsure. Row 2
# is Purples_r (dark at low certainty, light at high); row 3 is Purples (dark at
# wide intervals, light at narrow), because a wide interval IS low certainty.
# Single-hue sequential, so neither row carries a good/bad connotation, and
# neither collides with the YlGnBu used for coverage elsewhere in the chapter.
# The two uncertainty rows keep ONE direction convention -- dark = the mesh is
# unsure -- but deliberately different hues, so they are not read as the same
# quantity. Both are sequential and single-hue.
#
# Row 2 is the blue arm of the offset field's own RdBu_r, sampled [0.00, 0.47]:
# dark navy at low certainty running to near-white at high. It stops a hair
# short of true white (#ecf2f5, not #ffffff) so the late panel -- where nearly
# every cell is confident -- keeps its cell structure against the page.
#
# Row 3 is Oranges sampled [0.06, 0.95]: a hue that is neither that blue nor
# the YlGnBu used for coverage elsewhere in the chapter, trimmed at both ends
# for the same page-contrast reason.
CERT_CMAP = mcolors.ListedColormap(plt.cm.RdBu_r(np.linspace(0.00, 0.47, 256)))
HW_CMAP = mcolors.ListedColormap(plt.cm.Oranges(np.linspace(0.06, 0.95, 256)))

DIVIDER = "#7a8089"

OFF_VABS = 60.0                   # row 1: RdBu_r, -60 .. +60 deg
CERT_LO, CERT_HI = 0.78, 1.00     # row 2: dark blue at low certainty
HW_LO, HW_HI = 10.0, 155.0        # row 3: dark orange at wide intervals, deg

TRAIL = 40                        # steps of wearable history drawn behind "now"
# Traces now sit on all three rows, over a red-blue field, a blue certainty map
# and an orange half-width map, so the palette avoids every map hue and each
# line carries a white casing to stay readable whatever it crosses.
WCOL = ("#111418", "#7d2fa0", "#1b7a3e")


def certainty(nu, al, be):
    """runner.py's definition, including its clamp."""
    epi = be / (np.maximum(nu, 1e-6) * np.maximum(al - 1.0, 1e-6))
    return 1.0 / (1.0 + np.minimum(epi, 10.0))


def load():
    bel = np.load(RUN / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(RUN / "cell_ncov_steps.npy").astype(int)
    paths = np.load(RUN / "realised_wearable_paths.npy")   # (W, T, 2) row, col
    _, nu, al, be = averaged_params(bel, nc)
    T, G = nu.shape[0], nu.shape[1]
    field = np.asarray(build_dynamic_offset_field(G, T))
    return field, certainty(nu, al, be), half_width(nu, al, be) * 180.0, paths


def pick_steps(hw):
    """Early: first step after the mesh has seen the world once. Mid: the
    first step whose mean half-width crosses the midpoint between the opening
    level and the settled level -- i.e. half of the adaptation is spent.
    Late: inside the last-50 window every 5.4 table reports over."""
    m = np.nanmean(hw, axis=(1, 2))
    h0, hf = float(m[:5].mean()), float(m[-LAST:].mean())
    mid = int(np.argmax(m <= 0.5 * (h0 + hf)))
    return (5, mid, hw.shape[0] - 10), h0, hf, m


def panel(ax, data, cmap, vmin, vmax):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    # a real border, not a hairline: the late-timestep panels sit at the light
    # end of both single-hue scales and would otherwise bleed into the page.
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#22252a")
    return im


def draw_wearables(ax, paths, t):
    """Faint trace of the preceding TRAIL steps, then "now" as a filled marker."""
    t0 = max(0, t - TRAIL + 1)
    for w in range(paths.shape[0]):
        col = WCOL[w % len(WCOL)]
        tr = paths[w, t0:t + 1]
        if len(tr) > 1:
            ax.plot(tr[:, 1], tr[:, 0], color="#ffffff", linewidth=2.1,
                    alpha=0.85, solid_capstyle="round", zorder=3)
            ax.plot(tr[:, 1], tr[:, 0], color=col, linewidth=0.9,
                    alpha=0.9, solid_capstyle="round", zorder=4)
        ax.scatter(tr[-1, 1], tr[-1, 0], s=26, marker="o", c=col,
                   edgecolors="#ffffff", linewidths=0.8, zorder=5)


def main():
    field, cert, hw, paths = load()
    steps, h0, hf, mean_hw = pick_steps(hw)
    G = cert.shape[1]

    rows = (
        ("Offset field", "deg", field, "RdBu_r", -OFF_VABS, OFF_VABS, True),
        ("Fused certainty", "", cert, CERT_CMAP, CERT_LO, CERT_HI, True),
        ("Interval half-width", "deg", hw, HW_CMAP, HW_LO, HW_HI, True),
    )
    labels = ("early", "mid-adaptation", "late")

    # Explicit inch-space layout. Colourbars sit vertically to the right of
    # their row, which costs width rather than height; the panel side is then
    # set to keep the whole figure under half a page.
    lab_w = 0.30          # left strip carrying the row names
    gap_w, gap_h = 0.06, 0.04
    cb_pad, cb_w, cb_lab = 0.06, 0.10, 0.50
    arrow, top, bot = 0.16, 0.17, 0.02   # arrow = the time band above the titles
    # Panels shrink by exactly the height the time band adds, and the width that
    # frees goes to the colourbar gutter, so the figure height is unchanged.
    P = (WIDTH - lab_w - 2 * gap_w - cb_pad - cb_w - cb_lab) / 3.0 - arrow / 3.0
    cb_lab += arrow
    H = arrow + top + 3 * P + 2 * gap_h + bot
    fig = plt.figure(figsize=(WIDTH, H))

    def ax_in(x, y, w, h):                    # inches, measured from top-left
        return fig.add_axes([x / WIDTH, 1.0 - (y + h) / H, w / WIDTH, h / H])

    for r, (rlab, unit, arr, cmap, lo, hi, mark) in enumerate(rows):
        y = arrow + top + r * (P + gap_h)
        for c, t in enumerate(steps):
            ax = ax_in(lab_w + c * (P + gap_w), y, P, P)
            im = panel(ax, arr[t], cmap, lo, hi)
            if mark:
                draw_wearables(ax, paths, t)
            if r == 0:
                ax.set_title("t = %d  (%s)" % (t, labels[c]), fontsize=8,
                             color="#22252a", pad=4)
        fig.text(lab_w * 0.36 / WIDTH, 1.0 - (y + P / 2.0) / H, rlab,
                 rotation=90, ha="center", va="center", fontsize=8,
                 color="#22252a")
        cb_h = 0.84 * P
        cax = ax_in(lab_w + 3 * P + 2 * gap_w + cb_pad,
                    y + (P - cb_h) / 2.0, cb_w, cb_h)
        cb = fig.colorbar(im, cax=cax)
        if unit:
            cb.set_label(unit, fontsize=7, color="#22252a", labelpad=2)
        cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, colors="#22252a")
        cb.outline.set_linewidth(0.5)

    # Column dividers, spanning the full stack, with a caret on each: the
    # columns are a sequence, not three independent panels.
    y0, y1 = arrow + top, arrow + top + 3 * P + 2 * gap_h
    for i in (1, 2):
        x = lab_w + i * P + (i - 0.5) * gap_w
        fig.add_artist(Line2D([x / WIDTH, x / WIDTH],
                              [1.0 - y1 / H, 1.0 - y0 / H],
                              color=DIVIDER, linewidth=0.9, zorder=6))
        fig.add_artist(Line2D([x / WIDTH], [1.0 - (y0 + y1) / 2.0 / H],
                              marker=">", markersize=4.0, color=DIVIDER,
                              markeredgewidth=0.0, zorder=7))

    # Time band: left to right, above the column titles.
    ax0 = lab_w + P / 2.0
    ax1 = lab_w + 2 * (P + gap_w) + P / 2.0
    ya = arrow * 0.78
    fig.add_artist(mpatches.FancyArrowPatch(
        (ax0 / WIDTH, 1.0 - ya / H), (ax1 / WIDTH, 1.0 - ya / H),
        transform=fig.transFigure, color=DIVIDER, linewidth=0.7,
        arrowstyle="-|>", mutation_scale=7, shrinkA=0, shrinkB=0, zorder=6))
    fig.text((ax0 + ax1) / 2.0 / WIDTH, 1.0 - (ya - 0.075) / H, "time",
             ha="center", va="center", fontsize=6.5, color=DIVIDER)

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / (NAME + ".png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    print("wrote %s" % out)
    print("run              %s  (dynamic offset world, seed 42, averaged-NIG readout)" % RUN)
    print("grid             %dx%d cells, %d steps" % (G, G, cert.shape[0]))
    print("timesteps        early t=%d | mid t=%d | late t=%d" % steps)
    print("  mean half-width: opening (t<5) %.1f deg, settled (last %d) %.1f deg,"
          % (h0, LAST, hf))
    print("  midpoint %.1f deg first reached at t=%d; at the three steps: %.1f, %.1f, %.1f deg"
          % (0.5 * (h0 + hf), steps[1],
             mean_hw[steps[0]], mean_hw[steps[1]], mean_hw[steps[2]]))
    print("FIXED SCALES (identical across the three columns of each row):")
    print("  row 1  offset field         RdBu_r,                    -%.0f to +%.0f deg (data over the 3 steps: %.1f to %.1f)"
          % (OFF_VABS, OFF_VABS, field[list(steps)].min(), field[list(steps)].max()))
    print("  row 2  fused certainty      RdBu_r blue arm [0,.47],   %.2f to %.2f  (data: %.3f to %.3f)  dark blue = low certainty"
          % (CERT_LO, CERT_HI, cert[list(steps)].min(), cert[list(steps)].max()))
    print("  row 3  interval half-width  Oranges [.06,.95],         %.0f to %.0f deg   (data: %.1f to %.1f)  dark orange = wide interval"
          % (HW_LO, HW_HI, hw[list(steps)].min(), hw[list(steps)].max()))
    print("clipped cells    row1 %d, row2 %d, row3 %d"
          % (int((np.abs(field[list(steps)]) > OFF_VABS).sum()),
             int(((cert[list(steps)] < CERT_LO) | (cert[list(steps)] > CERT_HI)).sum()),
             int((hw[list(steps)] > HW_HI).sum())))
    print("wearables        3 traces on ALL rows, %d steps of history behind each current position" % TRAIL)
    print("figure           %.2f x %.2f in at dpi %d  (%d x %d px)"
          % (WIDTH, H, DPI, round(WIDTH * DPI), round(H * DPI)))
    print("panel width      %.2f in  ->  %.1f px per cell at dpi %d"
          % (P, P * DPI / G, DPI))


if __name__ == "__main__":
    main()
