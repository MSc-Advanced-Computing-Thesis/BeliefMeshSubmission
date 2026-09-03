# Supplementary video for 5.4.2: the same three quantities as the static
# nine-panel figure, animated across the whole run instead of sampled at three
# timesteps.
#
# LAYOUT. The static figure spends its horizontal axis on time. Here time is
# the animation, so the horizontal axis is free for the quantities: one row of
# three panels, which lets the eye compare all three at a single instant
# without scanning vertically, and gives a wide aspect for viewing. The row
# labels of the static figure become the panel titles, wording unchanged.
#
# Every scale, colour map, trail length and trace colour is imported from
# make_5_4_2_snapshots.py rather than restated, so the video and the figure
# cannot drift apart. Scales are fixed over the WHOLE run -- no autoscaling --
# which is what makes the contraction of the half-width panel visible.
#
# Not embedded in the thesis, so not bound by the 6.3 in text width.
#
# Run: python -u experiments/section5_2/make_5_4_2_video.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_5_4_2_snapshots import (CERT_CMAP, CERT_HI, CERT_LO, FIGS, HW_CMAP,
                                  HW_HI, HW_LO, OFF_VABS, RUN, TRAIL, WCOL,
                                  certainty, load)
from averaged_readout import half_width  # noqa: F401  (re-exported by load)

NAME = "ch5_5_4_2_dynamic_evolution"
FPS = 10
DPI = 130
VW = 12.0                       # viewing size, not text width
STRIDE = 1                      # every timestep; no subsampling

# inch-space layout, so the square panels fill the frame instead of floating in
# dead space and the rightmost colourbar label is not clipped at the edge
L_M, R_M, GAP = 0.10, 0.06, 0.30
CB_PAD, CB_W, CB_LAB = 0.07, 0.12, 0.42
HEAD, TITLE, BOT = 0.42, 0.26, 0.14
SIDE = (VW - L_M - R_M - 2 * GAP - 3 * (CB_PAD + CB_W + CB_LAB)) / 3.0
VH = HEAD + TITLE + SIDE + BOT
FIGSIZE = (VW, VH)


def build(fig, arrs, G):
    """One row of three panels, each with its own fixed vertical colour bar."""
    rows = (
        ("Offset field", "deg", arrs[0], "RdBu_r", -OFF_VABS, OFF_VABS),
        ("Fused certainty", "", arrs[1], CERT_CMAP, CERT_LO, CERT_HI),
        ("Interval half-width", "deg", arrs[2], HW_CMAP, HW_LO, HW_HI),
    )
    ims, traces = [], []
    y = HEAD + TITLE                       # top of the panels, inches from top
    for i, (lab, unit, arr, cmap, lo, hi) in enumerate(rows):
        x = L_M + i * (SIDE + CB_PAD + CB_W + CB_LAB + GAP)
        ax = fig.add_axes([x / VW, 1.0 - (y + SIDE) / VH, SIDE / VW, SIDE / VH])
        im = ax.imshow(arr[0], cmap=cmap, vmin=lo, vmax=hi, origin="upper",
                       interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.7); sp.set_color("#22252a")
        ax.set_title(lab, fontsize=11, color="#22252a", pad=6)
        cb_h = 0.86 * SIDE
        cax = fig.add_axes([(x + SIDE + CB_PAD) / VW,
                            1.0 - (y + (SIDE + cb_h) / 2.0) / VH,
                            CB_W / VW, cb_h / VH])
        cb = fig.colorbar(im, cax=cax)
        if unit:
            cb.set_label(unit, fontsize=8, color="#22252a", labelpad=2)
        cb.ax.tick_params(labelsize=8, length=2, width=0.6, colors="#22252a")
        cb.outline.set_linewidth(0.5)
        ims.append(im)

        per_panel = []
        for wi in range(len(WCOL)):
            case = Line2D([], [], color="#ffffff", linewidth=3.0, alpha=0.85,
                          solid_capstyle="round", zorder=3)
            line = Line2D([], [], color=WCOL[wi], linewidth=1.3, alpha=0.9,
                          solid_capstyle="round", zorder=4)
            dot = Line2D([], [], color=WCOL[wi], marker="o", markersize=6.5,
                         markeredgecolor="#ffffff", markeredgewidth=1.1,
                         linestyle="none", zorder=5)
            for a in (case, line, dot):
                ax.add_line(a)
            per_panel.append((case, line, dot))
        traces.append(per_panel)
        ax.set_xlim(-0.5, G - 0.5); ax.set_ylim(G - 0.5, -0.5)
    return ims, traces


def main():
    field, cert, hw, paths = load()
    T, G = cert.shape[0], cert.shape[1]
    arrs = (field, cert, hw)

    fig = Figure(figsize=FIGSIZE, dpi=DPI)
    canvas = FigureCanvasAgg(fig)
    ims, traces = build(fig, arrs, G)
    head = fig.text(0.5, 1.0 - HEAD * 0.55 / VH, "", ha="center", va="center", fontsize=11.5,
                    color="#22252a")

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / (NAME + ".mp4")
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=None)
    frames = range(0, T, STRIDE)
    for t in frames:
        for im, arr in zip(ims, arrs):
            im.set_data(arr[t])
        t0 = max(0, t - TRAIL + 1)
        for per_panel in traces:
            for wi, (case, line, dot) in enumerate(per_panel):
                tr = paths[wi, t0:t + 1]
                case.set_data(tr[:, 1], tr[:, 0])
                line.set_data(tr[:, 1], tr[:, 0])
                dot.set_data([tr[-1, 1]], [tr[-1, 0]])
        head.set_text("dynamic offset world, seed 42   |   step %3d / %d" % (t, T - 1))
        canvas.draw()
        frame = np.asarray(canvas.buffer_rgba())[..., :3]
        frame = frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2]
        writer.append_data(frame)
    writer.close()

    n = len(list(frames))
    print("wrote %s" % out)
    print("run              %s" % RUN)
    print("layout           one row of three panels (time is the animation, so the")
    print("                 horizontal axis carries the quantities instead)")
    print("frames           %d of %d timesteps, stride %d -> every timestep, no subsampling"
          % (n, T, STRIDE))
    print("frame rate       %d fps" % FPS)
    print("duration         %.1f s" % (n / FPS))
    print("frame size       %d x %d px (%.1f x %.1f in at dpi %d)"
          % (frame.shape[1], frame.shape[0], VW, VH, DPI))
    print("scales           FIXED over the whole run, identical to the static figure:")
    print("                   offset field       RdBu_r,   -%.0f to +%.0f deg" % (OFF_VABS, OFF_VABS))
    print("                   fused certainty    blue arm, %.2f to %.2f" % (CERT_LO, CERT_HI))
    print("                   interval half-wid  Oranges,  %.0f to %.0f deg" % (HW_LO, HW_HI))
    print("out-of-range     field %d, cert %d, hw %d cell-steps over the WHOLE run"
          % (int((np.abs(field) > OFF_VABS).sum()),
             int(((cert < CERT_LO) | (cert > CERT_HI)).sum()),
             int(((hw < HW_LO) | (hw > HW_HI)).sum())))
    print("wearables        3 traces on all panels, %d steps of history" % TRAIL)


if __name__ == "__main__":
    main()
