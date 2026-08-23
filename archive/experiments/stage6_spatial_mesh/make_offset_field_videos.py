# Validation pass for the independent offset-field realisations (2026-08).
# Renders one video per NEW field (heatmap over the full run, shared colour
# scale across all four, timestep annotated) and prints the comparison table
# against the existing field. NO experiments are launched here -- this exists
# so the fields can be inspected before any compute is spent.
#
# Encoding note: offsets are SIGNED, so the colour scale is diverging with a
# neutral midpoint pinned at 0 deg and symmetric limits shared across every
# field, so frames are comparable between videos as well as within one. The
# map is RdBu_r -- the SAME one the project's existing offset-field videos use
# (render_field_only.py, generate_video.py), so these read identically to the
# earlier renders rather than introducing a second convention.
#
# Run: python -u experiments/stage6_spatial_mesh/make_offset_field_videos.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.offset_field_variants import (FIELDS, build_field,
                                                        reproduce_original,
                                                        wrap_headroom)

OUT_DIR = Path("figures/offset_fields")
GRID, STEPS, FPS = 22, 390, 20
NEW_FIELDS = ["field1_seed11", "field2_seed23", "field3_seed37", "field4_seed51_dual"]

# Matches render_field_only.py / generate_video.py exactly -- do not swap for a
# bespoke map, the earlier field videos are the reference these are read against.
DIVERGING = "RdBu_r"


def characterise(f: np.ndarray, masks: np.ndarray) -> dict:
    any_mask = masks.any(axis=0)
    per_step = np.abs(np.diff(f, axis=0))
    jr = np.abs(np.diff(f, axis=1))
    jc = np.abs(np.diff(f, axis=2))
    adj = np.concatenate([jr.ravel(), jc.ravel()])
    cov = any_mask.mean(axis=(1, 2))
    return dict(
        lo=f.min(), hi=f.max(),
        step_mean=per_step.mean(), step_max=per_step.max(),
        adj_mean=adj.mean(), adj_max=adj.max(),
        cov_mean=100 * cov.mean(), cov_min=100 * cov.min(), cov_max=100 * cov.max(),
    )


def render_video(name: str, f: np.ndarray, masks: np.ndarray, vmax: float) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.mp4"
    spec = FIELDS[name]
    blocks = [w.block_pos for w in spec["wedges"]]

    fig = Figure(figsize=(5.2, 4.6), dpi=110)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(f[0], cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
    for (br, bc) in blocks:                      # mark each obstruction
        ax.plot(bc, br, marker="o", ms=7, mfc="none", mec="white", mew=1.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb"); s.set_linewidth(0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("offset (degrees)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    title = ax.set_title("", fontsize=10, pad=8)
    sub = fig.text(0.5, 0.015, spec["note"], ha="center", fontsize=7.5, color="#555555")

    writer = imageio.get_writer(out, fps=FPS, codec="libx264",
                                quality=8, macro_block_size=None)
    n_overlap = (masks.sum(axis=0) >= 2)
    for step in range(f.shape[0]):
        im.set_data(f[step])
        extra = f"   overlap cells: {int(n_overlap[step].sum())}" if masks.shape[0] > 1 else ""
        title.set_text(f"{name}    step {step:3d}/{f.shape[0]}{extra}")
        canvas.draw()
        frame = np.asarray(canvas.buffer_rgba())[..., :3]
        # libx264 requires even width/height; the Agg canvas is whatever the
        # figsize*dpi lands on, so trim rather than fight the layout
        frame = frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2]
        writer.append_data(frame)
    writer.close()
    return out


def main():
    reproduce_original()   # refuse to proceed if the generator drifted

    fields, stats = {}, {}
    for name in FIELDS:
        f, m = build_field(name, GRID, STEPS, return_masks=True)
        fields[name] = (f, m)
        stats[name] = characterise(f, m)

    vmax = max(max(abs(s["lo"]), abs(s["hi"])) for n, s in stats.items() if n in NEW_FIELDS)
    vmax = float(np.ceil(vmax / 10.0) * 10)
    print(f"shared colour scale: +/-{vmax:.0f} deg (symmetric, grey at 0)\n")

    hdr = (f"{'field':22s} {'range (deg)':>20s} {'per-step':>16s} "
           f"{'adjacent-cell':>17s} {'wedge coverage %':>22s}")
    print(hdr); print("-" * len(hdr))
    for name in FIELDS:
        s = stats[name]
        tag = name + ("  [ref]" if name == "field0_original" else "")
        print(f"{tag:22s} [{s['lo']:+7.1f},{s['hi']:+7.1f}] "
              f"{s['step_mean']:6.3f}/{s['step_max']:6.2f} "
              f"{s['adj_mean']:6.2f}/{s['adj_max']:7.2f} "
              f"{s['cov_mean']:7.1f} ({s['cov_min']:.1f}-{s['cov_max']:.1f})")
    print("\n(columns: range | mean/max per-step change | mean/max adjacent-cell diff | "
          "mean (min-max) coverage)")

    # dual-wedge overlap accounting
    print("\n--- dual-wedge overlap (field4_seed51_dual) ---")
    f, m = fields["field4_seed51_dual"]
    ov = (m.sum(axis=0) >= 2)
    steps_with = np.where(ov.any(axis=(1, 2)))[0]
    print(f"headroom before wrap could trigger: {wrap_headroom('field4_seed51_dual'):.0f} deg "
          f"(background 60 + wedges 50+50)")
    if len(steps_with):
        print(f"steps with a non-empty overlap: {len(steps_with)}/{STEPS} "
              f"(first {steps_with.min()}, last {steps_with.max()})")
        print(f"peak simultaneous overlap cells: {int(ov.sum(axis=(1,2)).max())} "
              f"({100*ov.sum(axis=(1,2)).max()/GRID**2:.1f}% of grid)")
        vals = f[ov]
        print(f"max |offset| realised INSIDE overlap: {np.abs(vals).max():.1f} deg "
              f"(range {vals.min():+.1f} to {vals.max():+.1f})")
    else:
        print("!! no overlap realised -- geometry needs adjustment")
    print(f"max |offset| anywhere in this field: {np.abs(f).max():.1f} deg")
    print(f"cells within 1e-9 of the +/-180 wrap boundary: "
          f"{int((np.abs(np.abs(f) - 180.0) < 1e-9).sum())}")

    print()
    for name in NEW_FIELDS:
        f, m = fields[name]
        p = render_video(name, f, m, vmax)
        print(f"saved {p}")


if __name__ == "__main__":
    main()
