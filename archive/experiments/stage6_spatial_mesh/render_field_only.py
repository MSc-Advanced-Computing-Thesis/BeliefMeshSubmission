# Standalone field-only animation -- no mesh/model computation, just the
# environment itself, for sanity-checking a field design before spending
# compute on a full run.
#
# Run: python -u experiments/stage6_spatial_mesh/render_field_only.py

from __future__ import annotations

import sys
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

OUT = Path("runs/stage6/offset_world/dynamic/field_preview.mp4")
G, T = 22, 390

field = build_dynamic_offset_field(G, T)
disp = ((field + 180) % 360) - 180
vabs = np.percentile(np.abs(disp), 99)  # actual range, not a fixed +-180 guess
print(f"field range: min {disp.min():.1f} max {disp.max():.1f} | display vmin/vmax +-{vabs:.1f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
writer = imageio.get_writer(OUT, fps=15, codec="libx264", quality=8, macro_block_size=1)

for t in range(T):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    im = ax.imshow(disp[t], cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                   aspect="equal", interpolation="nearest")
    block_r, block_c = 14.0, 8.0
    ax.plot(block_c, block_r, "s", color="white", markersize=9, markeredgecolor="black")
    ax.text(block_c, block_r - 1.3, "block", color="white", fontsize=8, ha="center",
           bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
    ax.set_title(f"Dynamic offset field -- step {t:03d}/{T}  "
                f"(smooth RBF background + sweeping wake behind the block)",
                color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("label offset (deg, wrapped for display)", color="#aaaaaa", fontsize=8)
    cb.ax.tick_params(colors="#aaaaaa", labelsize=7)
    plt.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    writer.append_data(np.asarray(canvas.buffer_rgba()))
    plt.close(fig)
    if t % 50 == 0:
        print(f"frame {t}/{T}")

writer.close()
print(f"Saved {OUT}")
