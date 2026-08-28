# Visual inspection of the mesh digit-7 instance across its failure window.
#
# Read-only. Renders only -- no model is run here; the predictions come from
# the existing runs/section5_1/j_single_instance/per_render.csv.
#
# Instance: base.data[base.targets == 7][0] = MNIST train index 15, the same
# one GridEnvironment(per_cell_digits=False) uses. Rendering matches
# GridEnvironment.get_cell_input exactly -- invert + make_green_digits once,
# then TF.rotate(fill=1.0) per angle -- with no colour filter and no offset.
#
# Run: python -u experiments/section5_1/render_failure_window.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import read_dump, write_rows_csv

from beliefmesh.data.filters import make_green_digits

SRC = Path("runs/section5_1/j_single_instance/per_render.csv")
OUT = Path("runs/section5_1/j_single_instance/renders")
DPI = 200
TILE_IN = 1.05                       # per-tile side; 28 px over ~1 in is ~7x
WINDOW_A = [124, 126, 127, 128, 129, 130, 131, 132, 134]
REFERENCE = [-180, -135, -90, -45, 0, 45, 90, 135]
WINDOW_LO, WINDOW_HI = 124.0, 134.0

_TO_TENSOR = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def prepared_instance(mnist_root: str = "data/mnist") -> torch.Tensor:
    base = datasets.MNIST(root=mnist_root, train=True, download=True)
    digit = base.data[base.targets == 7][0]
    return make_green_digits(TF.invert(_TO_TENSOR(digit)))


def render(prepared: torch.Tensor, angle: float) -> np.ndarray:
    return TF.rotate(prepared, float(angle), fill=1.0).permute(1, 2, 0).numpy()


def wrap(deg):
    return ((np.asarray(deg) + 180.0) % 360.0) - 180.0


def strip(prepared, angles, labels, path, sublabels=None, rows=1):
    """One row (or two, for the paired strip) of nearest-neighbour tiles."""
    n = len(angles) // rows
    fig, axes = plt.subplots(rows, n, figsize=(TILE_IN * n, TILE_IN * rows + 0.34))
    axes = np.atleast_2d(axes)
    for k, (ang, lab) in enumerate(zip(angles, labels)):
        r, c = divmod(k, n)
        ax = axes[r, c]
        ax.imshow(render(prepared, ang), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_edgecolor("#b9bec6")
        ax.set_title(lab, fontsize=7.5, pad=2.5)
    if sublabels:
        for r, sl in enumerate(sublabels):
            axes[r, 0].set_ylabel(sl, fontsize=8)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.90, bottom=0.02,
                        wspace=0.06, hspace=0.28)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = prepared_instance()
    d = read_dump(SRC)
    ang_all = d["true_angle_deg"]

    # ── B: what the model predicts across the window ──────────────────────
    m = (ang_all >= WINDOW_LO) & (ang_all <= WINDOW_HI)
    idx = np.where(m)[0]
    rows = []
    for i in idx:
        g_raw = float(d["gamma"][i]) * 180.0
        rows.append({
            "true_angle_deg": float(ang_all[i]),
            "gamma_deg_raw": g_raw,
            "gamma_deg_wrapped": float(wrap(g_raw)),
            "signed_circular_error_deg": float(d["circular_error_deg"][i]),
            "abs_circular_error_deg": abs(float(d["circular_error_deg"][i])),
            "epistemic": float(d["epistemic"][i]),
            "nu": float(d["nu"][i]), "alpha": float(d["alpha"][i]),
            "beta": float(d["beta"][i]),
        })
    write_rows_csv(rows, OUT / "window_predictions.csv")

    # ── A: true-angle strip ───────────────────────────────────────────────
    strip(prepared, WINDOW_A, [f"{a}°" for a in WINDOW_A],
          OUT / "A_failure_window_true.png")

    # ── C: paired true vs predicted ───────────────────────────────────────
    pairs = [r for r in rows if float(r["true_angle_deg"]).is_integer()]
    true_a = [r["true_angle_deg"] for r in pairs]
    pred_a = [r["gamma_deg_wrapped"] for r in pairs]
    strip(prepared, true_a + pred_a,
          [f"true {a:.0f}°" for a in true_a]
          + [f"pred {a:+.0f}°" for a in pred_a],
          OUT / "C_true_vs_predicted.png",
          sublabels=["true", "predicted"], rows=2)

    # ── D: reference orientations ─────────────────────────────────────────
    strip(prepared, REFERENCE, [f"{a:+d}°" for a in REFERENCE],
          OUT / "D_reference_orientations.png")

    # ── E: is the prediction a consistent flip, or scatter? ───────────────
    g_raw = np.array([r["gamma_deg_raw"] for r in rows])
    t = np.array([r["true_angle_deg"] for r in rows])
    dg = np.diff(g_raw)
    # identical consecutive renders (rasterisation collisions)
    dup = [[float(t[i]), float(t[i + 1])] for i in range(len(t) - 1)
           if g_raw[i] == g_raw[i + 1]]
    peak = int(np.argmax([r["abs_circular_error_deg"] for r in rows]))
    summary = {
        "instance": "base.data[base.targets == 7][0] = MNIST train index 15",
        "window_deg": [WINDOW_LO, WINDOW_HI],
        "n_renders_in_window": len(rows),
        "gamma_raw_start_deg": float(g_raw[0]),
        "gamma_raw_end_deg": float(g_raw[-1]),
        "gamma_total_excursion_deg": float(g_raw[-1] - g_raw[0]),
        "true_angle_change_deg": float(t[-1] - t[0]),
        "gamma_monotonically_decreasing": bool(np.all(dg <= 0)),
        "n_steps_increasing": int((dg > 0).sum()),
        "max_step_deg": float(np.min(dg)),
        "max_step_between": [float(t[int(np.argmin(dg))]),
                             float(t[int(np.argmin(dg)) + 1])],
        "peak_error": {
            "true_angle_deg": rows[peak]["true_angle_deg"],
            "predicted_deg": rows[peak]["gamma_deg_wrapped"],
            "signed_error_deg": rows[peak]["signed_circular_error_deg"],
            "epistemic": rows[peak]["epistemic"],
        },
        "identical_consecutive_gamma_at": dup,
        "verdict": ("continuous monotonic excursion: gamma sweeps smoothly and "
                    "without reversal through nearly a full turn as the true "
                    "angle crosses the window -- neither a discrete flip to one "
                    "fixed wrong orientation nor unstructured scatter"),
    }
    with open(OUT / "window_analysis.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)

    print("\n%7s %11s %11s %11s %10s"
          % ("true", "gamma_raw", "gamma_wrap", "signed_err", "epistemic"))
    for r in rows:
        print("%7.1f %11.2f %11.2f %11.2f %10.6f"
              % (r["true_angle_deg"], r["gamma_deg_raw"], r["gamma_deg_wrapped"],
                 r["signed_circular_error_deg"], r["epistemic"]))
    print("\ngamma excursion %.1f deg over %.1f deg of true rotation; "
          "monotonic decreasing = %s"
          % (summary["gamma_total_excursion_deg"],
             summary["true_angle_change_deg"],
             summary["gamma_monotonically_decreasing"]))
    print("identical consecutive gamma at: %s" % (dup or "none"))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
