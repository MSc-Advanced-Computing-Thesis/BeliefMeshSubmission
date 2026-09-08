"""Figure conventions shared by every generator: 6.3 in wide (LaTeX text
width), PNG at 200 dpi, one colour palette."""

from __future__ import annotations

import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared.paths import FIGURES, TABLES

WIDTH, DPI = 6.3, 200
INK, RED, BLUE, GREY, GREEN = "#22252a", "#c1121f", "#1f5fc1", "#8a8f98", "#1b7a3e"
ORANGE = "#E6821B"        # Average fusion, matching the fusion-rule illustration
PURPLE = "#8a5fc1"


def save(fig, name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / (name + ".png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % p)
    return p


def table_path(name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else TABLES
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def ms(v):
    """(mean, sd across seeds); sd = 0 for a single value."""
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def fmt(t, p=5):
    return "%.*f +/- %.*f" % (p, t[0], p, t[1])
