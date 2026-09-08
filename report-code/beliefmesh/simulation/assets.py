"""Locations of the repository's fixed assets: pretrained checkpoints and the
reference deployment environment.

Every experiment script resolves checkpoints and environment files through
this module rather than through hard-coded relative paths, so the scripts run
from any working directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The three pretrained models of report Section 4.4: baseline (32, 64, 128),
# narrow (16, 32, 64) and wide (64, 128, 256) convolutional widths. Each
# directory also carries the manifest of the pretraining run that produced it.
CHECKPOINTS: dict[str, Path] = {
    v: REPO_ROOT / "checkpoints" / v / "pretrained_digit7.pth"
    for v in ("baseline", "narrow", "wide")
}
BASELINE_CHECKPOINT = CHECKPOINTS["baseline"]

# Reference deployment of report Section 4.3: 36 nodes on a 6x6 lattice at
# stride 3, 7x7 fields of view, 22x22 cell deployment area.
#   environment_grids.npy  (T, 22, 22) colour-world field, 390 steps
#   node_centres.npy       (36, 2) node centres, (col, row) order as used by Mesh
#   wearable_path.npy      (T, 2) the original single-wearable path
ENVIRONMENT_DIR = REPO_ROOT / "beliefmesh" / "simulation" / "environments"

# Rotation ranges excluded from the proxy task (report Section 4.1).
EXCLUDED_ROTATION_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]


def git_commit() -> str:
    """Commit hash of the repository the code is running from, for manifests."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=REPO_ROOT).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
