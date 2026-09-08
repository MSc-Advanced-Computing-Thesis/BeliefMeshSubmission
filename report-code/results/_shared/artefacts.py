"""Reading stored run artefacts.

A run directory holds a manifest.yaml and a set of arrays. Arrays are stored
compressed (name.npz, key 'arr'); a plain name.npy is accepted too, so a
directory freshly written by the simulation runner reads the same way.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import yaml


def load_array(run_dir: str | Path, name: str) -> np.ndarray:
    run_dir = Path(run_dir)
    npy, npz = run_dir / (name + ".npy"), run_dir / (name + ".npz")
    if npy.exists():
        return np.load(npy, allow_pickle=True)
    with np.load(npz, allow_pickle=True) as f:
        return f["arr"]


def has_array(run_dir: str | Path, name: str) -> bool:
    run_dir = Path(run_dir)
    return (run_dir / (name + ".npy")).exists() or (run_dir / (name + ".npz")).exists()


def manifest(run_dir: str | Path) -> dict:
    with open(Path(run_dir) / "manifest.yaml", encoding="utf8") as f:
        return yaml.safe_load(f)


def run_dirs(pattern: str | Path) -> list[Path]:
    """Run directories whose manifest.yaml matches a glob pattern, sorted."""
    return [Path(os.path.dirname(p)) for p in
            sorted(glob.glob(str(Path(pattern) / "manifest.yaml"), recursive=True))]
