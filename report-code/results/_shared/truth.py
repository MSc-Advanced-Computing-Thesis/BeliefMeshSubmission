"""Ground truth reconstruction for stored mesh runs.

Ground truth is not stored with a run, but the environment is deterministic:
cell rotations follow from rotation_seed, and the label is
GridEnvironment._label(rotation, row, col, step) under the run's offset field.
The reconstruction is validated against the stored cell_mse_steps (single-
contributor cells agree to ~1e-7) by averaged_readout.validate_reduction().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.simulation.assets import EXCLUDED_ROTATION_RANGES as EX
from beliefmesh.simulation.field_variants import FIELDS, build_field
from beliefmesh.simulation.offset_fields import build_dynamic_offset_field

LAST = 50                 # the last-50 window every table reports over
_truth_cache: dict = {}


def wrap(x):
    """Wrap a normalised-angle difference into [-1, 1)."""
    return x - 2.0 * np.round(x / 2.0)


def offset_field_for(run_dir: Path, G: int, T: int) -> np.ndarray:
    """The offset field a stored run actually used, resolved from its path.

    Runs under the three-node colour world carry no offset (the label is the
    rotation itself). Runs whose path names one of the five environments
    (field0_original ... field4_seed51_dual) use that environment. Everything
    else uses the reference dynamic field (field0)."""
    parts = [p.lower() for p in Path(run_dir).parts]
    if any("three_node_colour_world" in p for p in parts):
        return np.zeros((T, G, G))
    for name in FIELDS:
        if any(name in p for p in parts):
            return np.asarray(build_field(name, G, T))
    return np.asarray(build_dynamic_offset_field(G, T))


def truth_for_field(field: np.ndarray, seed: int, T: int, G: int) -> np.ndarray:
    field = np.asarray(field)
    key = (seed, T, G, float(field.sum()), float(np.abs(field).sum()))
    if key in _truth_cache:
        return _truth_cache[key]
    env = GridEnvironment(G, np.full((T, G, G), 0.5), rotation_seed=seed,
                          offset_field=field, excluded_rotation_ranges=EX,
                          apply_colour_filter=False)
    rot = env.cell_rotations
    t = np.empty((T, G, G))
    for s in range(T):
        for r in range(G):
            for c in range(G):
                t[s, r, c] = env._label(rot[s, r, c], r, c, s).item()
    _truth_cache[key] = t
    return t


def truth_for_run(run_dir: Path, seed: int, T: int, G: int) -> np.ndarray:
    """Ground truth (T, G, G) for a stored run, under the run's own field."""
    return truth_for_field(offset_field_for(Path(run_dir), G, T), seed, T, G)


def truth_for(seed: int, T: int, G: int) -> np.ndarray:
    """Ground truth under the reference dynamic field (field0)."""
    return truth_for_field(np.asarray(build_dynamic_offset_field(G, T)), seed, T, G)
