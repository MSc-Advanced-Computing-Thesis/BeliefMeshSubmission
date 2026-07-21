# Mapping-conflict (offset) experiments. CHECKLIST Phase 13, PRE-REGISTERED.
#
# The colour axis proved circumventable: one condition-invariant function fits
# every region, so global parameter averaging wins. These runs replace colour
# heterogeneity with a smooth spatially-varying LABEL OFFSET: the image shows
# rotation theta, the true label is theta + offset(cell), colour held uniform.
# The mapping itself is now location-dependent.
#
# ANALYTIC FLOORS (computed from the actual field and recorded before/with
# results):
#   global floor  = spatial variance of the normalised offset field (+ base
#                   task error): a single shared function cannot output
#                   different answers for identical inputs at different
#                   locations; its best case is the mean offset.
#   per-node floor = mean within-FOV variance of the field (+ base error):
#                   a per-node model can absorb its local mean offset but not
#                   within-FOV variation.
# PREDICTIONS: fedavg_global pinned at >= global floor; fusion (and gossip,
# included as the spatially-adaptive weight-space comparator) approach the
# per-node floor, far below the global floor. If fedavg_global BEATS its
# floor, position information is leaking -- debug before claiming anything.
#
# Run from the project root: python -u experiments/stage6_spatial_mesh/run_offset_experiments.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.evaluation import git_commit
from stage6_spatial_mesh.runner import run_mesh_experiment

from beliefmesh.config import load_config
from beliefmesh.node.mesh import fov_cells

ENV_DIR = Path("experiments/stage6_spatial_mesh/environment_v2")
BASELINE_CHECKPOINT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
RUN_ROOT = Path("runs/stage6/offset")
OFFSET_MAX_DEG = 60.0   # linear gradient across columns: -60 .. +60 degrees
BASE_ERROR = 0.0123     # stage 0 baseline holdout MSE (clean-task error floor)
ARMS = ["frozen", "fedavg_global", "fedavg", "fusion"]


EVENT_STEP = 195          # midpoint: the "house" event fires here
EVENT_OFFSET_DEG = 45.0   # hard-edged additional offset inside the block
EVENT_BLOCK = (slice(5, 11), slice(5, 11))  # rows 5-10, cols 5-10


def build_offset_field(grid_size: int, total_steps: int, event: bool = True,
                       event_step: int | None = None) -> np.ndarray:
    """(T, H, W): smooth column gradient throughout, plus a HARD-EDGED block
    of extra offset appearing at EVENT_STEP -- an obstruction (e.g. collapsed
    structure) abruptly changing the local mapping mid-deployment. Tests
    adaptation to sudden discontinuous change, not just gentle gradients."""
    cols = np.linspace(-OFFSET_MAX_DEG, OFFSET_MAX_DEG, grid_size)
    base = np.tile(cols, (grid_size, 1))                    # (H, W)
    field = np.tile(base, (total_steps, 1, 1))              # (T, H, W)
    if event:
        start = EVENT_STEP if event_step is None else event_step
        field[start:, EVENT_BLOCK[0], EVENT_BLOCK[1]] += EVENT_OFFSET_DEG
    return field


def analytic_floors(field_deg: np.ndarray, node_centres, fov_size, grid_size):
    field_n = field_deg / 180.0
    global_floor = float(field_n.var())
    fov_vars = []
    for cx, cy in node_centres:
        cells = fov_cells(int(cx), int(cy), fov_size, grid_size)
        vals = np.array([field_n[r, c] for r, c in cells])
        fov_vars.append(vals.var())
    return global_floor, float(np.mean(fov_vars))


def main(arms: list[str]):
    cfg = load_config()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    grids = np.load(ENV_DIR / "environment_grids.npy")
    grid_size, total_steps = grids.shape[1], grids.shape[0]
    # colour held UNIFORM and STATIC: mid-value everywhere, every timestep --
    # the offset field is the only heterogeneity in the experiment
    all_grids = np.full((total_steps, grid_size, grid_size), 0.5)
    wearable_paths = [np.load(ENV_DIR / "wearable_path.npy")]
    node_centres = np.load(ENV_DIR / "node_centres.npy")

    field = build_offset_field(grid_size, total_steps)
    pre_gf, pre_ff = analytic_floors(field[0], node_centres, 7, grid_size)
    post_gf, post_ff = analytic_floors(field[-1], node_centres, 7, grid_size)
    floors = {
        "offset_field": (f"linear column gradient +/-{OFFSET_MAX_DEG} deg; hard-edged "
                         f"+{EVENT_OFFSET_DEG} deg block rows/cols 5-10 from step {EVENT_STEP}"),
        "event_step": EVENT_STEP,
        "base_error_reference": BASE_ERROR,
        "pre_event": {"global_floor": pre_gf, "global_floor_plus_base": pre_gf + BASE_ERROR,
                      "per_node_floor": pre_ff, "per_node_floor_plus_base": pre_ff + BASE_ERROR},
        "post_event": {"global_floor": post_gf, "global_floor_plus_base": post_gf + BASE_ERROR,
                       "per_node_floor": post_ff, "per_node_floor_plus_base": post_ff + BASE_ERROR},
    }
    global_floor, fov_floor = post_gf, post_ff  # last-50-steps metric is post-event
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RUN_ROOT / "analytic_floors.yaml", "w") as f:
        yaml.safe_dump({"git_commit": git_commit(), **floors}, f, sort_keys=False)
    print("ANALYTIC FLOORS (pre-registered):")
    print(yaml.safe_dump(floors, sort_keys=False))

    results = {}
    for arm in arms:
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        results[arm] = run_mesh_experiment(
            cfg, condition=f"offset_{arm}", run_dir=RUN_ROOT / arm,
            all_grids=all_grids, wearable_paths=wearable_paths,
            node_centres=node_centres, fov_size=7, mode=arm,
            baseline_checkpoint=BASELINE_CHECKPOINT,
            n_wearable_samples=1, n_train_repeats=10,
            title=f"Offset mapping-conflict ({arm})",
            offset_field=field,
            extra_manifest={"experiment": "mapping-conflict, CHECKLIST Phase 13",
                            "analytic_floors": floors},
        )

    print("\n=== OFFSET EXPERIMENT SUMMARY ===")
    print(f"global floor (field only / +base): {global_floor:.4f} / {global_floor + BASE_ERROR:.4f}")
    print(f"per-node floor (field only / +base): {fov_floor:.4f} / {fov_floor + BASE_ERROR:.4f}")
    for arm, res in results.items():
        print(f"{arm}: measured {res['mean_mse_last_50']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    main(parser.parse_args().arms)
