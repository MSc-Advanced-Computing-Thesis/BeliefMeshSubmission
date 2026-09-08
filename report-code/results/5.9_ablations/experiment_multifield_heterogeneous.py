# Heterogeneity comparison across INDEPENDENT offset-field realisations
# (2026-08).
#
# The five-seed heterogeneity result was conditional on a single field:
# build_dynamic_offset_field defaults to seed=7 and no comparator passes one,
# so reseeding varied cell rotations, wearable paths and device layout but
# never the spatial structure of the task. Here each run seed is PAIRED with
# its own independent field realisation (offset_field_variants.py), so the
# five runs differ in task structure as well.
#
#     seed   42 -> field0_original      (the field every prior run used)
#     seed 1042 -> field1_seed11
#     seed 2042 -> field2_seed23
#     seed 3042 -> field3_seed37
#     seed 4042 -> field4_seed51_dual   (two co-directional merging wakes --
#                                        a harsher field, reported separately)
#
# seed 42 + field0_original reproduces the existing heterogeneous_comparator
# run exactly (same field bit-for-bit, same layout, same everything), so it
# doubles as a consistency check on this script.
#
# Everything else is held at the values used in the existing five-seed run:
# lr=3e-5, lam=5.0, fov=7, 3 wearables, no colour filter, same excluded
# rotation ranges, interleaved third narrow / third baseline / third wide.
#
# Run: python -u experiments/stage6_spatial_mesh/run_multifield_heterogeneous.py --mode nig_product
#      python -u experiments/stage6_spatial_mesh/run_multifield_heterogeneous.py --mode naive --homogeneous

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from beliefmesh.simulation.cert_mse_metrics import per_timestep_then_averaged
from beliefmesh.simulation.field_variants import build_field
from experiment_heterogeneous_mesh import (CKPTS, EXCLUDED_RANGES,
                                                               LAM, LR, N_WEARABLES,
                                                               assign_variants)
from beliefmesh.simulation.runner import run_mesh_experiment
from beliefmesh.simulation.offset_fields import build_random_wander_path

from beliefmesh.config import load_config

ENV = ENVIRONMENT_DIR
MODES = ["naive", "nig_product"]
SEED_FIELD = {
    42:   "field0_original",
    1042: "field1_seed11",
    2042: "field2_seed23",
    3042: "field3_seed37",
    4042: "field4_seed51_dual",
}
ROOT = Path(ART + "/appendix_E_heterogeneous_capacity/mixed_capacity_five_environments")


def run_one(mode: str, seed: int, homogeneous: bool, root_override=None):
    cfg = load_config()
    cfg.model.lr = LR
    field_name = SEED_FIELD[seed]
    root = (Path(root_override) if root_override else ROOT) / ("homog" if homogeneous else "het")
    root.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)          # unused once the filter is off
    centres = np.load(ENV / "node_centres.npy")
    field = build_field(field_name, G, T)
    # wearable seeding identical to run_heterogeneous_comparator
    wearable_seed_base = 200 if seed == 42 else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
             for i in range(N_WEARABLES)]

    variants = None if homogeneous else assign_variants(len(centres), seed=seed)
    tag = f"{mode}_seed{seed}_{field_name}"
    print(f"[{tag}] field={field_name} homogeneous={homogeneous} "
          f"field_range=[{field.min():+.1f},{field.max():+.1f}]")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=f"multifield_{'homog' if homogeneous else 'het'}_{tag}",
        run_dir=root / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode,
        baseline_checkpoint=(CKPTS["baseline"] if homogeneous else CKPTS),
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Multifield heterogeneity ({mode}, seed={seed}, {field_name})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        node_variants=variants,
        track_disagreement=(mode == "nig_product"),
        apply_colour_filter=False,
    )
    cell_mse_steps = np.load(root / tag / "cell_mse_steps.npy")
    cell_cert_steps = np.load(root / tag / "cell_cert_steps.npy")
    n_t_total = cell_mse_steps.shape[0]
    whole = float(np.nanmean(cell_mse_steps))
    last50 = float(np.nanmean(cell_mse_steps[-50:]))
    r_mean, r_std, _, _ = per_timestep_then_averaged(
        cell_mse_steps, cell_cert_steps, n_t_total - 50, n_t_total)
    print(f"\n### {tag}: whole_run={whole:.5f} last50={last50:.5f} "
          f"r(new)={r_mean:+.4f}+/-{r_std:.4f}")
    if res.get("variant_mse"):
        print(f"    variant_mse: {res['variant_mse']}")
    d = res.get("disagreement_stats")
    if d:
        print(f"    nu_cv_mean={d['nu_cv_mean']:.5f} "
              f"weight_dev={d['max_weight_minus_uniform_mean']:.5f} "
              f"gamma_star_absdiff={d['gamma_star_vs_unweighted_absdiff_mean']:.5f}")
    return whole


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEED_FIELD))
    parser.add_argument("--homogeneous", action="store_true",
                        help="matched control: identical settings, all-baseline nodes")
    args = parser.parse_args()
    for seed in [int(s) for s in args.seeds.split(",")]:
        run_one(args.mode, seed, args.homogeneous, args.root)
    print("=== DONE ===")
