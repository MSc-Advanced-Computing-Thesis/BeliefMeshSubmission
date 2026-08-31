# Loss-weighting ablation (2026-08): is the heterogeneity result attributable
# to the fused TARGET alone, or does the per-sample loss weighting contribute?
#
# Fusion feeds a node's update through two separate channels:
#   1. gamma_star -- the nu-weighted mean of contributors, i.e. WHAT the node
#      trains toward.
#   2. cert_star  -- entering nig_loss as a per-sample weight, i.e. HOW MUCH
#      each supervised cell counts within that node's single update.
#
# Channel 2 is weaker than it appears. nig_loss reduces as
# (per_sample * weights).sum() / weights.sum() -- a NORMALISED weighted mean --
# so scaling every weight by a constant cancels exactly. A uniform cert_star
# across a node's supervised cells has literally no effect; only WITHIN-BATCH
# relative differences can change the update.
#
# temper_gradient=False sets trust_weights=None, disabling channel 2 while
# leaving channel 1 untouched. Comparing against the existing five-field
# heterogeneous nig_product runs isolates the weighting's contribution.
#
# Also collects the quantity that decides the question a priori: the spread of
# cert_star ACROSS CELLS WITHIN one node's update (track_trust_batches). This
# is NOT the per-event nu-CV already measured -- that is spread BETWEEN
# CONTRIBUTORS to one cell. If the within-update spread has not widened under
# heterogeneity, channel 2 cannot be contributing and the MSE comparison
# should confirm it.
#
# Run: python -u experiments/stage6_spatial_mesh/run_temper_ablation.py --arm het_notemper

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.offset_field_variants import build_field
from stage6_spatial_mesh.run_heterogeneous_comparator import (CKPTS, EXCLUDED_RANGES,
                                                               LAM, LR, N_WEARABLES,
                                                               assign_variants)
from stage6_spatial_mesh.run_multifield_heterogeneous import SEED_FIELD
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import build_random_wander_path

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
ROOT = Path("runs/stage6/offset_world/temper_ablation")

# arm -> (heterogeneous?, temper_gradient?)
ARMS = {
    "het_temper":    (True,  True),   # reference: reproduces the existing runs
    "het_notemper":  (True,  False),  # the ablation
    "homog_temper":  (False, True),   # homogeneous cert_star spread reference
}


def run_one(arm: str, seed: int, root: str | None = None):
    global ROOT
    if root:
        ROOT = Path(root)
    heterogeneous, temper = ARMS[arm]
    cfg = load_config()
    cfg.model.lr = LR
    field_name = SEED_FIELD[seed]
    root = ROOT / arm
    root.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_field(field_name, G, T)
    wearable_seed_base = 200 if seed == 42 else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
             for i in range(N_WEARABLES)]

    variants = assign_variants(len(centres), seed=seed) if heterogeneous else None
    tag = f"seed{seed}_{field_name}"
    print(f"[{arm}/{tag}] heterogeneous={heterogeneous} temper_gradient={temper}")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=f"temper_{arm}_{tag}", run_dir=root / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="nig_product",
        baseline_checkpoint=(CKPTS if heterogeneous else CKPTS["baseline"]),
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Temper ablation ({arm}, seed={seed}, {field_name})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        node_variants=variants,
        temper_gradient=temper,
        track_disagreement=True,
        track_trust_batches=True,
        apply_colour_filter=False,
    )
    cms = np.load(root / tag / "cell_mse_steps.npy")
    whole, last50 = float(np.nanmean(cms)), float(np.nanmean(cms[-50:]))
    print(f"\n### {arm}/{tag}: whole_run={whole:.5f} last50={last50:.5f}")
    tb = res.get("trust_batch_stats")
    if tb:
        print(f"    trust_batch: n_updates={tb['n_updates']} cells/update={tb['n_cells_mean']:.1f} "
              f"cert_mean={tb['cert_mean']:.5f} WITHIN-UPDATE cv_mean={tb['within_update_cv_mean']:.6f} "
              f"cv_median={tb['within_update_cv_median']:.6f} range_mean={tb['within_update_range_mean']:.6f} "
              f"cert_span=[{tb['cert_min_overall']:.4f},{tb['cert_max_overall']:.4f}]")
    d = res.get("disagreement_stats")
    if d:
        print(f"    nu_cv_mean={d['nu_cv_mean']:.5f} (BETWEEN-contributor, for contrast)")
    return whole


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEED_FIELD))
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    args = parser.parse_args()
    for seed in [int(s) for s in args.seeds.split(",")]:
        run_one(args.arm, seed, args.root)
    print("=== DONE ===")
