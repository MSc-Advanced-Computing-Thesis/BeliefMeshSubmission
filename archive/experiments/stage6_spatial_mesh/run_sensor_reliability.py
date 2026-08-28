# Imperfect ground truth (2026-08). Spec Sec 3.5 assigns every wearable
# measurement a training certainty of 1.0 and notes this could be lowered to
# reflect known sensor error; that has never been tested.
#
# MECHANISM LIMIT, stated up front because it bounds what condition A can
# possibly show. nig_loss reduces as (per_sample*w).sum()/w.sum() -- a
# NORMALISED weighted mean. A batch whose samples all originate from one
# wearable therefore has a uniform weight vector that cancels exactly, and
# reliability has no effect whatsoever. Varying reliability BETWEEN wearables
# only bites where a single node holds two wearables of differing reliability
# in its FOV within one step. anchor_weight_log measures how often that
# occurs; if it is rare, condition A is inert by construction and the MSE
# comparison is uninformative -- which is itself the reportable finding.
#
# Condition B avoids that trap entirely: the jitter corrupts the TARGET, which
# no normalisation can cancel. A node fed noisy labels should genuinely learn
# less, its nu should fall, and that asymmetry should reach fusion through the
# contributors' own parameters -- the same route device heterogeneity used.
#
# Evaluation is unaffected in both conditions: runner scores against
# environment truth, never against a jittered label.
#
# Run: python -u experiments/stage6_spatial_mesh/run_sensor_reliability.py --condition het_sensors --mode nig_product

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.cert_mse_metrics import per_timestep_then_averaged
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/sensor_reliability")
N_WEARABLES = 3
LR, LAM, SEED = 3e-5, 5.0, 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]

RELIABILITY = [1.0, 0.7, 0.4]
# jitter std in normalised label units (1.0 = 180 deg), proportional to
# unreliability: std = 0.1 * (1 - r) -> 0.0 / 5.4 / 10.8 degrees. The largest
# is comparable to the run's own RMSE (~18 deg), so it is a material sensor
# error rather than a token perturbation.
JITTER = [0.1 * (1.0 - r) for r in RELIABILITY]

CONDITIONS = {
    # (reliability, jitter)
    "baseline":     (None, None),                 # control: certainty 1.0, clean
    "het_sensors":  (RELIABILITY, None),          # A: differing assigned reliability only
    "matched_noise": (RELIABILITY, JITTER),       # B: reliability backed by real noise
}


def run_one(condition: str, mode: str, seed: int = SEED, root_override=None):
    reliability, jitter = CONDITIONS[condition]
    cfg = load_config()
    cfg.model.lr = LR
    root = (Path(root_override) if root_override else ROOT) / condition
    root.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    tag = f"{mode}_seed{seed}"
    print(f"[{condition}/{tag}] reliability={reliability} jitter_deg="
          f"{[round(j*180,1) for j in jitter] if jitter else None}")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=f"sensor_{condition}_{tag}", run_dir=root / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Sensor reliability ({condition}, {mode}, seed={seed})",
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        track_disagreement=(mode == "nig_product"),
        rho=0.2,
        track_trust_batches=True,
        wearable_reliability=reliability,
        wearable_label_jitter=jitter,
        apply_colour_filter=False,
    )
    cms = np.load(root / tag / "cell_mse_steps.npy")
    ccs = np.load(root / tag / "cell_cert_steps.npy")
    n = cms.shape[0]
    whole, last50 = float(np.nanmean(cms)), float(np.nanmean(cms[-50:]))
    r_mean, r_std, _, _ = per_timestep_then_averaged(cms, ccs, n - 50, n)
    print(f"\n### {condition}/{tag}: whole_run={whole:.5f} last50={last50:.5f} "
          f"r(new)={r_mean:+.4f}+/-{r_std:.4f}")
    d = res.get("disagreement_stats")
    if d:
        print(f"    GATING nu_cv_mean={d['nu_cv_mean']:.5f} "
              f"weight_dev={d['max_weight_minus_uniform_mean']:.5f} "
              f"gamma_star_absdiff={d['gamma_star_vs_unweighted_absdiff_mean']:.5f}")
    awl = root / tag / "anchor_weight_log.npy"
    if awl.exists():
        a = np.load(awl)
        mixed = a[:, 3] > 1
        print(f"    anchor batches: {len(a)} total, {int(mixed.sum())} "
              f"({100*mixed.mean():.2f}%) mix >1 reliability "
              f"-> the rest have a uniform weight vector that CANCELS")
    return whole


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=list(CONDITIONS), required=True)
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seeds; default (None) = single SEED, "
                             "reproducing the original single-seed invocation")
    parser.add_argument("--mode", choices=["naive", "nig_product", "gossip_uniform",
                                           "fedavg_global", "frozen"], required=True)
    args = parser.parse_args()
    for _sd in ([int(x) for x in args.seeds.split(",")] if args.seeds else [SEED]):
        run_one(args.condition, args.mode, _sd, args.root)
    print("=== DONE ===")
