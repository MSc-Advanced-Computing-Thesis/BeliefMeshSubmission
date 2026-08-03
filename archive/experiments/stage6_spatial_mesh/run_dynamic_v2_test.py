# First test of the Stage 6 fixes (CoordConv, excluded rotation ranges,
# calibration-favouring lam=5, lr=3e-5) on the REAL temporally-dynamic world
# -- the existing, already-validated build_dynamic_offset_field (RBF
# background drift + moving wind-shadow wake, field-generation-bug already
# fixed -- see run_stage4_dynamic_world_final.py, THESIS_EXPERIMENTS_SUMMARY.md
# Stage 4). Deliberately reusing that field as-is rather than building a new
# "lightly dynamic" one first, per Christian's call: try the existing world,
# step back only if it goes badly.
#
# Scope for this pass: fusion vs fedavg_global only (consensus/rho is only
# meaningful under staleness/drift, which this world finally has, but that's
# a separate follow-up once this baseline comparison is in).
#
# fusion uses wearable_policy="uncertainty_guided" (Voronoi zones,
# commit-then-reassess -- validated on the static world: MSE-competitive
# with random_wander, r collapses to ~0 there but expected to recover once
# the field is genuinely dynamic, per Christian's prediction). fedavg_global
# stays on random_wander -- see ARM_POLICY below for why.
#
# Single seed=42 first (fast sanity check) -- replicate across 5 seeds only
# if this looks sane, matching every other stage today.
#
# Run (both arms in parallel, separate processes):
#   python -u experiments/stage6_spatial_mesh/run_dynamic_v2_test.py --mode fusion
#   python -u experiments/stage6_spatial_mesh/run_dynamic_v2_test.py --mode fedavg_global

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.runner import run_mesh_experiment
from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ROOT = Path("runs/stage6/offset_world/dynamic_v2")
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]


# fusion gets uncertainty_guided routing (it has spatially-differentiated
# per-cell uncertainty to route on); fedavg_global stays on random_wander --
# it was never designed to leverage that signal (one shared model, no
# per-region specialisation), so routing by it isn't a meaningful lever for
# that arm and risks only adding noise. Each arm runs its own best available
# policy rather than forcing an identical one onto both.
ARM_POLICY = {"fusion": "uncertainty_guided", "fedavg_global": None}


def main(mode: str, policy_override: str | None = None, tag_suffix: str = ""):
    cfg = load_config()
    cfg.model.lr = LR
    ROOT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)  # deterministic (seed=7 default):
    np.save(ROOT / f"dynamic_field_v2_{mode}{tag_suffix}.npy", field)  # bit-identical every call

    spatial_var = field.reshape(T, -1).var(axis=1)
    print(f"field check: whole-run var={spatial_var.mean():.1f}, "
          f"last-50 var={spatial_var[-50:].mean():.1f}, "
          f"ratio={spatial_var.mean()/spatial_var[-50:].mean():.2f}")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    policy = ARM_POLICY[mode] if policy_override == "default" else \
             (None if policy_override == "random" else policy_override)
    tag = f"dynamic_v2_{mode}{tag_suffix}"
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=ROOT / tag,
        all_grids=uniform, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title=f"Dynamic world v2 (Stage 6 fixes, {tag})",
        offset_field=field, env_seed=SEED,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=policy, policy_step_size=0.4,
    )
    cell_mse_steps = np.load(ROOT / tag / "cell_mse_steps.npy")
    whole_run_mse = float(np.nanmean(cell_mse_steps))
    r = res.get("certainty_mse_pearson_r")
    print(f"\n### {tag} (policy={policy or 'random_wander'}): "
          f"last50={res['mean_mse_last_50']:.4f} whole_run={whole_run_mse:.4f} r={r:.4f}")
    print("(reference: fusion+uncertainty_guided=0.0195 r=-0.076, "
          "fedavg_global+random=0.0237 r=-0.118, both seed=42)")
    print("=== DONE ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fusion", "fedavg_global"], required=True)
    parser.add_argument("--policy", choices=["default", "random"], default="default",
                        help="'default' = ARM_POLICY (uncertainty_guided for fusion, "
                             "random for fedavg_global); 'random' forces random_wander "
                             "regardless of mode -- for isolating routing's own contribution")
    args = parser.parse_args()
    suffix = "_random" if args.policy == "random" and ARM_POLICY[args.mode] is not None else ""
    main(args.mode, args.policy, tag_suffix=suffix)
