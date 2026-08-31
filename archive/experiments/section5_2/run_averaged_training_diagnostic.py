# DIAGNOSTIC: averaged NIG fusion applied in TRAINING as well as at readout.
#
# ONE RUN, seed 42. No adoption, no default changed, no file on any reported
# code path edited. The averaging is injected by wrapping fuse_nig_product at
# runtime inside THIS process only.
#
# WHY THIS IS THE RIGHT INJECTION POINT. Two places consume the fused belief
# during training:
#   mesh.py:733  _sample_from_fused  -- the sampled training target is drawn
#                from the fused Student-t, whose scale depends on nu*, alpha*,
#                beta*. This is where summing evidence tightens the target.
#   mesh.py:768+ _aggregate          -- nig_product's gamma* and fused
#                certainty.
# Both call fuse_nig_product, as does runner.py for cell_fused_steps, so
# wrapping that single function makes training AND the saved fused arrays use
# w_i = 1/N consistently.
#
# NOTE gamma* is algebraically unchanged by the reweighting (the 1/N cancels),
# so the training difference is carried entirely by the SPREAD of the sampled
# target -- which is exactly the mechanism Section 5.4.3 identifies.
#
# Run: python -u experiments/section5_2/run_averaged_training_diagnostic.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

import beliefmesh.fusion.nig_product as nigmod
_ORIG = nigmod.fuse_nig_product


def averaged_fuse(beliefs, weights=None):
    """w_i = 1/N when no explicit weights are given: n_eff = 1, so
    nu* = mean(nu_i) instead of sum(nu_i). Single contributor is untouched."""
    if weights is None and len(beliefs) > 1:
        weights = [1.0 / len(beliefs)] * len(beliefs)
    return _ORIG(beliefs, weights)


def check_wrapper():
    b = [(0.10, 2.0, 3.0, 0.05), (0.12, 4.0, 3.5, 0.06), (0.09, 1.0, 2.5, 0.04)]
    s = _ORIG(b)
    a = averaged_fuse(b)
    print("wrapper check")
    print("  product  nu*=%.4f alpha*=%.4f beta*=%.6f gamma*=%.6f" % (s[1], s[2], s[3], s[0]))
    print("  averaged nu*=%.4f alpha*=%.4f beta*=%.6f gamma*=%.6f" % (a[1], a[2], a[3], a[0]))
    print("  nu* sum->mean: %.4f -> %.4f (expect mean of %s = %.4f)"
          % (s[1], a[1], [x[1] for x in b], np.mean([x[1] for x in b])))
    print("  gamma* unchanged: |d| = %.3e" % abs(s[0] - a[0]))
    one = [(0.3, 2.0, 3.0, 0.05)]
    assert averaged_fuse(one) == _ORIG(one), "single-contributor case must be untouched"
    assert abs(a[1] - np.mean([x[1] for x in b])) < 1e-12, "nu* is not the mean"
    assert abs(s[0] - a[0]) < 1e-12, "gamma* should be invariant"
    print("  PASSED\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    seeds = [int(x) for x in ap.parse_args().seeds.split(",")]

    check_wrapper()
    nigmod.fuse_nig_product = averaged_fuse
    import beliefmesh.node.mesh as meshmod
    import stage6_spatial_mesh.runner as runnermod
    for m in (meshmod, runnermod):
        if hasattr(m, "fuse_nig_product"):
            m.fuse_nig_product = averaged_fuse
            print("rebound fuse_nig_product in %s" % m.__name__)
    # mesh.py imports the symbol INSIDE _sample_from_fused and _aggregate, so
    # it resolves the patched module attribute at call time -- covered without
    # a module-level rebind.

    from stage6_spatial_mesh.runner import run_mesh_experiment
    from stage6_spatial_mesh.run_offset_experiments import (build_dynamic_offset_field,
                                                            build_random_wander_path)
    from beliefmesh.config import load_config

    ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
    CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
    EXCLUDED = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
    BASE_SEED, LAM, LR, N_WEAR = 42, 5.0, 3e-5, 3
    out = Path("runs/chapter5_v2/s5_diag_avg_training")

    cfg = load_config()
    cfg.model.lr = LR
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)

    for seed in seeds:
        # wearable seeding VERBATIM from run_mesh_sampled.py, so the diagnostic
        # and the product baseline see identical wearable paths at every seed
        base = 200 if seed == BASE_SEED else seed + 200
        paths = [build_random_wander_path(T, G, seed=base + i) for i in range(N_WEAR)]
        tag = "avg_training_seed%d" % seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        res = run_mesh_experiment(
            cfg, condition=tag, run_dir=out / tag,
            all_grids=np.full((T, G, G), 0.5), wearable_paths=paths,
            node_centres=centres, fov_size=7, mode="nig_product",
            baseline_checkpoint=CKPT, n_wearable_samples=1, n_train_repeats=1,
            title="Averaged fusion in TRAINING (diagnostic, seed %d)" % seed,
            offset_field=field, env_seed=seed,
            excluded_rotation_ranges=EXCLUDED,
            lam=LAM, wearable_policy=None, policy_step_size=0.4,
            sample_target=True, draws_per_target=1, track_disagreement=True,
        )
        print("\n### %s last50=%.5f" % (tag, res["mean_mse_last_50"]), flush=True)
    print("=== DONE ===")


if __name__ == "__main__":
    main()
