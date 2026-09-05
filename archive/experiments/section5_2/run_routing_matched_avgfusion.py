# MATCHED ROUTING COMPARISON UNDER AVERAGE FUSION IN TRAINING.
#
# Diagnostic only. Nothing here is adopted as a chapter convention.
#
# WHAT IS REUSED. build() and kwargs_for() are imported from
# run_routing_matched.py, not reimplemented, so the geometry, the dynamic
# offset field, lam=5.0, lr=3e-5, the 36-node stride-3 centres, the five
# seeds and -- critically -- the per-seed SHARED wearable paths are the same
# objects the Product Fusion comparison used. That script is not modified.
#
# THE ONLY DIFFERENCE is that fuse_nig_product is wrapped to supply uniform
# weights w_i = 1/N during TRAINING, giving nu* = mean(nu_i) and n_eff = 1
# instead of nu* = sum(nu_i). mode= stays "nig_product" because the wrapper
# is not a kwarg -- so the kwargs the runner receives are byte-identical to
# the Product Fusion arms', which --verify asserts.
#
# The patch is applied around each run and restored in a finally, and the
# module-level binding is asserted clean before the batch starts, so the
# wrapper cannot leak into anything else in the process.
#
# Run:    python -u experiments/section5_2/run_routing_matched_avgfusion.py --root <dir>
# Verify: python -u experiments/section5_2/run_routing_matched_avgfusion.py --verify

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beliefmesh.fusion.nig_product as nigmod
import stage6_spatial_mesh.runner as rn
from stage6_spatial_mesh.runner import run_mesh_experiment

import run_routing_matched as RM

_ORIG_FUSE = nigmod.fuse_nig_product
SEEDS = RM.SEEDS
ARMS = RM.ARMS
DEFAULT_ROOT = Path("runs/chapter5_v2/s5_9_routing_matched_avgfusion")


def averaged_fuse(beliefs, weights=None):
    """w_i = 1/N -> n_eff = 1, nu* = mean(nu_i). gamma* is unchanged."""
    if weights is None and len(beliefs) > 1:
        weights = [1.0 / len(beliefs)] * len(beliefs)
    return _ORIG_FUSE(beliefs, weights)


def run_all(root, seeds):
    assert nigmod.fuse_nig_product is _ORIG_FUSE, "fusion already patched"
    Path(root).mkdir(parents=True, exist_ok=True)
    out = {}
    for seed in seeds:
        shared = RM.build(seed)
        cfg = shared[0]
        for arm, policy in ARMS:
            kw = RM.kwargs_for(seed, arm, policy, root, shared)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            nigmod.fuse_nig_product = averaged_fuse
            rn.fuse_nig_product = averaged_fuse
            try:
                res = run_mesh_experiment(cfg, **kw)
            finally:
                nigmod.fuse_nig_product = _ORIG_FUSE
                rn.fuse_nig_product = _ORIG_FUSE
            cms = np.load(kw["run_dir"] / "cell_mse_steps.npy")
            whole = float(np.nanmean(cms))
            out.setdefault(arm, []).append((whole, res["mean_mse_last_50"]))
            print("### AVGFUSION %s seed=%d: last50=%.5f whole_run=%.5f"
                  % (arm, seed, res["mean_mse_last_50"], whole), flush=True)
    print("\n=== MATCHED ROUTING, AVERAGE FUSION IN TRAINING "
          "(dynamic offset world, 36 nodes stride 3, lam=5, lr=3e-5, "
          "%d seeds) ===" % len(seeds))
    for arm, _ in ARMS:
        v = out.get(arm, [])
        if v:
            w = np.array([x[0] for x in v]); l = np.array([x[1] for x in v])
            print("%-20s whole_run %.5f +/- %.5f   last50 %.5f +/- %.5f"
                  % (arm, w.mean(), w.std(ddof=1), l.mean(), l.std(ddof=1)))
    print("=== DONE ===")


def verify():
    """Two assertions. (1) The kwargs are identical to the Product Fusion
    comparison's -- same shared paths, same field, same geometry -- so the
    fusion rule is the only difference. (2) The wrapper actually changes the
    fused parameters, so an inert patch cannot pass as a result."""
    ok = RM.verify()
    print("\n--- AVERAGE FUSION PATCH ---")
    print("  mode kwarg stays 'nig_product'; the wrapper is not a kwarg, so the")
    print("  kwargs above are byte-identical to the Product Fusion arms'.")
    assert nigmod.fuse_nig_product is _ORIG_FUSE, "wrapper leaked at import"
    print("  module-level fuse_nig_product is UNPATCHED at rest: OK")

    beliefs = [(0.10, 4.0, 3.0, 0.20), (-0.05, 9.0, 5.0, 0.40)]
    gp, nup, ap, bp = _ORIG_FUSE(beliefs)
    ga, nua, aa, ba = averaged_fuse(beliefs)
    print("  product : gamma*=%.8f nu*=%.4f alpha*=%.4f beta*=%.6f" % (gp, nup, ap, bp))
    print("  average : gamma*=%.8f nu*=%.4f alpha*=%.4f beta*=%.6f" % (ga, nua, aa, ba))
    assert abs(gp - ga) < 1e-12, "gamma* should be IDENTICAL under the two rules"
    assert nua < nup - 1e-9, "wrapper is INERT -- nu* did not shrink"
    print("  gamma* identical (%.3e), nu* shrinks %.4f -> %.4f: patch is LIVE"
          % (abs(gp - ga), nup, nua))
    print("\nVERIFY %s" % ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify() else 1)
    run_all(a.root, [int(x) for x in a.seeds.split(",")])
