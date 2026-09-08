# MATCHED ROUTING COMPARISON -- uncertainty-guided vs random wearable policy.
#
# WHY A NEW SCRIPT RATHER THAN A PATCH. run_zoned_routing_test.py cannot be
# made to serve: it loads a STATIC offset field (dynamic_static_spatial_v2/
# field.npy), runs a single seed, and its result was read against an arm at a
# different node geometry. Every one of those has to change, so "inert against
# the existing invocation" is unsatisfiable by construction. That script is
# left byte-identical; this one is modelled on run_mesh_sampled.py -- the
# 5.3.1 generator -- so the comparison is matched to the rest of the chapter
# by construction rather than by reconstruction.
#
# WHAT MATCHED MEANS HERE. The two arms share: the dynamic offset field (the
# 5.3.1 world), lam=5.0, lr=3e-5, 36 nodes at stride 3 (environment_v2/
# node_centres.npy, the geometry every other section uses), the same five
# seeds, and -- per seed -- THE SAME wearable path array, so the random arm's
# path and the guided arm's starting conditions are one draw, not two.
# wearable_policy is the only difference.
#
# Run:    python -u experiments/section5_2/run_routing_matched.py --root <dir>
# Verify: python -u experiments/section5_2/run_routing_matched.py --verify

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

from beliefmesh.simulation.runner import run_mesh_experiment
from beliefmesh.simulation.offset_fields import (build_dynamic_offset_field,
                                                        build_random_wander_path)

from beliefmesh.config import load_config

# ── verbatim from run_mesh_sampled.py (the 5.3.1 generator) ───────────────
ENV = ENVIRONMENT_DIR
CKPT = CHECKPOINTS["baseline"]
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
SEED = 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
SEEDS = [42, 1042, 2042, 3042, 4042]
DEFAULT_ROOT = Path(ART + "/5.6_uncertainty_guided_measurement/routing_product_fusion")

ARMS = [("random", None), ("uncertainty_guided", "uncertainty_guided")]


def build(seed: int):
    """Everything both arms share at this seed. Called ONCE per seed, so the
    two arms cannot receive independently drawn paths."""
    cfg = load_config()
    cfg.model.lr = LR
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
             for i in range(N_WEARABLES)]
    return cfg, centres, field, paths, T, G


def kwargs_for(seed, arm, policy, root, shared):
    cfg, centres, field, paths, T, G = shared
    tag = "%s_seed%d" % (arm, seed)
    return dict(
        condition=tag, run_dir=Path(root) / tag,
        all_grids=np.full((T, G, G), 0.5), wearable_paths=paths,
        node_centres=centres, fov_size=7, mode="nig_product",
        baseline_checkpoint=CKPT, n_wearable_samples=1, n_train_repeats=1,
        title="Matched routing (%s, seed=%d)" % (arm, seed),
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=policy, policy_step_size=0.4,
    )


def run_all(root, seeds):
    Path(root).mkdir(parents=True, exist_ok=True)
    out = {}
    for seed in seeds:
        shared = build(seed)
        cfg = shared[0]
        for arm, policy in ARMS:
            kw = kwargs_for(seed, arm, policy, root, shared)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            res = run_mesh_experiment(cfg, **kw)
            cms = np.load(kw["run_dir"] / "cell_mse_steps.npy")
            whole = float(np.nanmean(cms))
            out.setdefault(arm, []).append((whole, res["mean_mse_last_50"]))
            print("### %s seed=%d: last50=%.5f whole_run=%.5f"
                  % (arm, seed, res["mean_mse_last_50"], whole), flush=True)
    print("\n=== MATCHED ROUTING (dynamic offset world, 36 nodes stride 3, "
          "lam=5, lr=3e-5, %d seeds) ===" % len(seeds))
    for arm, _ in ARMS:
        v = out.get(arm, [])
        if v:
            w = np.array([x[0] for x in v]); l = np.array([x[1] for x in v])
            print("%-20s whole_run %.5f +/- %.5f   last50 %.5f +/- %.5f"
                  % (arm, w.mean(), w.std(ddof=1), l.mean(), l.std(ddof=1)))
    print("=== DONE ===")


def verify():
    """Capture the kwargs both arms WOULD receive and assert they differ in
    nothing but wearable_policy. Returns the captured calls so an empty
    capture cannot pass as agreement."""
    calls = []
    for seed in SEEDS:
        shared = build(seed)
        for arm, policy in ARMS:
            calls.append((seed, arm, kwargs_for(seed, arm, policy, "/tmp/x", shared)))

    n_expected = len(SEEDS) * len(ARMS)
    print("captured %d calls (expected %d)" % (len(calls), n_expected))
    assert len(calls) == n_expected and n_expected > 0, "NO CALLS CAPTURED"

    def eq(a, b):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b))
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
        return a == b

    ok = True
    for i in range(0, len(calls), 2):
        (s1, a1, k1), (s2, a2, k2) = calls[i], calls[i + 1]
        assert s1 == s2
        diff = sorted(k for k in set(k1) | set(k2) if not eq(k1.get(k), k2.get(k)))
        expected = ["condition", "run_dir", "title", "wearable_policy"]
        status = "OK" if diff == expected else "MISMATCH"
        if diff != expected:
            ok = False
        print("  seed %-5d %-8s: differing kwargs = %s  [%s]"
              % (s1, "%s|%s" % (a1, a2), diff, status))
    print("\n  (condition/run_dir/title differ only because they NAME the arm;")
    print("   wearable_policy is the experimental difference. Anything else")
    print("   appearing above would be a confound.)")

    shapes = {k: (np.asarray(v).shape if isinstance(v, (np.ndarray, list)) else v)
              for k, v in calls[0][2].items()
              if k in ("node_centres", "all_grids", "offset_field", "wearable_paths")}
    print("\n  geometry check: node_centres %s, grids %s, offset_field %s, paths %s"
          % (shapes["node_centres"], shapes["all_grids"],
             shapes["offset_field"], shapes["wearable_paths"]))
    d = calls[0][2]["offset_field"]
    print("  offset field is DYNAMIC: max |f[t]-f[0]| = %.1f deg (static would be 0.0)"
          % float(np.abs(np.asarray(d) - np.asarray(d)[0]).max()))
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
