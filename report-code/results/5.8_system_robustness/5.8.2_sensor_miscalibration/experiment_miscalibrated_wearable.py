# MISCALIBRATED WEARABLE: one sensor is wrong and does not know it.
#
# Wearable 2 (0-indexed) reports every measurement offset by a CONSTANT
# +20 degrees for the whole run. The other two report correctly. Crucially
# wearable_reliability is left None, so all three enter training at certainty
# 1.0 -- the corrupted sensor does not flag itself. That is the deployment
# case; the existing sensor_reliability runs test a sensor that DOES report
# its own unreliability, which is the easier problem.
#
# Bias, not jitter: Gaussian noise averages out over samples and can be
# absorbed as variance, a constant offset cannot. It is applied to the
# REPORTED label only -- evaluation always scores against environment truth,
# so the measured damage is real error, not a relabelled target.
#
# CONFIGURATION is run_mesh_sampled.run_one's verbatim, which is what produced
# the five-seed baseline this is compared against (s531_b2 nig_product_sampled):
# 36 nodes, dynamic offset world, nig_product, sampled targets, draws 1,
# lam 5.0, lr 3e-5, exclusions applied. The ONLY difference is the bias list.
#
# Run: python -u experiments/section5_2/run_miscalibrated_wearable.py --verify
#      python -u experiments/section5_2/run_miscalibrated_wearable.py --root <dir>

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")
import tempfile as _tf
_SCRATCH = _Path(_tf.gettempdir()) / "beliefmesh_verify"

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from beliefmesh.simulation.runner import run_mesh_experiment
from beliefmesh.simulation.offset_fields import (build_dynamic_offset_field,
                                                        build_random_wander_path)

from beliefmesh.config import load_config

ENV = ENVIRONMENT_DIR
CKPT = CHECKPOINTS["baseline"]
N_WEARABLES, LR, LAM, SEED = 3, 3e-5, 5.0, 42
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
SEEDS = [42, 1042, 2042, 3042, 4042]

BAD_W = 2                      # which wearable is miscalibrated
BIAS_DEG = 20.0
BIAS = [0.0] * N_WEARABLES
BIAS[BAD_W] = BIAS_DEG / 180.0   # normalised label units, 1.0 = 180 deg

BASELINE = Path(ART + "/5.3_belief_aggregation/5.3.2_mesh_scale/aggregation_arms")
DEFAULT_ROOT = ART + "/5.8_system_robustness/5.8.2_sensor_miscalibration/miscalibrated_wearable"
FLOOR_PCT = 0.2


def run_one(seed, root, bias, tag):
    cfg = load_config()
    cfg.model.lr = LR
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    centres = np.load(ENV / "node_centres.npy")
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    all_grids = np.full((T, G, G), 0.5)
    field = build_dynamic_offset_field(G, T)
    wearable_seed_base = 200 if seed == SEED else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i)
             for i in range(N_WEARABLES)]

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    res = run_mesh_experiment(
        cfg, condition=tag, run_dir=root / tag,
        all_grids=all_grids, wearable_paths=paths, node_centres=centres,
        fov_size=7, mode="nig_product", baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title="Miscalibrated wearable (seed=%d, bias=%s)" % (seed, bias),
        offset_field=field, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED_RANGES,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        track_disagreement=True,
        sample_target=True, draws_per_target=1,
        wearable_label_bias=bias,
    )
    # the corrupted wearable's realised path, needed for the distance breakdown
    np.save(root / tag / "bad_wearable_path.npy", np.asarray(paths[BAD_W]))
    cms = np.load(root / tag / "cell_mse_steps.npy")
    print("### %s: last50=%.5f whole_run=%.5f"
          % (tag, res["mean_mse_last_50"], float(np.nanmean(cms))), flush=True)
    return res


def verify():
    """bias=None must reproduce the stored baseline. The parameter is new, so
    what is being proven is that ADDING it changed nothing when it is off."""
    print("=" * 96)
    print("INERTNESS CHECK -- wearable_label_bias OFF, seed 42, vs stored baseline")
    print("=" * 96)
    from _shared.artefacts import has_array, load_array
    ref = BASELINE / "nig_product_sampled_seed42"
    if not has_array(ref, "cell_mse_steps"):
        raise SystemExit("stored baseline missing: %s" % ref)
    out = _SCRATCH / "miscal_inert"
    run_one(42, out, None, "verify_bias_off_seed42")
    a = load_array(ref, "cell_mse_steps")
    b = np.load(out / "verify_bias_off_seed42" / "cell_mse_steps.npy")
    m = ~np.isnan(a)
    wa, wb = np.nanmean(a), np.nanmean(b)
    f = lambda x: np.nanmean(x[-50:])
    print("\n  NaN pattern match : %s" % np.array_equal(np.isnan(a), np.isnan(b)))
    print("  whole-run MSE     : %.6f vs %.6f  -> %+.3f%%  (floor %.1f%%)"
          % (wa, wb, 100 * (wb - wa) / wa, FLOOR_PCT))
    print("  last-50 MSE       : %.6f vs %.6f  -> %+.3f%%"
          % (f(a), f(b), 100 * (f(b) - f(a)) / f(a)))
    print("  per-cell L1       : %.3f%%"
          % (np.abs(a[m] - b[m]).sum() / np.abs(a[m]).sum() * 100))
    ok = abs(100 * (wb - wa) / wa) <= FLOOR_PCT
    print("\n  WITHIN THE REPRODUCTION FLOOR: %s" % ("YES" if ok else "NO"))

    # and prove the bias is LIVE, so an inert patch cannot pass as a result
    print("\n  bias vector that will be used: %s  (wearable %d, %+.0f deg)"
          % (BIAS, BAD_W, BIAS_DEG))
    assert any(b_ != 0 for b_ in BIAS), "bias vector is all zeros"
    print("  applied at BOTH label sites in mesh.py (grep wearable_label_bias)")
    return ok


def main(root):
    t0 = time.time()
    for i, seed in enumerate(SEEDS):
        print("\n>>> [%d/%d] seed %d  (elapsed %.1f min)"
              % (i + 1, len(SEEDS), seed, (time.time() - t0) / 60.0), flush=True)
        run_one(seed, root, BIAS, "miscal_seed%d" % seed)
    print("\n=== MISCALIBRATED WEARABLE COMPLETE: %.1f min ==="
          % ((time.time() - t0) / 60.0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT)
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify() else 1)
    main(a.root)
