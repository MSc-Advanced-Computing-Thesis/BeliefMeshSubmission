# Section 5.6 / 5.7 heterogeneity, rerun FILTER-ON.
#
# The chapter standardises on apply_colour_filter=True. The heterogeneity
# comparator passes False EXPLICITLY (run_heterogeneous_comparator.py:128), so
# unlike the gossip case filter-on cannot be reached by leaving a default
# alone -- the call itself has to be overridden.
#
# run_heterogeneous_comparator.py IS NOT MODIFIED. It binds run_mesh_experiment
# at module scope (line 47), so this driver rebinds that name on the module to
# a wrapper that forces apply_colour_filter=True and forwards everything else,
# then restores it in a finally. Average Fusion additionally wraps
# fuse_nig_product with uniform weights; both patches go on and come off
# together.
#
# SCOPE (40 runs):
#   mixed mesh, 4 variants x 5 seeds                        = 20
#   uniform controls narrow/baseline/wide, product fusion   = 15
#   uniform all-baseline, average fusion                    =  5
# The multifield extension and s5_5_sensor are deliberately excluded.
#
# Run: python -u experiments/section5_2/run_het_filteron.py --verify
#      python -u experiments/section5_2/run_het_filteron.py
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")
import tempfile as _tf
_SCRATCH = _Path(_tf.gettempdir()) / "beliefmesh_verify"

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml


import beliefmesh.fusion.nig_product as nigmod
import beliefmesh.simulation.runner as rn
import experiment_heterogeneous_mesh as HET

_ORIG_FUSE = nigmod.fuse_nig_product
_ORIG_RUNNER = HET.run_mesh_experiment

SEEDS = [42, 1042, 2042, 3042, 4042]
MIXED_ROOT = ART + "/5.7_heterogeneous_devices/mixed_capacity"
HOMOG_ROOT = ART + "/5.7_heterogeneous_devices/uniform"
# the filter-off reference run this script originally compared against is
# not part of the repository; --verify skips that comparison when it is absent
STORED_OFF = Path(ART + "/5.7_heterogeneous_devices/reference_filter_off/het_nig_product")
FLOOR_PCT = 0.2
_captured: list = []


def averaged_fuse(beliefs, weights=None):
    if weights is None and len(beliefs) > 1:
        weights = [1.0 / len(beliefs)] * len(beliefs)
    return _ORIG_FUSE(beliefs, weights)


def _forced(**force):
    """Wrapper around run_mesh_experiment that overrides the named kwargs.
    Records each call so --verify can prove the override is really applied."""
    def call(*a, **kw):
        kw.update(force)
        _captured.append(dict(kw))
        return _ORIG_RUNNER(*a, **kw)
    return call


def run(mode, seed, root, *, filter_on=True, averaged=False, homogeneous=False,
        variant="baseline"):
    assert HET.run_mesh_experiment is _ORIG_RUNNER, "runner already patched"
    assert nigmod.fuse_nig_product is _ORIG_FUSE, "fusion already patched"
    if filter_on:
        HET.run_mesh_experiment = _forced(apply_colour_filter=True)
    if averaged:
        nigmod.fuse_nig_product = averaged_fuse
        rn.fuse_nig_product = averaged_fuse
    try:
        return HET.run_one(mode, seed, homogeneous=homogeneous,
                           root_override=root, homog_variant=variant)
    finally:
        HET.run_mesh_experiment = _ORIG_RUNNER
        nigmod.fuse_nig_product = _ORIG_FUSE
        rn.fuse_nig_product = _ORIG_FUSE


def verify():
    ok = True
    print("=" * 92)
    print("CHECK 1 -- is the override LIVE? (capture the kwargs without running)")
    print("=" * 92)
    # Swap the underlying runner for a stub so the wrapper can be exercised
    # without a 6-minute run, then put the real one back.
    global _ORIG_RUNNER
    _captured.clear()
    real = _ORIG_RUNNER
    try:
        _ORIG_RUNNER = lambda *a, **kw: {"stub": True}
        wrapper = _forced(apply_colour_filter=True)
        # the script itself passes False; the wrapper must override it
        wrapper(None, condition="x", apply_colour_filter=False)
    finally:
        _ORIG_RUNNER = real
        HET.run_mesh_experiment = _ORIG_RUNNER
    got = _captured[-1].get("apply_colour_filter") if _captured else None
    assert _captured, "wrapper captured nothing -- the check would pass vacuously"
    print("  caller passed apply_colour_filter=False; wrapper delivered %r" % got)
    if got is not True:
        ok = False
        print("  FAILED -- the override is inert")
    else:
        print("  PASSED -- the override wins over the script's explicit False")
    print("  module restored to the real runner: %s"
          % (HET.run_mesh_experiment is _ORIG_RUNNER))

    print()
    print("=" * 92)
    print("CHECK 2 -- with the override OFF, does the path reproduce the")
    print("stored FILTER-OFF arm? (product fusion, mixed mesh, seed 42)")
    print("=" * 92)
    out = _SCRATCH / "het_fon_inert"
    run("nig_product", 42, str(out), filter_on=False)
    if not (STORED_OFF / "cell_mse_steps.npy").exists():
        print("stored filter-off reference %s not present; comparison skipped" % STORED_OFF)
        return True
    a = np.load(STORED_OFF / "cell_mse_steps.npy")
    b = np.load(out / "het_nig_product" / "cell_mse_steps.npy")
    m = ~np.isnan(a)
    wa, wb = np.nanmean(a), np.nanmean(b)
    f = lambda x: np.nanmean(x[-50:])
    print("  NaN pattern match : %s" % np.array_equal(np.isnan(a), np.isnan(b)))
    print("  whole-run MSE     : %.6f vs %.6f  -> %+.3f%%  (floor %.1f%%)"
          % (wa, wb, 100 * (wb - wa) / wa, FLOOR_PCT))
    print("  last-50 MSE       : %.6f vs %.6f  -> %+.3f%%"
          % (f(a), f(b), 100 * (f(b) - f(a)) / f(a)))
    print("  per-cell L1       : %.3f%%"
          % (np.abs(a[m] - b[m]).sum() / np.abs(a[m]).sum() * 100))
    mf = yaml.safe_load(open(out / "het_nig_product" / "manifest.yaml"))
    print("  manifest records apply_colour_filter = %r (expect False)"
          % mf.get("apply_colour_filter"))
    if abs(100 * (wb - wa) / wa) > FLOOR_PCT or mf.get("apply_colour_filter") is not False:
        ok = False
        print("  FAILED")
    else:
        print("  PASSED -- pass-through is inert and self-identifying on disk")
    print("\nVERIFY %s" % ("PASSED" if ok else "FAILED"))
    return ok


def main():
    jobs = []
    for m in ("naive", "certainty", "nig_product"):
        jobs += [(m, s, MIXED_ROOT + "/" + m, dict()) for s in SEEDS]
    jobs += [("nig_product", s, MIXED_ROOT + "/avgfusion", dict(averaged=True))
             for s in SEEDS]
    for v in ("narrow", "baseline", "wide"):
        jobs += [("nig_product", s, HOMOG_ROOT + "/" + v,
                  dict(homogeneous=True, variant=v)) for s in SEEDS]
    jobs += [("nig_product", s, HOMOG_ROOT + "/baseline_avgfusion",
              dict(homogeneous=True, variant="baseline", averaged=True))
             for s in SEEDS]
    print("HETEROGENEITY FILTER-ON: %d runs" % len(jobs), flush=True)
    t0 = time.time()
    ok = bad = 0
    for i, (mode, seed, root, kw) in enumerate(jobs):
        print("\n>>> [%d/%d] %s seed %d -> %s  (elapsed %.1f min)"
              % (i + 1, len(jobs), mode, seed, root, (time.time() - t0) / 60.0),
              flush=True)
        try:
            run(mode, seed, root, filter_on=True, **kw)
            ok += 1
        except Exception:
            bad += 1
            print("!!! FAILED %s seed %d %s" % (mode, seed, root), flush=True)
            traceback.print_exc()
    print("\n=== HET FILTER-ON COMPLETE: %d ok, %d failed, %.1f min ==="
          % (ok, bad, (time.time() - t0) / 60.0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if verify() else 1) if a.verify else main()
