# Average Fusion on the heterogeneity configuration and on the multifield runs.
#
# NEITHER GENERATOR IS MODIFIED. run_heterogeneous_comparator.py and
# run_multifield_heterogeneous.py are imported as modules and their
# fuse_nig_product binding is swapped for a uniform-weight wrapper for the
# duration of a run, then restored in a finally. mode stays "nig_product"
# because the wrapper is not a kwarg, so the runner receives byte-identical
# arguments to the stored nig_product arm -- which is what makes the
# inertness check meaningful.
#
# mesh.py imports fuse_nig_product INSIDE the functions that use it, so
# rebinding on the fusion module reaches the training path; runner.py binds it
# at import, so that module is rebound too. Both are restored.
#
# SCOPE (minimal, as agreed):
#   het        -- heterogeneous mixed mesh, 5 seeds
#   multifield -- one seed per field, following SEED_FIELD
# The homogeneous control is skipped: Section 5.3.2's mesh table already
# reports Average Fusion under homogeneous conditions.
#
# Run: python -u experiments/section5_2/run_avgfusion_het_multifield.py --verify-inert
#      python -u experiments/section5_2/run_avgfusion_het_multifield.py --which both

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR.parent / "src"))

import beliefmesh.fusion.nig_product as nigmod
import stage6_spatial_mesh.runner as rn
import stage6_spatial_mesh.run_heterogeneous_comparator as HET
import stage6_spatial_mesh.run_multifield_heterogeneous as MF

_ORIG = nigmod.fuse_nig_product

HET_ROOT = "runs/chapter5_v2/s5_6_het_avgfusion"
MF_ROOT = "runs/chapter5_v2/s5_6_multifield_avgfusion"
MF_CERT_ROOT = "runs/chapter5_v2/s5_6_multifield_certainty"
# full regeneration under current code (2026-09-04): every arm of both
# configurations plus the three uniform controls, so the section is
# internally consistent rather than spanning three code vintages.
RG_HET = "runs/chapter5_v2/s5_6_het_rg"
RG_MF = "runs/chapter5_v2/s5_6_multifield_rg"
RG_HOMOG = "runs/chapter5_v2/s5_6_homog_rg"
HOMOG_VARIANTS = ["narrow", "baseline", "wide"]
# Homogeneous ALL-BASELINE mesh under Average Fusion, filter OFF. The
# existing homogeneous Average Fusion runs (s5_diag_avg_training) come from a
# generator that leaves apply_colour_filter at runner's default True, so they
# cannot sit in a figure beside the filter-off heterogeneous arms.
RG_HOMOG_AVG = "runs/chapter5_v2/s5_6_homog_avgfusion_rg"
HET_STORED = Path("runs/chapter5_v2/s5_6_het/het_nig_product")
MF_STORED = Path("runs/chapter5_v2/s5_6_multifield/het/"
                 "nig_product_seed42_field0_original")
SEEDS = [42, 1042, 2042, 3042, 4042]
FLOOR_PCT = 0.2


def averaged_fuse(beliefs, weights=None):
    """w_i = 1/N -> n_eff = 1, nu* = mean(nu_i). gamma* is unchanged."""
    if weights is None and len(beliefs) > 1:
        weights = [1.0 / len(beliefs)] * len(beliefs)
    return _ORIG(beliefs, weights)


def with_wrapper(fn, patched=True):
    assert nigmod.fuse_nig_product is _ORIG, "fusion already patched"
    if patched:
        nigmod.fuse_nig_product = averaged_fuse
        rn.fuse_nig_product = averaged_fuse
    try:
        return fn()
    finally:
        nigmod.fuse_nig_product = _ORIG
        rn.fuse_nig_product = _ORIG


def run_het(seed, root, patched=True):
    return with_wrapper(lambda: HET.run_one("nig_product", seed,
                                            homogeneous=False,
                                            root_override=root), patched)


def run_mf(seed, root, patched=True):
    return with_wrapper(lambda: MF.run_one("nig_product", seed,
                                           homogeneous=False,
                                           root_override=root), patched)


def run_het_mode(mode, seed, root, patched=False):
    return with_wrapper(lambda: HET.run_one(mode, seed, homogeneous=False,
                                            root_override=root), patched)


def run_mf_mode(mode, seed, root, patched=False):
    return with_wrapper(lambda: MF.run_one(mode, seed, homogeneous=False,
                                           root_override=root), patched)


def run_homog(variant, seed, root, patched=False):
    return with_wrapper(lambda: HET.run_one("nig_product", seed,
                                            homogeneous=True,
                                            root_override=root,
                                            homog_variant=variant), patched)


def run_mf_certainty(seed, root):
    """Certainty on multifield. run_multifield_heterogeneous declares
    MODES = ["naive", "nig_product"] and --mode is choices=MODES, so the CLI
    cannot select it -- but run_one takes mode as a plain argument and passes
    it straight through, and 'certainty' is a valid mesh mode. Calling run_one
    directly adds the arm without editing the reported script. No wrapper:
    certainty is its own aggregation rule, not a fusion weighting."""
    return with_wrapper(lambda: MF.run_one("certainty", seed,
                                           homogeneous=False,
                                           root_override=root), patched=False)


def compare(stored, new, label):
    a = np.load(Path(stored) / "cell_mse_steps.npy")
    b = np.load(Path(new) / "cell_mse_steps.npy")
    m = ~np.isnan(a)
    wa, wb = np.nanmean(a), np.nanmean(b)
    f = lambda x: np.nanmean(x[-50:])
    print("  %s" % label)
    print("    NaN pattern match : %s" % np.array_equal(np.isnan(a), np.isnan(b)))
    print("    whole-run MSE     : %.6f vs %.6f  -> %+.3f%%  (floor %.1f%%)"
          % (wa, wb, 100 * (wb - wa) / wa, FLOOR_PCT))
    print("    last-50 MSE       : %.6f vs %.6f  -> %+.3f%%"
          % (f(a), f(b), 100 * (f(b) - f(a)) / f(a)))
    print("    per-cell L1       : %.3f%%"
          % (np.abs(a[m] - b[m]).sum() / np.abs(a[m]).sum() * 100))
    ok = abs(100 * (wb - wa) / wa) <= FLOOR_PCT
    print("    WITHIN FLOOR      : %s" % ("YES" if ok else "NO"))
    return ok


def verify():
    print("=" * 92)
    print("INERTNESS -- wrapper OFF must reproduce the stored nig_product arm")
    print("=" * 92)
    # the wrapper must be LIVE when on, or an inert patch passes as a result
    b = [(0.10, 4.0, 3.0, 0.20), (-0.05, 9.0, 5.0, 0.40)]
    gp, nup, ap, bp = _ORIG(b)
    ga, nua, aa, ba = averaged_fuse(b)
    print("  wrapper check: gamma* %.8f vs %.8f (identical %s), nu* %.4f -> %.4f"
          % (gp, ga, abs(gp - ga) < 1e-12, nup, nua))
    assert abs(gp - ga) < 1e-12 and nua < nup - 1e-9, "wrapper is inert or wrong"
    print("  wrapper is LIVE\n")

    s = Path("runs/scratch/avgfus_inert")
    run_het(42, str(s / "het"), patched=False)
    ok1 = compare(HET_STORED, s / "het" / "het_nig_product", "HETEROGENEOUS, seed 42")
    print()
    run_mf(42, str(s / "mf"), patched=False)
    ok2 = compare(MF_STORED, s / "mf" / "het" / "nig_product_seed42_field0_original",
                  "MULTIFIELD, seed 42 / field0_original")
    print("\nVERIFY %s" % ("PASSED" if (ok1 and ok2) else "FAILED"))
    return ok1 and ok2


def main(which):
    t0 = time.time()
    jobs = []
    if which in ("het", "both"):
        jobs += [("het", s, HET_ROOT) for s in SEEDS]
    if which in ("multifield", "both"):
        jobs += [("mf", s, MF_ROOT) for s in sorted(MF.SEED_FIELD)]
    if which in ("mf_certainty", "both"):
        jobs += [("mfc", s, MF_CERT_ROOT) for s in sorted(MF.SEED_FIELD)]
    if which == "regen":
        # product fusion and average fusion share mode="nig_product"; the
        # wrapper is what separates them, hence the patched flag in the tuple
        for m in ("naive", "certainty", "nig_product"):
            jobs += [("het:%s" % m, s, RG_HET + "/" + m) for s in SEEDS]
        jobs += [("het:avg", s, RG_HET + "/avgfusion") for s in SEEDS]
        for m in ("naive", "certainty", "nig_product"):
            jobs += [("mf:%s" % m, s, RG_MF + "/" + m) for s in sorted(MF.SEED_FIELD)]
        jobs += [("mf:avg", s, RG_MF + "/avgfusion") for s in sorted(MF.SEED_FIELD)]
        for v in HOMOG_VARIANTS:
            jobs += [("homog:%s" % v, s, RG_HOMOG + "/" + v) for s in SEEDS]
    if which == "homog_avg":
        jobs += [("homogavg", s, RG_HOMOG_AVG) for s in SEEDS]
    print("AVERAGE FUSION: %d runs -> %s" % (len(jobs), which), flush=True)
    ok = bad = 0
    for i, (kind, seed, root) in enumerate(jobs):
        print("\n>>> [%d/%d] %s seed %d  (elapsed %.1f min)"
              % (i + 1, len(jobs), kind, seed, (time.time() - t0) / 60.0), flush=True)
        try:
            if kind.startswith("het:"):
                m = kind.split(":", 1)[1]
                run_het_mode("nig_product" if m == "avg" else m, seed, root,
                             patched=(m == "avg"))
            elif kind.startswith("mf:"):
                m = kind.split(":", 1)[1]
                run_mf_mode("nig_product" if m == "avg" else m, seed, root,
                            patched=(m == "avg"))
            elif kind == "homogavg":
                run_homog("baseline", seed, root, patched=True)
            elif kind.startswith("homog:"):
                run_homog(kind.split(":", 1)[1], seed, root)
            else:
                {"het": run_het, "mf": run_mf,
                 "mfc": run_mf_certainty}[kind](seed, root)
            ok += 1
        except Exception:
            bad += 1
            print("!!! FAILED %s seed %d" % (kind, seed), flush=True)
            traceback.print_exc()
    print("\n=== AVGFUSION HET/MULTIFIELD COMPLETE: %d ok, %d failed, %.1f min ==="
          % (ok, bad, (time.time() - t0) / 60.0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-inert", action="store_true")
    ap.add_argument("--which",
                    choices=["het", "multifield", "mf_certainty", "both", "regen",
                             "homog_avg"],
                    default="both")
    a = ap.parse_args()
    if a.verify_inert:
        sys.exit(0 if verify() else 1)
    main(a.which)
