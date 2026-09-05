# Section 5.5 parameter-exchange comparison across FIVE OFFSET FIELDS.
#
# Four arms (fusion, gossip_uniform, gossip_weighted, fedavg_global) x five
# fields x ONE seed = 20 runs.
#
# run_gossip_comparator.py IS NOT MODIFIED. It hardcodes
#   field = build_dynamic_offset_field(G, T)          (line 73)
# and exposes no field option, but it imports that name into its own module
# namespace, so this driver rebinds the name on the MODULE for the duration of
# a run and restores it in a finally. Same pattern as the average-fusion
# routing runs. The reported script stays byte-identical.
#
# SEED IS HELD AT 42 FOR EVERY FIELD, deliberately. The multifield
# heterogeneity runs pair each field with a different seed (SEED_FIELD in
# run_multifield_heterogeneous.py), which is fine for their purpose but would
# make "spread across environments" untrue here -- the spread would mix field
# variation with rotation/wearable-path variation. Holding the seed fixed
# makes the field the only thing that changes, so the reported sd is across
# ENVIRONMENTS and nothing else. It also means field0/seed42 reproduces the
# already-reported run, which is what --verify-inert exploits.
#
# Because the seed is fixed, these numbers are NOT comparable with the
# five-seed table on the single field: that one varies rotations and paths on
# one environment, this one varies the environment with rotations and paths
# held fixed. Neither is a superset of the other.
#
# Run: python -u experiments/section5_2/run_gossip_multifield.py --verify-inert
#      python -u experiments/section5_2/run_gossip_multifield.py --root <dir>

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

import stage6_spatial_mesh.run_gossip_comparator as GC
from stage6_spatial_mesh.offset_field_variants import FIELDS, build_field
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field

_ORIG_BUILDER = GC.build_dynamic_offset_field

SEED = 42
SEEDS = [42, 1042, 2042, 3042, 4042]
FIELD_NAMES = ["field0_original", "field1_seed11", "field2_seed23",
               "field3_seed37", "field4_seed51_dual"]
MODES = ["fusion", "gossip_uniform", "gossip_weighted", "fedavg_global"]
DEFAULT_ROOT = "runs/chapter5_v2/s5_5_gossip_multifield"
# the already-reported single-field runs, used only by --verify-inert
STORED = Path("runs/chapter5_v2/s5_5_gossip_fusion/gossip_cmp_fusion")
FLOOR_PCT = 0.2      # established GPU-nondeterminism floor on whole-run MSE


def tag_for(mode: str, seed: int) -> str:
    """run_gossip_comparator's own naming, reproduced so existing runs can be
    detected. Seed 42 is unsuffixed there, which is why the 20 runs already on
    disk slot into the crossed grid without collision."""
    return "gossip_cmp_%s" % mode if seed == SEED else "gossip_cmp_%s_seed%d" % (mode, seed)


def run_one_on_field(mode: str, field_name: str, root: Path, seed: int = SEED):
    """One arm on one field at one seed. The rebind is the ONLY difference from
    calling run_gossip_comparator directly, and it is undone before returning."""
    assert GC.build_dynamic_offset_field is _ORIG_BUILDER, "builder already patched"
    GC.ROOT = Path(root)
    GC.build_dynamic_offset_field = lambda g, t: build_field(field_name, g, t)
    try:
        return GC.run_one(mode, seed)
    finally:
        GC.build_dynamic_offset_field = _ORIG_BUILDER


def verify_inert(root: Path):
    """field0 must BE the reported field, and running through the rebind must
    reproduce the stored arrays. Checked in that order so a builder mismatch is
    distinguished from a run-to-run difference."""
    G, Tn = 22, 390
    print("=" * 96)
    print("INERTNESS CHECK -- field0_original, seed %d, arm fusion" % SEED)
    print("=" * 96)
    ref = np.asarray(_ORIG_BUILDER(G, Tn))
    got = np.asarray(build_field("field0_original", G, Tn))
    same = np.array_equal(got, ref)
    print("1) build_field('field0_original') vs build_dynamic_offset_field:")
    print("   bit-identical %s   max|diff| %.3e   sums %.10f / %.10f"
          % (same, np.abs(got - ref).max(), ref.sum(), got.sum()))
    assert same, "field0_original is NOT the reported field"
    print("   PASSED -- the rebind cannot change the environment for field0")

    if not (STORED / "cell_mse_steps.npy").exists():
        raise SystemExit("stored reference missing: %s" % STORED)
    print("\n2) running arm 'fusion' on field0 through the rebind ...", flush=True)
    run_one_on_field("fusion", "field0_original", root)
    new = root / "gossip_cmp_fusion"

    a = np.load(STORED / "cell_mse_steps.npy")
    b = np.load(new / "cell_mse_steps.npy")
    m = ~np.isnan(a)
    wa, wb = np.nanmean(a), np.nanmean(b)
    l1 = np.abs(a[m] - b[m]).sum() / np.abs(a[m]).sum() * 100
    f = lambda x: np.nanmean(x[340:390])
    print("\n3) stored vs rebuilt, cell_mse_steps:")
    print("   NaN pattern match : %s" % np.array_equal(np.isnan(a), np.isnan(b)))
    print("   whole-run MSE     : %.6f vs %.6f  -> %+.3f%%  (floor %.1f%%)"
          % (wa, wb, 100 * (wb - wa) / wa, FLOOR_PCT))
    print("   last-50 MSE       : %.6f vs %.6f  -> %+.3f%%"
          % (f(a), f(b), 100 * (f(b) - f(a)) / f(a)))
    print("   per-cell L1       : %.3f%%" % l1)
    ok = abs(100 * (wb - wa) / wa) <= FLOOR_PCT
    print("\n   WITHIN THE REPRODUCTION FLOOR: %s" % ("YES" if ok else "NO"))
    if not ok:
        print("   -> NOT proceeding. The rebind is not inert, or the machine's")
        print("      floor differs from the established %.1f%%." % FLOOR_PCT)
    return ok


def main(root):
    """Crossed design: every field x every seed x every arm. Runs already on
    disk are REUSED, not redone -- the 20 seed-42 runs were produced under this
    same code, and run_gossip_comparator's own tagging leaves seed 42
    unsuffixed, so they occupy exactly the seed-42 row of the grid."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    jobs, skipped = [], 0
    for fname in FIELD_NAMES:
        for seed in SEEDS:
            for mode in MODES:
                if (root / fname / tag_for(mode, seed) / "manifest.yaml").exists():
                    skipped += 1
                    continue
                jobs.append((mode, fname, seed))
    total = len(jobs)
    grid = len(FIELD_NAMES) * len(SEEDS) * len(MODES)
    print("GOSSIP MULTIFIELD CROSSED: %d fields x %d seeds x %d arms = %d"
          % (len(FIELD_NAMES), len(SEEDS), len(MODES), grid), flush=True)
    print("  %d already on disk (reused), %d to run -> %s"
          % (skipped, total, root), flush=True)
    t0 = time.time()
    done = failed = 0
    for mode, fname, seed in jobs:
        n = done + failed + 1
        print("\n>>> [%d/%d] %s on %s seed %d  (elapsed %.1f min)"
              % (n, total, mode, fname, seed, (time.time() - t0) / 60.0), flush=True)
        try:
            run_one_on_field(mode, fname, root / fname, seed)
            done += 1
        except Exception:
            failed += 1
            print("!!! FAILED %s / %s / seed %d" % (mode, fname, seed), flush=True)
            traceback.print_exc()
    print("\n=== GOSSIP MULTIFIELD COMPLETE: %d ok, %d failed, %.1f min ==="
          % (done, failed, (time.time() - t0) / 60.0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-inert", action="store_true")
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT)
    a = ap.parse_args()
    if a.verify_inert:
        sys.exit(0 if verify_inert(Path("runs/scratch/gossip_inert")) else 1)
    main(a.root)
