# Appendix E experiment: the four aggregation rules on the mixed-capacity mesh
# across the five offset environments, one run per environment (each field is
# paired with its own seed: SEED_FIELD in experiment_multifield_heterogeneous).
#
# Arms: naive, certainty, nig_product (Product fusion) and avgfusion, which is
# nig_product run with fuse_nig_product wrapped so that every fusion uses
# per-contributor weights 1/N (Average fusion, report Section 3.4). Nothing
# else differs between the last two arms.
#
# Output layout (matches the stored artefacts):
#   <root>/<arm>/het/<mode>_seed<seed>_<field>/
#
# Run: python -u experiment_mixed_capacity_five_environments.py --root <dir>
#      python -u experiment_mixed_capacity_five_environments.py --arms naive,nig_product

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR
ART = str(ART_DIR).replace("\\", "/")

import argparse
import time
import traceback
from pathlib import Path

import beliefmesh.fusion.nig_product as nigmod
import beliefmesh.simulation.runner as rn
import experiment_multifield_heterogeneous as MF

_ORIG = nigmod.fuse_nig_product
DEFAULT_ROOT = ART + "/appendix_E_heterogeneous_capacity/mixed_capacity_five_environments"
ARMS = ["naive", "certainty", "nig_product", "avgfusion"]


def averaged_fuse(beliefs, weights=None):
    """w_i = 1/N -> n_eff = 1, nu* = mean(nu_i). gamma* is unchanged."""
    if weights is None and len(beliefs) > 1:
        weights = [1.0 / len(beliefs)] * len(beliefs)
    return _ORIG(beliefs, weights)


def run_arm(arm: str, seed: int, root: Path):
    mode = "nig_product" if arm == "avgfusion" else arm
    patched = arm == "avgfusion"
    assert nigmod.fuse_nig_product is _ORIG, "fusion already patched"
    if patched:
        nigmod.fuse_nig_product = averaged_fuse
        rn.fuse_nig_product = averaged_fuse
    try:
        return MF.run_one(mode, seed, homogeneous=False,
                          root_override=str(root / arm))
    finally:
        nigmod.fuse_nig_product = _ORIG
        rn.fuse_nig_product = _ORIG


def main(arms, seeds, root: Path):
    jobs = [(a, s) for a in arms for s in seeds]
    print("APPENDIX E: %d runs -> %s" % (len(jobs), root), flush=True)
    t0 = time.time()
    ok = bad = 0
    for i, (arm, seed) in enumerate(jobs):
        print("\n>>> [%d/%d] %s seed %d (%s)  elapsed %.1f min"
              % (i + 1, len(jobs), arm, seed, MF.SEED_FIELD[seed],
                 (time.time() - t0) / 60.0), flush=True)
        try:
            run_arm(arm, seed, root)
            ok += 1
        except Exception:
            bad += 1
            print("!!! FAILED %s seed %d" % (arm, seed), flush=True)
            traceback.print_exc()
    print("\n=== APPENDIX E COMPLETE: %d ok, %d failed, %.1f min ==="
          % (ok, bad, (time.time() - t0) / 60.0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT)
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--seeds", type=str,
                    default=",".join(str(s) for s in sorted(MF.SEED_FIELD)))
    a = ap.parse_args()
    main([x for x in a.arms.split(",") if x], [int(x) for x in a.seeds.split(",")],
         Path(a.root))
