# Section 5.9 ablation family -- full results under the AVERAGED NIG readout
# (Chapter 5's convention). Readout only over stored beliefs.
#
# Run: python -u experiments/section5_2/run_ablation_results.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from averaged_readout import load_run, metrics

LAST = 50
BASE = "runs/chapter5_v2"


def arm(pattern):
    """(n, whole, last50, cov, ratio) as (mean, sd) tuples, averaged readout."""
    M = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        d = Path(os.path.dirname(p))
        sd = int(yaml.safe_load(open(p))["env_seed"])
        M.append(metrics(load_run(d, sd), last=LAST)["averaged"])
    if not M:
        return None
    ms = lambda k: (st.mean([m[k] for m in M]),
                    st.stdev([m[k] for m in M]) if len(M) > 1 else 0.0)
    return len(M), ms("whole"), ms("last50"), ms("cov"), ms("ratio")


def table(title, rows, ref=None, note=None):
    print("\n" + "=" * 102)
    print(title)
    print("=" * 102)
    print("%-26s %3s %-21s %-21s %-15s %-15s %9s"
          % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage",
             "hw : RMS", "vs ref"))
    refv = None
    for lab, r in rows:
        if r is None:
            print("%-26s %3s %s" % (lab, "-", "NOT RUN"))
            continue
        n, w, l, c, q = r
        if lab == ref:
            refv = w[0]
        pct = (100 * (w[0] - refv) / refv) if refv else 0.0
        f = lambda t, p=5: "%.*f +/- %.*f" % (p, t[0], p, t[1])
        print("%-26s %3d %-21s %-21s %-15s %-15s %+8.2f%%"
              % (lab, n, f(w), f(l), f(c, 3), f(q, 3), pct))
    if note:
        print("  " + note)


def main():
    print("SECTION 5.9 -- ABLATION RESULTS")
    print("cell space, AVERAGED NIG readout, 5 seeds, mean +/- sd")
    print("Reference arm for each family is marked; 'vs ref' is whole-run MSE.")

    # ---- 1. lambda sweep -------------------------------------------------
    rows = [("lam = %s%s" % (v, "  (default)" if v == "5.0" else ""),
             arm("%s/s5_9_lam_%s/*/manifest.yaml" % (BASE, v.replace(".", "p"))))
            for v in ("1.0", "2.5", "5.0", "10.0", "25.0")]
    table("1. LAMBDA SWEEP  (lr and rho fixed; lam is the only axis)", rows,
          ref="lam = 5.0  (default)")

    # ---- 2. consensus rho ------------------------------------------------
    rhos = ["0p01", "0p05", "0p1", "0p2", "0p5", "0p8", "0p99"]
    rows = [("rho = %s%s" % (r.replace("p", "."), "  (default)" if r == "0p2" else ""),
             arm("%s/s5_9_consensus_rho/consensus_rho%s_seed*/manifest.yaml" % (BASE, r)))
            for r in rhos]
    table("2. CONSENSUS WEIGHTING  (rho sweep, mode=consensus)", rows,
          ref="rho = 0.2  (default)")

    # ---- 3. nig_product_weighted ----------------------------------------
    rows = [("nig_product  (default)",
             arm("%s/s531_b2/nig_product_sampled_seed*/manifest.yaml" % BASE)),
            ("nig_product_weighted",
             arm("%s/s5_9_nig_weighted/*/manifest.yaml" % BASE))]
    table("3. WEIGHTED FUSION  (consensus-trust x certainty tempering)", rows,
          ref="nig_product  (default)")

    # ---- 4. uncertainty measure -----------------------------------------
    rows = [("%s%s" % (m, "  (default)" if m == "epistemic" else ""),
             arm("%s/s5_9_uncertainty_%s/*/manifest.yaml" % (BASE, m)))
            for m in ("epistemic", "aleatoric", "total")]
    rows.append(("colour world x3", None))
    table("4. UNCERTAINTY MEASURE  (which quantity drives certainty)", rows,
          ref="epistemic  (default)",
          note="colour-world variant NOT RUN -- batch 1 was killed before it started.")

    # ---- 5. gradient tempering ------------------------------------------
    rows = [("het, tempering ON", arm("%s/s5_9_temper_het_temper/**/manifest.yaml" % BASE)),
            ("het, tempering OFF", arm("%s/s5_9_temper_het_notemper/**/manifest.yaml" % BASE)),
            ("homog, tempering ON", arm("%s/s5_9_temper_homog_temper/**/manifest.yaml" % BASE))]
    table("5. GRADIENT TEMPERING  (multifield: each seed a different field)", rows,
          ref="het, tempering ON",
          note="these use the multifield offset variants, so absolute MSE is not "
               "comparable to the other families.")

    # ---- 6. matched routing ---------------------------------------------
    rows = [("random policy", arm("%s/s5_9_routing_matched/random_seed*/manifest.yaml" % BASE)),
            ("uncertainty_guided", arm("%s/s5_9_routing_matched/uncertainty_guided_seed*/manifest.yaml" % BASE))]
    table("6. MATCHED ROUTING  (policy is the only difference; verified)", rows,
          ref="random policy")

    # ---- 5.6 uniform vs mixed -------------------------------------------
    rows = [("heterogeneous (mixed)", arm("%s/s5_6_het/het_nig_product*/manifest.yaml" % BASE)),
            ("uniform narrow", arm("%s/s5_6_homog_narrow/**/manifest.yaml" % BASE)),
            ("uniform baseline", arm("%s/s5_6_homog_baseline/**/manifest.yaml" % BASE)),
            ("uniform wide", arm("%s/s5_6_homog_wide/**/manifest.yaml" % BASE))]
    table("7. SECTION 5.6 -- MIXED vs UNIFORM CAPACITY (all matched)", rows,
          ref="heterogeneous (mixed)")


if __name__ == "__main__":
    main()
