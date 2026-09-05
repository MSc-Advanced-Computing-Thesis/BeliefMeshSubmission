# Section 5.6 / 5.7 regeneration results, all arms under current code.
#
# Three tables, each in the shape of the existing one so rows drop in:
#   heterogeneity   -- naive, certainty, product fusion, average fusion
#   multifield      -- the same four, SEED_FIELD pairing
#   uniform control -- all-narrow, all-baseline, all-wide
#
# Each regenerated arm is shown against its stored counterpart where one
# exists, so the code-drift question is answered per arm rather than assumed.
#
# Run: python -u experiments/section5_2/run_s56_regen_results.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from arm_names import name as arm_name
from averaged_readout import load_run, metrics

LAST = 50
B = "runs/chapter5_v2"


def arm(pattern):
    M = []
    for p in sorted(glob.glob(pattern)):
        d = Path(os.path.dirname(p))
        sd = int(yaml.safe_load(open(p))["env_seed"])
        M.append(metrics(load_run(d, sd), last=LAST)["averaged"])
    if not M:
        return None
    ms = lambda k: (st.mean([m[k] for m in M]),
                    st.stdev([m[k] for m in M]) if len(M) > 1 else 0.0)
    return dict(n=len(M), whole=ms("whole"), last50=ms("last50"),
                cov=ms("cov"), ratio=ms("ratio"))


def table(title, rows, note=None):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
    print("%-22s %2s %-21s %-21s %-15s %-15s"
          % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS"))
    for lab, pat in rows:
        r = arm(pat)
        if not r:
            print("%-22s %2s  -- no runs --" % (lab, "-"))
            continue
        f = lambda k, p=5: "%.*f +/- %.*f" % (p, r[k][0], p, r[k][1])
        print("%-22s %2d %-21s %-21s %-15s %-15s"
              % (lab, r["n"], f("whole"), f("last50"), f("cov", 3), f("ratio", 3)))
    if note:
        print("  " + note)


def drift(title, pairs):
    print("\n" + "-" * 108)
    print("REGENERATED vs STORED -- %s" % title)
    print("-" * 108)
    print("%-22s %14s %14s %12s %12s" % ("arm", "whole-run", "last-50",
                                         "coverage", "hw:RMS"))
    for lab, new_pat, old_pat in pairs:
        a, b = arm(new_pat), arm(old_pat)
        if not a or not b:
            print("%-22s  (no stored counterpart)" % lab)
            continue
        d = lambda k: 100 * (a[k][0] - b[k][0]) / b[k][0]
        print("%-22s %13.2f%% %13.2f%% %11.2f%% %11.2f%%"
              % (lab, d("whole"), d("last50"), d("cov"), d("ratio")))


HET = [("Naive", "%s/s5_6_het_rg/naive/*/manifest.yaml" % B),
       ("Certainty", "%s/s5_6_het_rg/certainty/*/manifest.yaml" % B),
       ("Product fusion", "%s/s5_6_het_rg/nig_product/*/manifest.yaml" % B),
       ("Average fusion", "%s/s5_6_het_rg/avgfusion/*/manifest.yaml" % B)]
MF = [("Naive", "%s/s5_6_multifield_rg/naive/het/*/manifest.yaml" % B),
      ("Certainty", "%s/s5_6_multifield_rg/certainty/het/*/manifest.yaml" % B),
      ("Product fusion", "%s/s5_6_multifield_rg/nig_product/het/*/manifest.yaml" % B),
      ("Average fusion", "%s/s5_6_multifield_rg/avgfusion/het/*/manifest.yaml" % B)]
HOMOG = [("all-narrow", "%s/s5_6_homog_rg/narrow/*/manifest.yaml" % B),
         ("all-baseline", "%s/s5_6_homog_rg/baseline/*/manifest.yaml" % B),
         ("all-wide", "%s/s5_6_homog_rg/wide/*/manifest.yaml" % B)]

table("HETEROGENEITY -- mixed mesh, 5 seeds, averaged readout, last-50, "
      "mean +/- sd", HET)
table("MULTIFIELD -- one seed per field (SEED_FIELD), averaged readout, "
      "last-50, mean +/- sd", MF,
      "spread is across FIELD and SEED jointly: SEED_FIELD pairs each field "
      "with its own seed.")
table("UNIFORM CONTROLS -- product fusion, 5 seeds", HOMOG)

drift("heterogeneity",
      [("Naive", HET[0][1], "%s/s5_6_het/het_naive*/manifest.yaml" % B),
       ("Certainty", HET[1][1], "%s/s5_6_het/het_certainty*/manifest.yaml" % B),
       ("Product fusion", HET[2][1], "%s/s5_6_het/het_nig_product*/manifest.yaml" % B)])
drift("multifield",
      [("Naive", MF[0][1], "%s/s5_6_multifield/het/naive_*/manifest.yaml" % B),
       ("Product fusion", MF[2][1],
        "%s/s5_6_multifield/het/nig_product_*/manifest.yaml" % B)])
drift("uniform controls",
      [("all-narrow", HOMOG[0][1], "%s/s5_6_homog_narrow/*/manifest.yaml" % B),
       ("all-baseline", HOMOG[1][1], "%s/s5_6_homog_baseline/*/manifest.yaml" % B),
       ("all-wide", HOMOG[2][1], "%s/s5_6_homog_wide/*/manifest.yaml" % B)])
