# Section 5.2 metrics split by region: the 28 SHARED (peer-supervised) cells
# against the 21 STUDENT-EXCLUSIVE (unsupervised) cells.
#
# Readout only over stored arrays.
#
# WHAT IS AND IS NOT COMPUTABLE. student_cell_mse_steps is per cell, so MSE
# splits exactly. Coverage and hw:RMS are NOT computable by region: the runs
# store only student_cell_epi_steps (epi = beta/(nu(alpha-1))), and the
# Student-t half-width needs scale = sqrt(beta(1+nu)/(nu alpha)) and df = 2
# alpha, which epi alone does not determine. The manifest's
# student_coverage_90 is a single full-FOV scalar (n = 49 x 50).
#
# Run: python -u experiments/section5_2/run_52_region_split.py

from __future__ import annotations
import glob, statistics as st, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from beliefmesh.node.mesh import fov_cells

FOV, G, LAST = 7, 22, 50
ANCHOR, STUDENT = (3, 3), (6, 3)
ARMS = ["frozen", "student", "student_sampled", "direct"]
ROOT = "runs/section5_2/offset/offset"

a = set(fov_cells(*ANCHOR, FOV, G)); s = set(fov_cells(*STUDENT, FOV, G))
REG = {"full FOV (49)": np.array(sorted(s)),
       "shared / supervised (28)": np.array(sorted(a & s)),
       "student-exclusive (21)": np.array(sorted(s - a))}


def per_arm(arm):
    out = {k: {"whole": [], "last50": []} for k in REG}
    for d in sorted(glob.glob("%s/%s_seed*" % (ROOT, arm))):
        m = np.load(d + "/student_cell_mse_steps.npy")
        for k, cells in REG.items():
            v = m[:, cells[:, 0], cells[:, 1]]
            out[k]["whole"].append(float(np.nanmean(v)))
            out[k]["last50"].append(float(np.nanmean(v[-LAST:])))
    return out


D = {arm: per_arm(arm) for arm in ARMS}
ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

print("=" * 96)
print("SECTION 5.2 -- MSE BY REGION  (5 seeds, mean +/- sd)")
print("=" * 96)
for k in REG:
    print("\n--- %s ---" % k)
    print("%-20s %-24s %-24s" % ("arm", "whole-run MSE", "last-50 MSE"))
    for arm in ARMS:
        w, l = ms(D[arm][k]["whole"]), ms(D[arm][k]["last50"])
        print("%-20s %-24s %-24s" % (arm, "%.5f +/- %.5f" % w, "%.5f +/- %.5f" % l))

print("\n" + "=" * 96)
print("GAP CLOSED vs DIRECT SUPERVISION,  (frozen - arm) / (frozen - direct)")
print("100% = matches direct;  0% = no better than frozen")
print("=" * 96)
for win in ("whole", "last50"):
    print("\n--- %s ---" % ("whole-run" if win == "whole" else "last-50"))
    print("%-20s %-22s %-22s %-22s" % ("arm", "full FOV", "shared (supervised)",
                                       "exclusive (unsupervised)"))
    for arm in ("student", "student_sampled"):
        row = "%-20s" % arm
        for k in REG:
            f = st.mean(D["frozen"][k][win]); dd = st.mean(D["direct"][k][win])
            x = st.mean(D[arm][k][win])
            gc = 100 * (f - x) / (f - dd) if abs(f - dd) > 1e-12 else float("nan")
            row += " %-21s" % ("%+.1f%%" % gc)
        print(row)
print("\n  difficulty check -- direct arm (ground truth everywhere) by region:")
for k in REG:
    print("    %-26s whole-run %.5f   last-50 %.5f"
          % (k, st.mean(D["direct"][k]["whole"]), st.mean(D["direct"][k]["last50"])))
print("\n  frozen (no training at all) by region:")
for k in REG:
    print("    %-26s whole-run %.5f   last-50 %.5f"
          % (k, st.mean(D["frozen"][k]["whole"]), st.mean(D["frozen"][k]["last50"])))
