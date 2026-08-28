# Section 5.3.1 tables: aggregation parity, and compute cost.
# Stored artefacts only. Plain text to stdout, CSV to runs/chapter5_v2/.
#
# Run: python -u experiments/section5_2/make_531_tables.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_estimators import LAST, load

OUT = Path("runs/chapter5_v2")
ARMS = ["nig_product", "naive", "certainty"]
# frozen never trains, so its cell-space MSE is target-rule independent and the
# stored arm is reusable. It was NOT regenerated, so it has no cell_nig_steps
# and its interval quantities are not computable -- see the note in the output.
FROZEN_GLOB = "runs/stage6/offset_world/aggregation_comparator/agg_cmp_frozen*/cell_mse_steps.npy"


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def fmt(t, p=5):
    return "%.*f +/- %.*f" % (p, t[0], p, t[1])


def parity_table():
    G = defaultdict(list)
    for p in sorted(glob.glob("runs/chapter5_v2/s531_b2/*_sampled_seed*/manifest.yaml")):
        d = Path(os.path.dirname(p))
        arm = d.name.rsplit("_seed", 1)[0].replace("_sampled", "")
        R = load(d)
        mse, nig = R["mse"], R["nig"]
        T = mse.shape[0]
        sl = slice(T - LAST, T)
        e = np.sqrt(mse[sl])
        nu, al, be = (nig[sl, ..., i] for i in range(3))
        k = np.isfinite(e) & np.isfinite(nu) & (al > 1.0)
        sc = np.sqrt(be[k] * (1 + nu[k]) / (nu[k] * al[k]))
        hw = sps.t.ppf(0.95, df=2 * al[k]) * sc
        m = float(np.nanmean(mse[sl]))
        G[arm].append(dict(whole=float(np.nanmean(mse)), last50=m,
                           cov=float((e[k] <= hw).mean()),
                           ratio=float(np.mean(hw) * 180 / (np.sqrt(m) * 180))))
    fw, fl = [], []
    for f in sorted(glob.glob(FROZEN_GLOB)):
        a = np.load(f)
        fw.append(float(np.nanmean(a)))
        fl.append(float(np.nanmean(a[-LAST:])))

    ref = ms([r["whole"] for r in G["nig_product"]])[0]
    rows = []
    for arm in ARMS:
        v = G[arm]
        w = ms([r["whole"] for r in v])
        rows.append((arm, len(v), w, ms([r["last50"] for r in v]),
                     ms([r["cov"] for r in v]), ms([r["ratio"] for r in v]),
                     100 * (w[0] - ref) / ref))
    if fw:
        rows.append(("frozen", len(fw), ms(fw), ms(fl), None, None,
                     100 * (ms(fw)[0] - ref) / ref))

    print("=" * 96)
    print("TABLE 5.3.1 -- AGGREGATION PARITY   (cell space, last-50 window, 5 seeds, mean +/- sd)")
    print("=" * 96)
    print("%-13s %3s %-21s %-21s %-15s %-15s %10s"
          % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage",
             "hw : RMS", "vs nig_prod"))
    for arm, n, w, l, c, r, pct in rows:
        print("%-13s %3d %-21s %-21s %-15s %-15s %+9.2f%%"
              % (arm, n, fmt(w), fmt(l),
                 fmt(c, 3) if c else "not computable ",
                 fmt(r, 3) if r else "not computable ", pct))
    print()
    print("frozen: interval quantities are NOT computable. Frozen never trains, so its")
    print("  cell-space MSE is independent of the target rule and the stored arm is")
    print("  reused -- but that arm predates cell_nig_steps, so it has no (nu, alpha,")
    print("  beta) and the Student-t interval cannot be reconstructed. Regenerating it")
    print("  would take 5 runs; no new runs were requested.")

    with open(OUT / "table_5_3_1_parity.csv", "w", encoding="utf8") as f:
        f.write("arm,n_seeds,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,"
                "last50_mse_sd,coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd,"
                "whole_run_pct_diff_vs_nig_product\n")
        for arm, n, w, l, c, r, pct in rows:
            cc = "%.6f,%.6f" % c if c else ","
            rr = "%.6f,%.6f" % r if r else ","
            f.write("%s,%d,%.6f,%.6f,%.6f,%.6f,%s,%s,%.4f\n"
                    % (arm, n, w[0], w[1], l[0], l[1], cc, rr, pct))
    print("  wrote %s" % (OUT / "table_5_3_1_parity.csv"))


def cost_table():
    G = defaultdict(list)
    for p in sorted(glob.glob("runs/chapter5_v2/s531_b2/*/manifest.yaml")):
        m = yaml.safe_load(open(p))
        r = m["results"]
        arm = os.path.basename(os.path.dirname(p)).rsplit("_seed", 1)[0]
        fus, fwd, bwd = (r["fusion_time_total_sec"], r["forward_time_total_sec"],
                         r["backward_time_total_sec"])
        tot = fus + fwd + bwd
        G[arm].append((fus, fwd, bwd, r["step_wall_time_total_sec"],
                       100 * fus / tot if tot > 0 else 0.0))
    print()
    print("=" * 96)
    print("TABLE 5.3.1b -- COMPUTE COST   (seconds per 390-step run, 5-seed means)")
    print("=" * 96)
    print("%-26s %3s %10s %10s %10s %11s %12s"
          % ("arm", "n", "fusion s", "forward s", "backward s", "wall s",
             "fusion % of\n"[:12]))
    print("%-26s %3s %10s %10s %10s %11s %12s"
          % ("", "", "", "", "", "", "fwd+bwd+fus"))
    for arm in sorted(G):
        v = G[arm]
        f = lambda i: st.mean([x[i] for x in v])
        print("%-26s %3d %10.2f %10.2f %10.2f %11.1f %11.2f%%"
              % (arm, len(v), f(0), f(1), f(2), f(3), f(4)))
    print()
    print("NOTE -- instrumentation artefact, not a measurement:")
    print("  naive and certainty register 0.00 s of fusion time because their")
    print("  aggregation is a plain (or certainty-weighted) MEAN of contributor")
    print("  gammas, which is not instrumented as fusion. They do perform")
    print("  aggregation; it is simply not timed under that label. This must NOT be")
    print("  read as fusion being infinitely more expensive than the alternatives --")
    print("  the meaningful comparison is fusion's ~2% share of measured compute")
    print("  against the forward and backward passes it informs.")

    with open(OUT / "table_5_3_1_compute.csv", "w", encoding="utf8") as f:
        f.write("arm,n_seeds,fusion_s,forward_s,backward_s,step_wall_s,fusion_pct_of_measured\n")
        for arm in sorted(G):
            v = G[arm]
            g = lambda i: st.mean([x[i] for x in v])
            f.write("%s,%d,%.4f,%.4f,%.4f,%.2f,%.4f\n"
                    % (arm, len(v), g(0), g(1), g(2), g(3), g(4)))
    print("  wrote %s" % (OUT / "table_5_3_1_compute.csv"))


if __name__ == "__main__":
    parity_table()
    cost_table()
