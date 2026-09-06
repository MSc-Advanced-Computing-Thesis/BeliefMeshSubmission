# Section 5.4 parameter exchange, CROSSED multifield: 5 fields x 5 seeds x 4 arms.
#
# Filter-on throughout, matching the chapter standard. run_gossip_comparator
# takes runner.py's default of True, so no override was needed.
#
# Three reports:
#   1. four variants pooled over all 25 runs each
#   2. the same PER ENVIRONMENT (5 seeds per cell) -- the breakdown that
#      decides whether last-50 parity holds field by field or only in the mean
#   3. communication volume, totals and bytes per node per timestep
#
# Run: python -u experiments/section5_2/run_gossip_multifield_crossed_results.py

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

LAST, N_NODES, T_STEPS = 50, 36, 390
ROOT = Path("runs/chapter5_v2/s5_5_gossip_multifield")
FIELDS = ["field0_original", "field1_seed11", "field2_seed23",
          "field3_seed37", "field4_seed51_dual"]
SHORT = {f: f.split("_")[0] for f in FIELDS}
MODES = ["fusion", "gossip_uniform", "gossip_weighted", "fedavg_global"]


def runs_for(field, mode):
    out = []
    for p in sorted(glob.glob(str(ROOT / field / ("gossip_cmp_%s*" % mode)
                                   / "manifest.yaml"))):
        nm = os.path.basename(os.path.dirname(p))
        if nm.replace("gossip_cmp_", "").rsplit("_seed", 1)[0] != mode:
            continue           # 'fusion' would otherwise catch nothing else,
        d = Path(os.path.dirname(p))          # but be explicit anyway
        man = yaml.safe_load(open(p))
        m = metrics(load_run(d, int(man["env_seed"])), last=LAST)["averaged"]
        r = man.get("results", {}) or {}
        m["comm_total"] = r.get("comm_bytes_total", float("nan"))
        m["comm_step"] = r.get("comm_bytes_mean_per_step", float("nan"))
        out.append(m)
    return out


ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
CELL = {(f, m): runs_for(f, m) for f in FIELDS for m in MODES}
POOL = {m: [r for f in FIELDS for r in CELL[(f, m)]] for m in MODES}
print("loaded %d runs (%d cells x 5 seeds)"
      % (sum(len(v) for v in POOL.values()), len(FIELDS) * len(MODES)))

print("\n" + "=" * 108)
print("TABLE 5.4 (CROSSED) -- PARAMETER EXCHANGE ACROSS 5 ENVIRONMENTS x 5 SEEDS")
print("filter-on, averaged readout, last-%d window; n = 25 runs per variant" % LAST)
print("=" * 108)
print("%-18s %3s %-21s %-21s %-15s %-15s"
      % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS"))
for m in MODES:
    v = POOL[m]
    f = lambda k, p=5: "%.*f +/- %.*f" % (p, ms([x[k] for x in v])[0],
                                          p, ms([x[k] for x in v])[1])
    print("%-18s %3d %-21s %-21s %-15s %-15s"
          % (arm_name(m), len(v), f("whole"), f("last50"), f("cov", 3), f("ratio", 3)))
print("\n  sd pools ACROSS environments AND seeds, so it is larger than a")
print("  single-environment sd and is not comparable with the 5-seed tables.")

print("\n" + "=" * 108)
print("PER ENVIRONMENT (5 seeds per cell)")
print("=" * 108)
for f in FIELDS:
    print("\n--- %s" % f)
    print("  %-18s %-21s %-21s %-15s %-15s"
          % ("arm", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS"))
    for m in MODES:
        v = CELL[(f, m)]
        if not v:
            continue
        g = lambda k, p=5: "%.*f +/- %.*f" % (p, ms([x[k] for x in v])[0],
                                              p, ms([x[k] for x in v])[1])
        print("  %-18s %-21s %-21s %-15s %-15s"
              % (arm_name(m), g("whole"), g("last50"), g("cov", 3), g("ratio", 3)))

print("\n" + "=" * 108)
print("LAST-50 PARITY, FIELD BY FIELD  (per cent vs Product fusion; negative = better)")
print("=" * 108)
print("%-20s %14s %12s %12s %12s" % ("field", "fusion last-50", "Gossip unif",
                                     "Gossip wgt", "FedAvg"))
breaches = []
for f in FIELDS:
    b = ms([x["last50"] for x in CELL[(f, "fusion")]])[0]
    row = [100 * (ms([x["last50"] for x in CELL[(f, m)]])[0] - b) / b
           for m in MODES[1:]]
    print("%-20s %14.5f %11.1f%% %11.1f%% %11.1f%%" % (f, b, *row))
    if max(abs(x) for x in row[:2]) > 10.0:
        breaches.append(SHORT[f])
print("\n  fields where a gossip arm differs from belief exchange by >10%%: %s"
      % (breaches if breaches else "none"))
print("  -> last-50 parity holds in every individual field: %s"
      % ("YES" if not breaches else "NO"))

print("\n" + "=" * 108)
print("COMMUNICATION VOLUME")
print("=" * 108)
print("%-18s %18s %16s %22s"
      % ("arm", "total (MB)", "bytes/step", "bytes/node/step"))
base = None
for m in MODES:
    v = POOL[m]
    tot = ms([x["comm_total"] for x in v])
    per = ms([x["comm_step"] for x in v])
    if base is None:
        base = tot[0]
    print("%-18s %18s %16s %22s"
          % (arm_name(m), "{:,.1f}".format(tot[0] / 1e6),
             "{:,.0f}".format(per[0]), "{:,.0f}".format(per[0] / N_NODES)))
print("\n  bytes/node/step = mean bytes per step / %d nodes" % N_NODES)
for m in MODES[1:]:
    print("  %-18s %.0fx belief exchange"
          % (arm_name(m), ms([x["comm_total"] for x in POOL[m]])[0] / base))
