# Section 5.5 parameter exchange across five offset fields -- results.
#
# THE SPREAD REPORTED HERE IS ACROSS ENVIRONMENTS, NOT ACROSS SEEDS. Every run
# uses seed 42; only the offset field changes. The five-seed table on the
# single field varies rotations and wearable paths within ONE environment.
# The two sd's answer different questions and must not be compared or pooled.
#
# Averaged NIG readout throughout (Chapter 5's convention).
#
# Run: python -u experiments/section5_2/run_gossip_multifield_results.py

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from arm_names import name as arm_name
from averaged_readout import load_run, metrics

LAST = 50
SEED = 42
ROOT = Path("runs/chapter5_v2/s5_5_gossip_multifield")
FIELDS = ["field0_original", "field1_seed11", "field2_seed23",
          "field3_seed37", "field4_seed51_dual"]
SHORT = {f: f.split("_")[0] for f in FIELDS}
MODES = ["fusion", "gossip_uniform", "gossip_weighted", "fedavg_global"]


def one(mode, fname):
    d = ROOT / fname / ("gossip_cmp_%s" % mode)
    if not (d / "manifest.yaml").exists():
        return None
    m = metrics(load_run(d, SEED), last=LAST)["averaged"]
    man = yaml.safe_load(open(d / "manifest.yaml"))
    r = man.get("results", {})
    m["comm_mb"] = r.get("comm_bytes_total", float("nan")) / 1e6
    m["comm_per_step"] = r.get("comm_bytes_mean_per_step", float("nan"))
    return m


D = {(mo, f): one(mo, f) for mo in MODES for f in FIELDS}
have = {k: v for k, v in D.items() if v}
print("loaded %d of %d runs" % (len(have), len(MODES) * len(FIELDS)))

ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

print("\n" + "=" * 112)
print("SECTION 5.5 PARAMETER EXCHANGE ACROSS FIVE OFFSET FIELDS")
print("seed %d held FIXED; the offset field is the only thing that varies" % SEED)
print("=" * 112)
print("*** THE +/- FIGURES ARE SPREAD ACROSS ENVIRONMENTS, NOT ACROSS SEEDS. ***")
print("*** The five-seed single-field table varies rotations and wearable    ***")
print("*** paths within ONE environment. The two are not comparable and must ***")
print("*** not be pooled or quoted interchangeably.                          ***")
print("=" * 112)
print("%-16s %2s %-20s %-20s %-15s %-15s %-14s"
      % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage",
         "hw : RMS", "comm (MB)"))
for mo in MODES:
    v = [D[(mo, f)] for f in FIELDS if D[(mo, f)]]
    if not v:
        continue
    f_ = lambda k, p=5: "%.*f +/- %.*f" % (p, ms([x[k] for x in v])[0],
                                           p, ms([x[k] for x in v])[1])
    print("%-16s %2d %-20s %-20s %-15s %-15s %-14s"
          % (arm_name(mo), len(v), f_("whole"), f_("last50"), f_("cov", 3),
             f_("ratio", 3), f_("comm_mb", 1)))

print("\n" + "=" * 112)
print("PER-FIELD LAST-50 MSE -- does belief/gossip parity hold in EVERY field?")
print("=" * 112)
print("%-16s" % "arm" + "".join("%14s" % SHORT[f] for f in FIELDS))
for mo in MODES:
    row = "%-16s" % arm_name(mo)
    for f in FIELDS:
        x = D[(mo, f)]
        row += "%14s" % ("%.5f" % x["last50"] if x else "--")
    print(row)

print("\nPARITY TEST, per field: |fusion - gossip| relative to fusion's last-50.")
print("Parity is claimed in the single-field table; it must hold field by field")
print("or the claim is an artefact of averaging.")
print("%-14s %14s %14s %14s %12s"
      % ("field", "fusion", "gossip_uni", "gossip_wtd", "worst gap"))
worst_overall, breaches = 0.0, []
for f in FIELDS:
    a = D[("fusion", f)]
    u, w = D[("gossip_uniform", f)], D[("gossip_weighted", f)]
    if not (a and u and w):
        print("%-14s %14s" % (SHORT[f], "incomplete"))
        continue
    gaps = [abs(x["last50"] - a["last50"]) / a["last50"] * 100 for x in (u, w)]
    worst = max(gaps)
    worst_overall = max(worst_overall, worst)
    if worst > 10.0:
        breaches.append((SHORT[f], worst))
    print("%-14s %14.5f %14.5f %14.5f %11.1f%%"
          % (SHORT[f], a["last50"], u["last50"], w["last50"], worst))
print("\n  worst gap across all five fields: %.1f%%" % worst_overall)
print("  fields where a gossip arm differs from belief exchange by >10%%: %s"
      % ([b[0] for b in breaches] if breaches else "none"))
print("  -> parity holds in every individual field: %s"
      % ("YES" if not breaches else "NO -- see above"))
