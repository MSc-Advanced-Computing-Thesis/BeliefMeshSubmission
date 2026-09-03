# Empirical 90% coverage crossing across every configuration with stored NIG
# parameters. Same within-run scaling method used on the 5.3.1 arms:
# scale a run's own half-widths by k until coverage hits exactly 0.90, then
# report the hw:RMS it would have at that k.
#
# Readout only. No runs, no training path touched.
#
# Run: python -u experiments/section5_2/run_crossing_survey.py

from __future__ import annotations
import glob, os, statistics as st, sys
from pathlib import Path
import numpy as np, yaml

sys.path.insert(0, "experiments/section5_2")
from averaged_readout import load_run
LAST = 50


def crossing(d, seed):
    R = load_run(Path(d), seed); T = R["T"]; sl = slice(T - LAST, T)
    e2, hw = R["averaged"]
    ok = np.isfinite(e2[sl]) & np.isfinite(hw[sl])
    if ok.sum() < 100:
        return None
    e = np.sqrt(e2[sl][ok]); h = hw[sl][ok]
    rms = float(np.sqrt(np.mean(e2[sl][ok])))
    k = float(np.quantile(e / h, 0.90))
    return dict(cov=float((e <= h).mean()), ratio=float(np.mean(h) / rms),
                cross=k * float(np.mean(h)) / rms,
                p90=float(np.quantile(e, 0.90) / rms))


def group(pat, key=None):
    """key(manifest) -> label; None means one group for the whole pattern."""
    G = {}
    for p in sorted(glob.glob(pat, recursive=True)):
        m = yaml.safe_load(open(p)); d = Path(os.path.dirname(p))
        lab = key(m, d) if key else ""
        v = crossing(d, int(m["env_seed"]))
        if v:
            G.setdefault(lab, []).append(v)
    return G


ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
BASE = "runs/chapter5_v2"
BLOCKS = [
    ("5.3.1 aggregation", "%s/s531_b2/*_sampled_seed*/manifest.yaml" % BASE,
     lambda m, d: d.name.rsplit("_seed", 1)[0].replace("_sampled", "")),
    ("5.3.1 frozen", "%s/s5_3_1_frozen/*/manifest.yaml" % BASE, lambda m, d: "frozen"),
    ("5.4.1 density sweep", "%s/s5_4_1_overlap/*/manifest.yaml" % BASE,
     lambda m, d: "coverage %.2f (%d nodes)" % (m["mean_cell_coverage"], m["n_nodes"])),
    ("5.4.2 field", "%s/s5_4_2_*/*/manifest.yaml" % BASE,
     lambda m, d: "static" if "s5_4_2_static" in str(d).replace("\\", "/") else "dynamic"),
    ("5.6 capacity", "%s/s5_6_het/het_nig_product*/manifest.yaml" % BASE,
     lambda m, d: "heterogeneous"),
    ("5.6 capacity", "%s/s5_6_homog_narrow/**/manifest.yaml" % BASE, lambda m, d: "uniform narrow"),
    ("5.6 capacity", "%s/s5_6_homog_baseline/**/manifest.yaml" % BASE, lambda m, d: "uniform baseline"),
    ("5.6 capacity", "%s/s5_6_homog_wide/**/manifest.yaml" % BASE, lambda m, d: "uniform wide"),
    ("5.9 routing", "%s/s5_9_routing_matched/*/manifest.yaml" % BASE,
     lambda m, d: "random" if d.name.startswith("random") else "uncertainty_guided"),
]

print("=" * 104)
print("EMPIRICAL 90%% COVERAGE CROSSING -- within-run half-width scaling, last-50")
print("=" * 104)
print("%-22s %-26s %3s %-18s %-18s %-16s"
      % ("section", "arm", "n", "CROSSING hw:RMS", "p90(e)/RMS", "observed hw:RMS"))
allx = []
seen = set()
for sec, pat, key in BLOCKS:
    G = group(pat, key)
    for lab in sorted(G):
        V = G[lab]
        x = ms([v["cross"] for v in V]); q = ms([v["p90"] for v in V])
        r = ms([v["ratio"] for v in V])
        head = sec if (sec, ) not in seen else ""
        seen.add((sec, ))
        print("%-22s %-26s %3d %-18s %-18s %-16s"
              % (head, lab, len(V), "%.3f +/- %.3f" % x, "%.3f +/- %.3f" % q,
                 "%.3f +/- %.3f" % r))
        allx.append((sec, lab, x[0], q[0]))

xs = [a[2] for a in allx]
print("\n" + "=" * 104)
print("DOES THE 1.65 - 1.87 BAND HOLD?")
print("=" * 104)
print("  crossing across ALL %d configurations: min %.3f  max %.3f  spread %.3f"
      % (len(xs), min(xs), max(xs), max(xs) - min(xs)))
inb = [a for a in allx if 1.65 <= a[2] <= 1.87]
out = [a for a in allx if not (1.65 <= a[2] <= 1.87)]
print("  inside the band : %d/%d" % (len(inb), len(allx)))
print("  outside         : %d/%d" % (len(out), len(allx)))
for sec, lab, x, q in sorted(out, key=lambda t: t[2]):
    print("      %-22s %-26s %.3f  (p90/RMS %.3f)" % (sec, lab, x, q))
qs = [a[3] for a in allx]
print("\n  shape statistic p90(e)/RMS: min %.3f  max %.3f  (Gaussian = 1.645)"
      % (min(qs), max(qs)))
r = np.corrcoef(qs, xs)[0, 1]
print("  correlation between shape and crossing: r = %.3f" % r)
lo, hi = min(xs), max(xs)
print("\n  a band covering every configuration would need to span %.2f - %.2f"
      % (np.floor(lo * 100) / 100, np.ceil(hi * 100) / 100))
