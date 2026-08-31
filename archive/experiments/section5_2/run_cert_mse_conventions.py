# cert-MSE correlation: BOTH aggregation forms x BOTH readout conventions.
#
# Aggregation forms (cert_mse_metrics.py):
#   averaged_then_correlated  -- mean per cell over the window, then correlate
#       across cells. This is what runner.py writes into the manifest as
#       certainty_mse_pearson_r. Attenuates through range restriction.
#   per_timestep_then_averaged -- correlate across cells WITHIN each timestep,
#       then average those r values. Implemented and called by 17 scripts, but
#       only at PRINT time: never stored in any manifest.
#
# Readout conventions:
#   argmax   -- stored cell_cert_steps / cell_mse_steps (the displaced rule)
#   averaged -- certainty and error recomputed from the fused (nu*, alpha*,
#               beta*, gamma*), the adopted rule
#
# Run: python -u experiments/section5_2/run_cert_mse_conventions.py

from __future__ import annotations
import glob, os, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from averaged_readout import averaged_params, truth_for_run, wrap_np
from stage6_spatial_mesh.cert_mse_metrics import (averaged_then_correlated,
                                                  per_timestep_then_averaged)
LAST = 50


def cert_of(nu, al, be):
    epi = be / (np.maximum(nu, 1e-6) * np.maximum(al - 1.0, 1e-6))
    return 1.0 / (1.0 + np.minimum(epi, 10.0))


def both(d, seed):
    d = Path(d)
    bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(d / "cell_ncov_steps.npy").astype(int)
    mse = np.load(d / "cell_mse_steps.npy"); cert = np.load(d / "cell_cert_steps.npy")
    T = mse.shape[0]; lo, hi = T - LAST, T
    gam, nu_s, al_s, be_s = averaged_params(bel, nc)
    e2 = wrap_np(gam - truth_for_run(d, seed, T, mse.shape[1])) ** 2
    ca = cert_of(nu_s, al_s, be_s)
    return dict(
        arg_avg=averaged_then_correlated(mse, cert, lo, hi),
        arg_pts=per_timestep_then_averaged(mse, cert, lo, hi)[0],
        avg_avg=averaged_then_correlated(e2, ca, lo, hi),
        avg_pts=per_timestep_then_averaged(e2, ca, lo, hi)[0],
        pts_sd=per_timestep_then_averaged(e2, ca, lo, hi)[1])


G = defaultdict(list); MAN = defaultdict(list)
for pat, name in ((("runs/chapter5_v2/s531_b2/*_sampled_seed*/manifest.yaml"), None),
                  (("runs/chapter5_v2/s5_3_1_frozen/*/manifest.yaml"), "frozen")):
    for p in sorted(glob.glob(pat)):
        d = Path(os.path.dirname(p)); m = yaml.safe_load(open(p))
        arm = name or d.name.rsplit("_seed", 1)[0].replace("_sampled", "")
        G[arm].append(both(d, int(m["env_seed"])))
        MAN[arm].append(m["results"]["certainty_mse_pearson_r"])

ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
worst = max(abs(ms([x["arg_avg"] for x in G[a]])[0] - ms(MAN[a])[0]) for a in G)
print("VALIDATION -- argmax averaged-then-correlated vs the manifest field")
print("  max |recomputed - manifest| across arms: %.3e  %s\n"
      % (worst, "PASSED" if worst < 1e-9 else "FAILED"))

print("=" * 100)
print("CERT-MSE PEARSON r -- 5.3.1 ARMS, 5 SEEDS (mean +/- sd across seeds)")
print("=" * 100)
print("%-14s %-24s %-24s | %-24s %-24s"
      % ("", "ARGMAX readout", "", "AVERAGED readout", ""))
print("%-14s %-24s %-24s | %-24s %-24s"
      % ("arm", "averaged-then-corr", "per-timestep", "averaged-then-corr", "per-timestep"))
for arm in ("nig_product", "naive", "certainty", "frozen"):
    if arm not in G: continue
    f = lambda k: "%+.3f +/- %.3f" % ms([x[k] for x in G[arm]])
    print("%-14s %-24s %-24s | %-24s %-24s"
          % (arm, f("arg_avg"), f("arg_pts"), f("avg_avg"), f("avg_pts")))
print()
print("  (the manifest field is the ARGMAX / averaged-then-correlated column)")
