# cert-MSE correlation under the AVERAGED readout.
#
# WHAT THE STORED FIGURE IS. runner.py:334-355 builds step_best by argmax over
# per-node certainty, then writes BOTH cell_cert_steps and cell_mse_steps from
# it. runner.py:380-391 averages each over the last 50 steps per cell and takes
# Pearson r ACROSS CELLS (one point per cell). So the reported
# certainty_mse_pearson_r is an argmax-convention quantity: the certainty of
# the single highest-certainty covering node against that same node's error.
#
# Under the adopted averaged readout both terms must come from the fused
# belief instead: certainty from (nu*, alpha*, beta*), error from gamma*.
# Certainty uses the pipeline's own definition, including its clamp:
#     cert = 1 / (1 + min(beta/(max(nu,1e-6) max(alpha-1,1e-6)), 10))
#
# VALIDATION: the argmax path is recomputed from the stored arrays and checked
# against the manifest's own value before any averaged figure is reported.
#
# Run: python -u experiments/section5_2/run_cert_mse_recompute.py

from __future__ import annotations

import glob, os, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import numpy as np, yaml
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from averaged_readout import averaged_params, truth_for_run, wrap_np

LAST = 50


def cert_of(nu, al, be):
    epi = be / (np.maximum(nu, 1e-6) * np.maximum(al - 1.0, 1e-6))
    return 1.0 / (1.0 + np.minimum(epi, 10.0))


def per_cell_r(cert, e2):
    """Mean over the last LAST steps per cell, then Pearson across cells."""
    c = np.nanmean(cert[-LAST:], axis=0)
    m = np.nanmean(e2[-LAST:], axis=0)
    ok = np.isfinite(c) & np.isfinite(m)
    if ok.sum() < 3:
        return float("nan")
    return float(pearsonr(c[ok], m[ok])[0])


def run(d, seed):
    d = Path(d)
    bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(d / "cell_ncov_steps.npy").astype(int)
    mse = np.load(d / "cell_mse_steps.npy")
    cert = np.load(d / "cell_cert_steps.npy")
    T, G = mse.shape[0], mse.shape[1]
    gam, nu_s, al_s, be_s = averaged_params(bel, nc)
    e2 = wrap_np(gam - truth_for_run(d, seed, T, G)) ** 2
    return per_cell_r(cert, mse), per_cell_r(cert_of(nu_s, al_s, be_s), e2)


G = defaultdict(list)
for p in sorted(glob.glob("runs/chapter5_v2/s531_b2/*_sampled_seed*/manifest.yaml")):
    d = Path(os.path.dirname(p)); m = yaml.safe_load(open(p))
    arm = d.name.rsplit("_seed", 1)[0].replace("_sampled", "")
    ra, rv = run(d, int(m["env_seed"]))
    G[arm].append((m["results"]["certainty_mse_pearson_r"], ra, rv))
for p in sorted(glob.glob("runs/chapter5_v2/s5_3_1_frozen/*/manifest.yaml")):
    d = Path(os.path.dirname(p)); m = yaml.safe_load(open(p))
    ra, rv = run(d, int(m["env_seed"]))
    G["frozen"].append((m["results"]["certainty_mse_pearson_r"], ra, rv))

print("VALIDATION -- argmax r recomputed from stored arrays vs the manifest value")
worst = max(abs(a - b) for v in G.values() for a, b, _ in v)
print("  max |recomputed - manifest| over %d runs: %.3e"
      % (sum(len(v) for v in G.values()), worst))
print("  %s\n" % ("PASSED" if worst < 1e-6 else "FAILED -- recipe does not match"))

print("=" * 92)
print("CERT-MSE PEARSON r  (per-cell means over last 50, correlated across cells)")
print("=" * 92)
print("%-14s %3s %-22s %-22s %10s" % ("arm", "n", "argmax (as reported)",
                                      "averaged (corrected)", "change"))
for arm in ("nig_product", "naive", "certainty", "frozen"):
    v = G.get(arm)
    if not v:
        continue
    a = [x[1] for x in v]; b = [x[2] for x in v]
    ma, sa = st.mean(a), st.stdev(a)
    mb, sb = st.mean(b), st.stdev(b)
    print("%-14s %3d %-22s %-22s %+10.3f"
          % (arm, len(v), "%+.3f +/- %.3f" % (ma, sa),
             "%+.3f +/- %.3f" % (mb, sb), mb - ma))
