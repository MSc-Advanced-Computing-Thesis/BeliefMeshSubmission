# Reliability-diagram / binned-calibration check, built entirely from data
# already saved by run_mesh_experiment (cell_cert_steps.npy, cell_mse_steps.npy
# -- per-step, per-cell arrays covering the whole grid every run already
# writes). No new experiments needed for this.
#
# r (Pearson correlation) only tests whether certainty and error are
# *monotonically related* -- it says nothing about whether the certainty
# NUMBER means anything, and it's compressed by range-restriction whenever
# certainty is bunched near 1.0. Binning certainty into quantiles and looking
# at empirical RMSE per bin gives a more robust, distribution-free view of
# the same relationship, and the low-bin-vs-high-bin RMSE ratio is a more
# interpretable effect size than r for judging "is this spread actually
# informative" independent of its exact linear shape.
#
# NOTE: not a true calibration check in the strict probabilistic sense (that
# would need per-prediction PIT values from the raw NIG (gamma,nu,alpha,beta)
# tuples, which existing runs don't save) -- this is a *monotonicity/
# informativeness* reliability curve, which is what's actually decidable
# from data already on disk.
#
# Also separates runs by whether they predate or postdate the consensus-
# tempered-gradient edit to mesh.py (2026-07-29 15:25:21) -- every lam>=1.0
# run was launched after that edit and has tempering silently baked in.
#
# Run: python -u experiments/stage6_spatial_mesh/analyze_calibration.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("runs/stage6/offset_world/dynamic_lam_rho_sweep")
N_BINS = 8
TEMPERING_CUTOFF = "2026-07-29T15:25:21"  # mesh.py edit adding trust-tempered gradient

RUNS = [
    # (tag, label, lam, rho)
    ("consensus_rho0.0001", "rho=0.0001, lam=0.1 (default)", 0.1, 0.0001),
    ("consensus_rho0.01", "rho=0.01, lam=0.1", 0.1, 0.01),
    ("consensus_rho0.05", "rho=0.05, lam=0.1", 0.1, 0.05),
    ("consensus_rho0.1", "rho=0.1, lam=0.1", 0.1, 0.1),
    ("consensus_rho0.2", "rho=0.2 (default), lam=0.1", 0.1, 0.2),
    ("consensus_rho0.4", "rho=0.4, lam=0.1", 0.1, 0.4),
    ("consensus_rho0.6", "rho=0.6, lam=0.1", 0.1, 0.6),
    ("consensus_rho0.99", "rho=0.99, lam=0.1", 0.1, 0.99),
    ("consensus_lam0.01_rho0.05", "lam=0.01, rho=0.05", 0.01, 0.05),
    ("consensus_lam0.05_rho0.05", "lam=0.05, rho=0.05", 0.05, 0.05),
    ("consensus_lam0.1_rho0.05", "lam=0.1, rho=0.05", 0.1, 0.05),
    ("consensus_lam1.0", "lam=1.0, rho=0.2 (default)", 1.0, 0.2),
    ("consensus_lam1.0_rho0.0001", "lam=1.0, rho=0.0001", 1.0, 0.0001),
    ("consensus_lam1.0_rho0.99", "lam=1.0, rho=0.99", 1.0, 0.99),
    ("consensus_lam2.0", "lam=2.0, rho=0.2 (default)", 2.0, 0.2),
    ("consensus_lam5.0", "lam=5.0, rho=0.2 (default)", 5.0, 0.2),
    ("consensus_lam10.0", "lam=10.0, rho=0.2 (default)", 10.0, 0.2),
]


def reliability_curve(cert: np.ndarray, mse: np.ndarray, n_bins: int = N_BINS):
    """Quantile-bin cert, return per-bin (mean_cert, rmse, n) plus the
    low-bin/high-bin RMSE ratio as a single effect-size summary."""
    order = np.argsort(cert)
    cert_s, mse_s = cert[order], mse[order]
    edges = np.array_split(np.arange(len(cert_s)), n_bins)
    rows = []
    for idx in edges:
        if len(idx) == 0:
            continue
        rows.append((cert_s[idx].mean(), np.sqrt(mse_s[idx].mean()), len(idx)))
    lo_rmse = rows[0][1]
    hi_rmse = rows[-1][1]
    ratio = lo_rmse / hi_rmse if hi_rmse > 1e-9 else float("nan")
    return rows, ratio


def main():
    print(f"{'tag':<32}{'lam':>6}{'rho':>8}{'temper':>8}{'lo_rmse':>10}{'hi_rmse':>10}"
          f"{'ratio':>8}  monotonic")
    print("-" * 100)
    results = []
    for tag, label, lam, rho in RUNS:
        run_dir = ROOT / tag
        cert_path = run_dir / "cell_cert_steps.npy"
        mse_path = run_dir / "cell_mse_steps.npy"
        if not cert_path.exists() or not mse_path.exists():
            print(f"{tag:<32} MISSING")
            continue
        cert = np.load(cert_path).ravel()
        mse = np.load(mse_path).ravel()
        valid = ~np.isnan(cert) & ~np.isnan(mse)
        cert, mse = cert[valid], mse[valid]

        import time, os
        ctime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(run_dir)))
        tempered = "YES" if ctime > TEMPERING_CUTOFF else "no"

        rows, ratio = reliability_curve(cert, mse)
        rmses = [r[1] for r in rows]
        monotonic = all(rmses[i] >= rmses[i + 1] - 1e-4 for i in range(len(rmses) - 1))
        results.append((tag, label, lam, rho, tempered, rows, ratio, monotonic))
        print(f"{tag:<32}{lam:>6}{rho:>8}{tempered:>8}{rows[0][1]:>10.4f}"
              f"{rows[-1][1]:>10.4f}{ratio:>8.2f}  {monotonic}")

    print("\n=== Full reliability curves (cert_bin_mean -> RMSE, n_bins=8) ===")
    for tag, label, lam, rho, tempered, rows, ratio, monotonic in results:
        curve = " -> ".join(f"{c:.3f}:{r:.4f}" for c, r, n in rows)
        print(f"\n{label} [{tag}] (tempered={tempered})")
        print(f"  {curve}")
        print(f"  lo/hi RMSE ratio={ratio:.2f}  monotonic={monotonic}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
