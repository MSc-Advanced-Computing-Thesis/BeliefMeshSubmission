# Post-hoc analysis for the lambda ablation
# (run_certainty_calibration_diagnostic.py --lam ...). Pure read-from-saved-
# arrays, no re-simulation.

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path("runs/stage6/certainty_calibration_diagnostic/lam_ablation")
WORLDS = ["offset", "colour"]
LAMS = [0.5, 1.5, 5.0, 15.0, 50.0]
LAST_N = 50
CMAP = plt.get_cmap("viridis")
LAM_COLORS = {lam: CMAP(i / (len(LAMS) - 1)) for i, lam in enumerate(LAMS)}


def load(world, lam):
    tag = f"lam{lam:g}"
    d = np.load(ROOT / world / tag / "diagnostic_arrays.npz")
    return d


def block_r(cell_mse, cell_cert, lo, hi):
    m = cell_mse[lo:hi]
    c = cell_cert[lo:hi]
    valid = np.isfinite(m) & np.isfinite(c)
    if valid.sum() < 8:
        return None, int(valid.sum())
    r, p = pearsonr(c[valid], m[valid])
    return (r, p), int(valid.sum())


def node_block_r(node_mse, node_cert, lo, hi, hop=None, node_hop=None):
    m, c = node_mse[lo:hi], node_cert[lo:hi]
    if hop is not None:
        h = np.minimum(node_hop[lo:hi], 3)
        keep = h == hop
    else:
        keep = np.ones_like(m, dtype=bool)
    valid = keep & np.isfinite(m) & np.isfinite(c)
    if valid.sum() < 8:
        return None, int(valid.sum())
    r, p = pearsonr(c[valid], m[valid])
    return (r, p), int(valid.sum())


def step_means_from_log(applied_log, T):
    steps = applied_log[:, 0].astype(int)
    cert, nu, beta = applied_log[:, 2], applied_log[:, 3], applied_log[:, 4]
    mean_nu = np.full(T, np.nan); mean_beta = np.full(T, np.nan); cert_std = np.full(T, np.nan)
    for t in range(T):
        m = steps == t
        if m.sum() == 0:
            continue
        mean_nu[t] = nu[m].mean(); mean_beta[t] = beta[m].mean(); cert_std[t] = cert[m].std()
    return mean_nu, mean_beta, cert_std


def smooth(x, window=21):
    half = window // 2
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(len(x)):
        lo, hi = max(0, t - half), min(len(x), t + half + 1)
        seg = x[lo:hi]; seg = seg[np.isfinite(seg)]
        if len(seg):
            out[t] = seg.mean()
    return out


def main():
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("1. HEADLINE: whole_run MSE, last50 MSE, cert-MSE r, per lam per world")
    print("=" * 78)
    headline = {}
    for world in WORLDS:
        print(f"--- {world} ---")
        for lam in LAMS:
            d = load(world, lam)
            cms, ccs = d["cell_mse_steps"], d["cell_cert_steps"]
            whole = float(np.nanmean(cms))
            last50 = float(np.nanmean(cms[-LAST_N:]))
            avg_mse_last = np.nanmean(cms[-LAST_N:], axis=0)
            avg_cert_last = np.nanmean(ccs[-LAST_N:], axis=0)
            valid = np.isfinite(avg_mse_last) & np.isfinite(avg_cert_last)
            r = float(pearsonr(avg_cert_last[valid], avg_mse_last[valid])[0])
            headline[(world, lam)] = (whole, last50, r)
            print(f"  lam={lam:6.1f}  whole_run={whole:.4f}  last50={last50:.4f}  r={r:+.4f}")

    print()
    print("=" * 78)
    print("2. CERT-MSE r IN LARGE NON-OVERLAPPING BLOCKS (cell-level), per lam")
    print("=" * 78)
    blocks = [("0-50", 0, 50), ("100-150", 100, 150), ("170-220", 170, 220),
             ("270-320", 270, 320), ("340-390", 340, 390)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, world in zip(axes, WORLDS):
        print(f"--- {world} ---")
        for lam in LAMS:
            d = load(world, lam)
            cms, ccs = d["cell_mse_steps"], d["cell_cert_steps"]
            centers, rs = [], []
            row = []
            for name, lo, hi in blocks:
                res, n = block_r(cms, ccs, lo, hi)
                r = res[0] if res else np.nan
                row.append(f"{name}:{r:+.3f}(n={n})")
                centers.append((lo + hi) / 2)
                rs.append(r)
            ax.plot(centers, rs, "o-", color=LAM_COLORS[lam], label=f"lam={lam:g}", linewidth=1.4)
            print(f"  lam={lam:6.1f}  " + "  ".join(row))
        ax.axhline(0, color="#999", linewidth=0.6)
        ax.set_title(world); ax.set_xlabel("timestep (block center)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("block cert-MSE r"); axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "1_block_r_by_lam.png", dpi=150)
    plt.close(fig)
    print(f"saved {fig_dir / '1_block_r_by_lam.png'}")

    print()
    print("=" * 78)
    print("3. mean nu*, mean beta*, applied-cert spread, AND per-cell MSE over time")
    print("=" * 78)
    for world in WORLDS:
        fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
        print(f"--- {world} ---")
        for lam in LAMS:
            d = load(world, lam)
            applied_log = d["applied_log"]
            cms = d["cell_mse_steps"]
            T = cms.shape[0]
            mean_nu, mean_beta, cert_std = step_means_from_log(applied_log, T)
            per_step_mse = np.nanmean(cms, axis=(1, 2))
            axes[0].plot(smooth(mean_nu), color=LAM_COLORS[lam], label=f"lam={lam:g}", linewidth=1.3)
            axes[1].plot(smooth(mean_beta), color=LAM_COLORS[lam], linewidth=1.3)
            axes[2].plot(smooth(cert_std), color=LAM_COLORS[lam], linewidth=1.3)
            axes[3].plot(smooth(per_step_mse), color=LAM_COLORS[lam], linewidth=1.3)
            print(f"  lam={lam:6.1f}  nu* {np.nanmean(mean_nu[:50]):.3f}->{np.nanmean(mean_nu[-50:]):.3f}  "
                  f"beta* {np.nanmean(mean_beta[:50]):.5f}->{np.nanmean(mean_beta[-50:]):.5f}  "
                  f"cert_std {np.nanmean(cert_std[:50]):.5f}->{np.nanmean(cert_std[-50:]):.5f}  "
                  f"per-cell MSE {np.nanmean(per_step_mse[:50]):.5f}->{np.nanmean(per_step_mse[-50:]):.5f}  "
                  f"beta*/MSE ratio first50={np.nanmean(mean_beta[:50])/np.nanmean(per_step_mse[:50]):.4f} "
                  f"last50={np.nanmean(mean_beta[-50:])/np.nanmean(per_step_mse[-50:]):.4f}")
        axes[0].set_ylabel(r"mean $\nu^*$"); axes[0].legend(fontsize=7)
        axes[1].set_ylabel(r"mean $\beta^*$")
        axes[2].set_ylabel("applied cert std")
        axes[3].set_ylabel("mean per-cell MSE")
        axes[3].set_xlabel("timestep")
        for ax in axes:
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.suptitle(world)
        fig.tight_layout()
        fig.savefig(fig_dir / f"2_nu_beta_spread_mse_{world}.png", dpi=150)
        plt.close(fig)
        print(f"saved {fig_dir / f'2_nu_beta_spread_mse_{world}.png'}")

    print()
    print("=" * 78)
    print("4. APPLIED-CERTAINTY DISTRIBUTION + CLAMP BINDING, per lam per world")
    print("=" * 78)
    for world in WORLDS:
        print(f"--- {world} ---")
        for lam in LAMS:
            d = load(world, lam)
            applied_log = d["applied_log"]
            raw_unc, cert = applied_log[:, 1], applied_log[:, 2]
            bind_frac = float(np.mean(raw_unc >= 10.0))
            print(f"  lam={lam:6.1f}  applied_cert mean={cert.mean():.4f} std={cert.std():.4f} "
                  f"median={np.median(cert):.4f} p10={np.percentile(cert,10):.4f} "
                  f"p90={np.percentile(cert,90):.4f}  clamp_bind_frac={bind_frac:.4f}")

    print()
    print("=" * 78)
    print("5. HOP-0 (anchor) CERT-MSE r, first50 vs last50, per lam per world")
    print("=" * 78)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, world in zip(axes, WORLDS):
        print(f"--- {world} ---")
        lam_vals, r_first, r_last = [], [], []
        for lam in LAMS:
            d = load(world, lam)
            node_mse, node_cert, node_hop = d["node_mse"], d["node_cert"], d["node_hop"]
            T = node_mse.shape[0]
            res1, n1 = node_block_r(node_mse, node_cert, 0, 50, hop=0, node_hop=node_hop)
            res2, n2 = node_block_r(node_mse, node_cert, T - 50, T, hop=0, node_hop=node_hop)
            r1 = res1[0] if res1 else np.nan
            r2 = res2[0] if res2 else np.nan
            lam_vals.append(lam); r_first.append(r1); r_last.append(r2)
            print(f"  lam={lam:6.1f}  hop0 first50 r={r1:+.4f} (n={n1})  last50 r={r2:+.4f} (n={n2})")
        ax.plot(lam_vals, r_first, "o-", label="first50", color="#2C6E9E")
        ax.plot(lam_vals, r_last, "o-", label="last50", color="#C7622E")
        ax.set_xscale("log")
        ax.axhline(0, color="#999", linewidth=0.6)
        ax.set_title(world); ax.set_xlabel(r"$\lambda$")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("hop-0 cert-MSE r"); axes[0].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "3_hop0_r_vs_lam.png", dpi=150)
    plt.close(fig)
    print(f"saved {fig_dir / '3_hop0_r_vs_lam.png'}")


if __name__ == "__main__":
    main()
