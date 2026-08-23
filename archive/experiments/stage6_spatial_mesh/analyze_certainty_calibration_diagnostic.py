# Post-hoc analysis for run_certainty_calibration_diagnostic.py -- pure
# read-from-saved-arrays, no re-simulation. Produces the four requested
# views (rolling cert-MSE r, nu*/beta*/cert-spread evolution, hop-broken-down
# rolling r, cumulative-training-step-aligned r) for both worlds and saves
# figures + a printed summary.

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path("runs/stage6/certainty_calibration_diagnostic")
WORLDS = ["offset", "colour"]
WINDOW = 21  # steps, centered rolling window for the wall-clock time series
COLORS = {"offset": "#2C6E9E", "colour": "#C7622E"}
HOP_COLORS = {0: "#FFD700", 1: "#FFA500", 2: "#FF4500", 3: "#8B0000"}


def load(world):
    d = np.load(ROOT / world / "diagnostic_arrays.npz")
    return d["node_mse"], d["node_cert"], d["node_hop"], d["node_cum_steps"], d["applied_log"]


def rolling_r(node_mse, node_cert, window=WINDOW, hop_mask=None, node_hop=None):
    """Pearson r of pooled (cert, mse) across nodes, within a centered rolling
    window of wall-clock steps. hop_mask: if given with node_hop, restrict to
    cells at that hop distance (capped at 3) within the window."""
    T = node_mse.shape[0]
    half = window // 2
    centers, rs, ns = [], [], []
    for t in range(half, T - half):
        lo, hi = t - half, t + half + 1
        mse_win = node_mse[lo:hi]
        cert_win = node_cert[lo:hi]
        if hop_mask is not None:
            hop_win = np.minimum(node_hop[lo:hi], 3)
            keep = (hop_win == hop_mask)
        else:
            keep = np.ones_like(mse_win, dtype=bool)
        valid = keep & np.isfinite(mse_win) & np.isfinite(cert_win)
        if valid.sum() < 8:
            continue
        r, _ = pearsonr(cert_win[valid], mse_win[valid])
        centers.append(t)
        rs.append(r)
        ns.append(int(valid.sum()))
    return np.array(centers), np.array(rs), np.array(ns)


def step_means_from_log(applied_log, T):
    """Per-step mean nu, mean beta, cert std from the (step, raw_unc, cert, nu,
    beta) applied_uncertainty_log, then centered rolling-smoothed."""
    steps = applied_log[:, 0].astype(int)
    cert = applied_log[:, 2]
    nu = applied_log[:, 3]
    beta = applied_log[:, 4]
    mean_nu = np.full(T, np.nan)
    mean_beta = np.full(T, np.nan)
    cert_std = np.full(T, np.nan)
    for t in range(T):
        m = steps == t
        if m.sum() == 0:
            continue
        mean_nu[t] = nu[m].mean()
        mean_beta[t] = beta[m].mean()
        cert_std[t] = cert[m].std()
    return mean_nu, mean_beta, cert_std


def smooth(x, window=WINDOW):
    half = window // 2
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(len(x)):
        lo, hi = max(0, t - half), min(len(x), t + half + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[t] = seg.mean()
    return out


def cum_step_binned_r(node_mse, node_cert, node_cum_steps, n_bins=20):
    """Pool all (cum_steps, cert, mse) records across nodes/steps, bin by
    QUANTILE of cum_steps (so each bin has ~equal n), compute r per bin."""
    valid = np.isfinite(node_mse) & np.isfinite(node_cert) & (node_cum_steps > 0)
    cum = node_cum_steps[valid].astype(float)
    cert = node_cert[valid]
    mse = node_mse[valid]
    order = np.argsort(cum)
    cum, cert, mse = cum[order], cert[order], mse[order]
    edges = np.array_split(np.arange(len(cum)), n_bins)
    centers, rs, ns = [], [], []
    for idx in edges:
        if len(idx) < 8:
            continue
        r, _ = pearsonr(cert[idx], mse[idx])
        centers.append(float(np.median(cum[idx])))
        rs.append(r)
        ns.append(len(idx))
    return np.array(centers), np.array(rs), np.array(ns)


def wallclock_binned_r(node_mse, node_cert, n_bins=20):
    T = node_mse.shape[0]
    edges = np.linspace(0, T, n_bins + 1).astype(int)
    centers, rs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mse_win = node_mse[lo:hi]
        cert_win = node_cert[lo:hi]
        valid = np.isfinite(mse_win) & np.isfinite(cert_win)
        if valid.sum() < 8:
            continue
        r, _ = pearsonr(cert_win[valid], mse_win[valid])
        centers.append((lo + hi) / 2)
        rs.append(r)
        ns.append(int(valid.sum()))
    return np.array(centers), np.array(rs), np.array(ns)


def main():
    data = {w: load(w) for w in WORLDS}
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("1. ROLLING CERT-MSE CORRELATION (overall, wall-clock, window=%d)" % WINDOW)
    print("=" * 70)
    fig, ax = plt.subplots(figsize=(7, 4))
    overall = {}
    for w in WORLDS:
        node_mse, node_cert, node_hop, node_cum, applied_log = data[w]
        centers, rs, ns = rolling_r(node_mse, node_cert)
        overall[w] = (centers, rs, ns)
        ax.plot(centers, rs, color=COLORS[w], label=w, linewidth=1.6)
        first10 = rs[:10].mean() if len(rs) >= 10 else np.nan
        last10 = rs[-10:].mean() if len(rs) >= 10 else np.nan
        print(f"{w}: r at start (mean of first 10 window-centers)={first10:.4f}, "
              f"r at end (mean of last 10)={last10:.4f}, delta={last10-first10:+.4f}")
        # simple linear trend (slope) over the whole rolling series as a decay indicator
        if len(rs) > 5:
            slope = np.polyfit(centers, rs, 1)[0]
            print(f"    linear trend slope over full run: {slope:+.6f} per step "
                  f"({slope*390:+.4f} total over 390 steps)")
    ax.axhline(0, color="#999", linewidth=0.6)
    ax.set_xlabel("timestep (window center)")
    ax.set_ylabel("rolling cert-MSE Pearson r")
    ax.set_title(f"Rolling cert-MSE correlation (window={WINDOW} steps)")
    ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "1_rolling_r_overall.png", dpi=150)
    plt.close(fig)
    print(f"saved {fig_dir / '1_rolling_r_overall.png'}")

    print()
    print("=" * 70)
    print("2. FUSED nu*, beta*, APPLIED-CERT SPREAD OVER TIME")
    print("=" * 70)
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for w in WORLDS:
        node_mse, node_cert, node_hop, node_cum, applied_log = data[w]
        T = node_mse.shape[0]
        mean_nu, mean_beta, cert_std = step_means_from_log(applied_log, T)
        axes[0].plot(smooth(mean_nu), color=COLORS[w], label=w, linewidth=1.4)
        axes[1].plot(smooth(mean_beta), color=COLORS[w], label=w, linewidth=1.4)
        axes[2].plot(smooth(cert_std), color=COLORS[w], label=w, linewidth=1.4)
        print(f"{w}: mean nu* first50={np.nanmean(mean_nu[:50]):.3f} -> "
              f"last50={np.nanmean(mean_nu[-50:]):.3f}")
        print(f"{w}: mean beta* first50={np.nanmean(mean_beta[:50]):.5f} -> "
              f"last50={np.nanmean(mean_beta[-50:]):.5f}")
        print(f"{w}: applied-cert std first50={np.nanmean(cert_std[:50]):.5f} -> "
              f"last50={np.nanmean(cert_std[-50:]):.5f}")
    axes[0].set_ylabel(r"mean $\nu^*$"); axes[0].legend()
    axes[1].set_ylabel(r"mean $\beta^*$")
    axes[2].set_ylabel("applied cert std (spread)")
    axes[2].set_xlabel("timestep")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "2_nu_beta_spread.png", dpi=150)
    plt.close(fig)
    print(f"saved {fig_dir / '2_nu_beta_spread.png'}")

    print()
    print("=" * 70)
    print("3. ROLLING CERT-MSE r BROKEN DOWN BY HOP DISTANCE")
    print("=" * 70)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, w in zip(axes, WORLDS):
        node_mse, node_cert, node_hop, node_cum, applied_log = data[w]
        for hop in range(4):
            centers, rs, ns = rolling_r(node_mse, node_cert, hop_mask=hop, node_hop=node_hop)
            if len(centers) == 0:
                continue
            label = f"hop {hop}" if hop < 3 else "hop 3+"
            ax.plot(centers, rs, color=HOP_COLORS[hop], label=label, linewidth=1.3)
            first10 = rs[:10].mean() if len(rs) >= 10 else np.nan
            last10 = rs[-10:].mean() if len(rs) >= 10 else np.nan
            print(f"{w} hop{hop}: n_windows={len(centers)} r_start={first10:.4f} "
                  f"r_end={last10:.4f} delta={last10-first10:+.4f}")
        ax.axhline(0, color="#999", linewidth=0.6)
        ax.set_title(w); ax.set_xlabel("timestep")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("rolling cert-MSE r"); axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "3_rolling_r_by_hop.png", dpi=150)
    plt.close(fig)
    print(f"saved {fig_dir / '3_rolling_r_by_hop.png'}")

    print()
    print("=" * 70)
    print("4. r vs CUMULATIVE TRAINING STEPS PER NODE, vs r vs WALL-CLOCK STEP")
    print("=" * 70)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for w in WORLDS:
        node_mse, node_cert, node_hop, node_cum, applied_log = data[w]
        c_cum, r_cum, n_cum = cum_step_binned_r(node_mse, node_cert, node_cum)
        c_wc, r_wc, n_wc = wallclock_binned_r(node_mse, node_cert)
        axes[0].plot(c_cum, r_cum, "o-", color=COLORS[w], label=w, markersize=3, linewidth=1.2)
        axes[1].plot(c_wc, r_wc, "o-", color=COLORS[w], label=w, markersize=3, linewidth=1.2)
        print(f"{w}: cum-steps-binned r: first bin={r_cum[0]:.4f} (median cum_steps={c_cum[0]:.0f}), "
              f"last bin={r_cum[-1]:.4f} (median cum_steps={c_cum[-1]:.0f})")
        print(f"{w}: wallclock-binned r: first bin={r_wc[0]:.4f}, last bin={r_wc[-1]:.4f}")
    for ax in axes:
        ax.axhline(0, color="#999", linewidth=0.6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend()
    axes[0].set_xlabel("median cumulative training samples (per-node, pooled, quantile-binned)")
    axes[0].set_ylabel("cert-MSE r within bin")
    axes[0].set_title("r vs cumulative training amount")
    axes[1].set_xlabel("wall-clock timestep (equal-width bins)")
    axes[1].set_title("r vs wall-clock timestep")
    fig.tight_layout()
    fig.savefig(fig_dir / "4_r_vs_cumsteps_vs_wallclock.png", dpi=150)
    plt.close(fig)
    print(f"saved {fig_dir / '4_r_vs_cumsteps_vs_wallclock.png'}")


if __name__ == "__main__":
    main()
