# Recompute the certainty-calibration-erosion diagnostic under the corrected
# per-timestep-then-averaged cert-MSE convention (2026-08), reading ONLY the
# already-saved runs/stage6/certainty_calibration_diagnostic/lam_ablation/
# {world}/lam5/diagnostic_arrays.npz (offset & colour, seed 42, nig_product,
# lam=5.0 -- identical run to the original erosion diagnostic; lam_ablation's
# lam=5 point IS that same configuration, and its npz additionally has the
# cell-level arrays the original diagnostic script didn't save). No
# experiments re-run.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.cert_mse_metrics import (per_timestep_then_averaged, pooled_raw)

ROOT = Path("runs/stage6/certainty_calibration_diagnostic/lam_ablation")
WORLDS = ["offset", "colour"]
BLOCKS = [("0-50", 0, 50), ("100-150", 100, 150), ("170-220", 170, 220),
         ("270-320", 270, 320), ("340-390", 340, 390)]


def load(world):
    return np.load(ROOT / world / "lam5" / "diagnostic_arrays.npz")


def node_group(node_mse, node_cert, node_hop, lo, hi, hop=None):
    m, c = node_mse[lo:hi], node_cert[lo:hi]
    if hop is not None:
        h = np.minimum(node_hop[lo:hi], 3)
        keep = h == hop
        m = np.where(keep, m, np.nan)
        c = np.where(keep, c, np.nan)
    return m, c


print("=" * 100)
print("1. BLOCK-LEVEL CERT-MSE r (cell-level), OLD (pooled raw, as previously reported) vs NEW (per-timestep-then-avg)")
print("=" * 100)
for world in WORLDS:
    d = load(world)
    cms, ccs = d["cell_mse_steps"], d["cell_cert_steps"]
    print(f"\n--- {world} ---")
    for name, lo, hi in BLOCKS:
        old_r, old_n = pooled_raw(cms, ccs, lo, hi)
        new_mean, new_std, new_n, _ = per_timestep_then_averaged(cms, ccs, lo, hi)
        print(f"  {name:10s}  old(pooled,n={old_n:6d})={old_r:+.4f}   "
              f"new(per-t,n_t={new_n:2d})={new_mean:+.4f} +/- {new_std:.4f}")

print()
print("=" * 100)
print("2. HOP-DISTANCE BREAKDOWN (node-level), first50 vs last50, OLD vs NEW")
print("=" * 100)
for world in WORLDS:
    d = load(world)
    node_mse, node_cert, node_hop = d["node_mse"], d["node_cert"], d["node_hop"]
    T = node_mse.shape[0]
    print(f"\n--- {world} ---")
    for hop in range(3):
        m1, c1 = node_group(node_mse, node_cert, node_hop, 0, 50, hop)
        m2, c2 = node_group(node_mse, node_cert, node_hop, T - 50, T, hop)
        old1, n1 = pooled_raw(m1, c1, 0, 50)
        old2, n2 = pooled_raw(m2, c2, 0, 50)
        new1_mean, new1_std, new1_n, _ = per_timestep_then_averaged(m1, c1, 0, 50)
        new2_mean, new2_std, new2_n, _ = per_timestep_then_averaged(m2, c2, 0, 50)
        print(f"  hop{hop}: first50  old={old1:+.4f}(n={n1})  new={new1_mean:+.4f}+/-{new1_std:.4f}(n_t={new1_n})")
        print(f"        last50   old={old2:+.4f}(n={n2})  new={new2_mean:+.4f}+/-{new2_std:.4f}(n_t={new2_n})")

print()
print("=" * 100)
print("3. r vs CUMULATIVE TRAINING EXPOSURE -- attempting a per-timestep analog")
print("=" * 100)
for world in WORLDS:
    d = load(world)
    node_mse, node_cert, node_hop, node_cum = d["node_mse"], d["node_cert"], d["node_hop"], d["node_cum_steps"]
    T = node_mse.shape[0]
    print(f"\n--- {world} ---")

    # OLD (pooled, quantile-binned by cum_steps) -- as previously reported
    valid = np.isfinite(node_mse) & np.isfinite(node_cert) & (node_cum > 0)
    cum = node_cum[valid].astype(float)
    cert_v = node_cert[valid]
    mse_v = node_mse[valid]
    # need the wall-clock step of each record too, for the per-timestep attempt
    step_idx = np.broadcast_to(np.arange(T)[:, None], node_mse.shape)[valid]
    order = np.argsort(cum)
    cum_s, cert_s, mse_s, step_s = cum[order], cert_v[order], mse_v[order], step_idx[order]
    n_bins = 10
    edges = np.array_split(np.arange(len(cum_s)), n_bins)
    print("  OLD (pooled_raw within cum_steps quantile bin):")
    old_results = []
    for idx in edges:
        if len(idx) < 8:
            continue
        r, _ = pearsonr(cert_s[idx], mse_s[idx])
        old_results.append((float(np.median(cum_s[idx])), r, len(idx)))
        print(f"    median cum_steps={np.median(cum_s[idx]):8.0f}  r={r:+.4f}  n={len(idx)}")

    # NEW attempt: within each cum_steps bin, group further by shared
    # wall-clock step, correlate within (bin, step) subgroups with >=8 nodes,
    # then average those subgroup r's -- the per-timestep analog, restricted
    # to whatever subgroups actually have enough simultaneous nodes.
    print("  NEW attempt (per-timestep within each cum_steps bin, where enough simultaneous nodes exist):")
    any_new_computed = False
    for idx in edges:
        if len(idx) < 8:
            continue
        steps_in_bin = step_s[idx]
        sub_rs = []
        for t in np.unique(steps_in_bin):
            sub_mask = steps_in_bin == t
            if sub_mask.sum() < 8:
                continue
            r, _ = pearsonr(cert_s[idx][sub_mask], mse_s[idx][sub_mask])
            sub_rs.append(r)
        if sub_rs:
            any_new_computed = True
            print(f"    median cum_steps={np.median(cum_s[idx]):8.0f}  "
                  f"new_mean={np.mean(sub_rs):+.4f} std={np.std(sub_rs):.4f} "
                  f"(n_subgroups={len(sub_rs)}, out of {len(np.unique(steps_in_bin))} distinct steps in bin)")
        else:
            print(f"    median cum_steps={np.median(cum_s[idx]):8.0f}  "
                  f"NO subgroup reached >=8 simultaneous nodes (max nodes/step available: 36) -- not computable")
    if not any_new_computed:
        print("  => per-timestep convention is NOT computable on this axis for this mesh size "
              "(36 nodes total, split across cum_steps AND wall-clock-step within each bin "
              "leaves too few simultaneous observations). Falling back to pooled_raw above "
              "as the only viable estimator for this specific view.")

print()
print("=" * 100)
print("4. beta*/MSE RATIO CONFIRMATION (uses no correlation metric -- should be unchanged)")
print("=" * 100)
for world in WORLDS:
    d = load(world)
    applied_log = d["applied_log"]
    cms = d["cell_mse_steps"]
    T = cms.shape[0]
    steps = applied_log[:, 0].astype(int)
    beta = applied_log[:, 4]
    mean_beta = np.full(T, np.nan)
    for t in range(T):
        m = steps == t
        if m.sum():
            mean_beta[t] = beta[m].mean()
    per_step_mse = np.nanmean(cms, axis=(1, 2))
    ratio_first = np.nanmean(mean_beta[:50]) / np.nanmean(per_step_mse[:50])
    ratio_last = np.nanmean(mean_beta[-50:]) / np.nanmean(per_step_mse[-50:])
    print(f"  {world}: beta*/MSE ratio first50={ratio_first:.4f}  last50={ratio_last:.4f}  "
          f"(previously reported: offset 0.607->0.298, colour 1.345->0.460)")
