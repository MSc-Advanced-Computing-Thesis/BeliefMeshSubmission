# Corrupted-supervision comparator analysis (2026-08). Pure read-from-saved-
# arrays; launches nothing.
#
# Question: is BeliefMesh's insensitivity to corrupted ground truth a property
# of BELIEF FUSION, or of the mesh architecture generally? Each arm is compared
# against ITS OWN clean baseline, never against nig_product's.
#
# Mechanism checks, which matter as much as the headline degradation:
#   fedavg  -- corruption enters one shared model and is broadcast to every
#              node, so degradation should be GLOBAL, not concentrated near the
#              unreliable wearable.
#   gossip  -- corruption spreads through the overlap graph by weight
#              averaging, so the degraded region should GROW over the run.
#   routing -- if cells near the unreliable wearable are NOT elevated in an arm
#              that claims robustness, check whether evaluation is simply
#              routing around degraded nodes (it takes the most-certain belief
#              per cell across covering nodes) rather than the mesh genuinely
#              absorbing the corruption.
#
# Run: python -u experiments/stage6_spatial_mesh/analyze_corrupted_supervision.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.cert_mse_metrics import per_timestep_then_averaged
from stage6_spatial_mesh.run_offset_experiments import build_random_wander_path

ROOT = Path("runs/stage6/offset_world/sensor_reliability")
ARMS = ["nig_product", "naive", "gossip_uniform", "fedavg_global", "frozen"]
SEED, G, T = 42, 22, 390
FLOOR = 0.2          # % reproduction floor established earlier (GPU nondeterminism)
# wearable 2 (0-indexed) is the unreliable one: reliability 0.4, jitter 10.8 deg.
# Paths are built with seed 200+i, matching run_sensor_reliability.
BAD_W = 2


def load(cond: str, arm: str, name: str):
    p = ROOT / cond / f"{arm}_seed{SEED}" / name
    return np.load(p) if p.exists() else None


def bad_path_distance() -> np.ndarray:
    """(G,G) min Chebyshev distance from each cell to any cell the unreliable
    wearable visited. Chebyshev because FOVs are square."""
    path = build_random_wander_path(T, G, seed=200 + BAD_W)
    pts = np.unique(np.array([[int(p[0]), int(p[1])] for p in path]), axis=0)
    rr, cc = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    d = np.full((G, G), 1e9)
    for r, c in pts:
        d = np.minimum(d, np.maximum(np.abs(rr - r), np.abs(cc - c)))
    return d


def main():
    dist = bad_path_distance()
    print(f"unreliable wearable = index {BAD_W} (reliability 0.4, jitter 10.8 deg)")
    print(f"cells within distance<=3 of its path: {int((dist <= 3).sum())}/{G*G}\n")

    print("=== headline: each arm vs ITS OWN clean baseline ===")
    hdr = (f"{'arm':16s} {'clean':>9s} {'corrupt':>9s} {'abs d':>9s} {'rel %':>7s} | "
           f"{'cleanL50':>9s} {'corrL50':>9s} {'rel %':>7s} | {'exceeds floor?':>14s}")
    print(hdr); print("-" * len(hdr))
    rows = {}
    for arm in ARMS:
        cl, co = load("baseline", arm, "cell_mse_steps.npy"), load("matched_noise", arm, "cell_mse_steps.npy")
        if cl is None or co is None:
            print(f"{arm:16s} {'(missing)':>9s}")
            continue
        w0, w1 = float(np.nanmean(cl)), float(np.nanmean(co))
        l0, l1 = float(np.nanmean(cl[-50:])), float(np.nanmean(co[-50:]))
        rel, relL = 100 * (w1 - w0) / w0, 100 * (l1 - l0) / l0
        rows[arm] = dict(cl=cl, co=co, w0=w0, w1=w1, rel=rel)
        flag = "YES" if abs(rel) > FLOOR else "no (within noise)"
        print(f"{arm:16s} {w0:9.5f} {w1:9.5f} {w1-w0:+9.5f} {rel:+7.2f} | "
              f"{l0:9.5f} {l1:9.5f} {relL:+7.2f} | {flag:>14s}")

    print("\n=== cert-MSE r (corrected convention), arms producing uncertainty ===")
    for arm in ARMS:
        out = []
        for cond in ("baseline", "matched_noise"):
            m, c = load(cond, arm, "cell_mse_steps.npy"), load(cond, arm, "cell_cert_steps.npy")
            if m is None or c is None:
                continue
            r, s, _, _ = per_timestep_then_averaged(m, c, m.shape[0] - 50, m.shape[0])
            out.append(f"{cond}={r:+.4f}+/-{s:.4f}")
        if out:
            print(f"  {arm:16s} " + "  ".join(out))

    print("\n=== MECHANISM 1: is degradation global or local to the bad wearable? ===")
    print(f"{'arm':16s} {'r(deg,dist)':>12s} {'p':>10s} | {'near d<=3':>10s} {'far d>=7':>10s} {'near-far':>10s}")
    for arm, d in rows.items():
        deg = np.nanmean(d["co"], axis=0) - np.nanmean(d["cl"], axis=0)   # (G,G)
        ok = np.isfinite(deg)
        r, p = pearsonr(dist[ok].ravel(), deg[ok].ravel())
        near, far = deg[(dist <= 3) & ok].mean(), deg[(dist >= 7) & ok].mean()
        print(f"{arm:16s} {r:+12.3f} {p:10.2e} | {near:+10.6f} {far:+10.6f} {near-far:+10.6f}")
    print("  r>0 => degradation grows WITH distance from the bad wearable (global/diffuse)")
    print("  r<0 => degradation concentrated NEAR it (local)")

    print("\n=== MECHANISM 2: does the degraded region grow over the run? ===")
    print(f"{'arm':16s} " + " ".join(f"{f'q{i+1}':>9s}" for i in range(4))
          + f" {'frac cells worse q1':>20s} {'q4':>8s}")
    for arm, d in rows.items():
        qs, fr = [], []
        for i in range(4):
            lo, hi = i * T // 4, (i + 1) * T // 4
            dg = np.nanmean(d["co"][lo:hi], axis=0) - np.nanmean(d["cl"][lo:hi], axis=0)
            qs.append(np.nanmean(dg))
            fr.append(np.nanmean(dg > 0))
        print(f"{arm:16s} " + " ".join(f"{q:+9.6f}" for q in qs)
              + f" {fr[0]:20.3f} {fr[3]:8.3f}")
    print("  a rising series / rising fraction => corruption spreading through the mesh")

    print("\n=== MECHANISM 3: routing check -- absolute error near the bad path ===")
    print(f"{'arm':16s} {'cond':>13s} {'near d<=3':>10s} {'far d>=7':>10s} {'ratio':>7s}")
    for arm in rows:
        for cond, key in (("clean", "cl"), ("corrupt", "co")):
            m = np.nanmean(rows[arm][key], axis=0)
            ok = np.isfinite(m)
            near, far = m[(dist <= 3) & ok].mean(), m[(dist >= 7) & ok].mean()
            print(f"{arm:16s} {cond:>13s} {near:10.6f} {far:10.6f} {near/far:7.3f}")
    print("  ratio ~1 under corruption => no local scar; either genuine absorption")
    print("  or evaluation routing around the degraded node (see coverage note)")


if __name__ == "__main__":
    main()
