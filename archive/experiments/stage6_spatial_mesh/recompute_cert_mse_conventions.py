# Recompute cert-MSE correlation under the corrected per-timestep-then-
# averaged convention (2026-08), reading ONLY already-saved
# cell_mse_steps.npy / cell_cert_steps.npy (or the lam-ablation npz's
# embedded arrays) -- no experiments are re-run. Pure analysis/reporting
# change; does not touch any experiment code path.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.cert_mse_metrics import (MIN_CELLS_PER_TIMESTEP,
                                                   averaged_then_correlated,
                                                   per_timestep_then_averaged,
                                                   pooled_raw)

SEEDS5 = [42, 1042, 2042, 3042, 4042]
LAST50 = "last50"


def load_cell_arrays(path: Path):
    mse_f = path / "cell_mse_steps.npy"
    cert_f = path / "cell_cert_steps.npy"
    if mse_f.exists() and cert_f.exists():
        return np.load(mse_f), np.load(cert_f)
    return None, None


def run_stats(cell_mse, cell_cert, lo, hi):
    old = averaged_then_correlated(cell_mse, cell_cert, lo, hi)
    pooled, n_pooled = pooled_raw(cell_mse, cell_cert, lo, hi)
    new_mean, new_std, n_t, rs = per_timestep_then_averaged(cell_mse, cell_cert, lo, hi)
    t, p = (float("nan"), float("nan"))
    if n_t > 1:
        t, p = ttest_1samp(rs, 0.0)
    return dict(old=old, pooled=pooled, pooled_n=n_pooled,
               new_mean=new_mean, new_std=new_std, new_n=n_t,
               new_t=float(t), new_p=float(p))


def aggregate_across_seeds(per_seed_stats: list[dict]) -> dict:
    olds = [s["old"] for s in per_seed_stats if s and np.isfinite(s["old"])]
    news = [s["new_mean"] for s in per_seed_stats if s and np.isfinite(s["new_mean"])]
    within_stds = [s["new_std"] for s in per_seed_stats if s and np.isfinite(s["new_std"])]
    return dict(
        old_mean=float(np.mean(olds)) if olds else float("nan"),
        old_std=float(np.std(olds)) if len(olds) > 1 else float("nan"),
        n_old=len(olds),
        new_mean=float(np.mean(news)) if news else float("nan"),
        new_std_across_seeds=float(np.std(news)) if len(news) > 1 else float("nan"),
        new_within_run_std_avg=float(np.mean(within_stds)) if within_stds else float("nan"),
        n_new=len(news),
    )


def print_group_table(title, rows):
    print(f"\n{'=' * 100}")
    print(title)
    print('=' * 100)
    print(f"{'arm':40s} {'old (avg-then-corr)':>22s} {'new (per-t-then-avg)':>26s} {'ordering note':>10s}")
    for name, agg in rows:
        old_s = f"{agg['old_mean']:+.4f}" + (f"±{agg['old_std']:.4f}" if agg['n_old'] > 1 else "") + f" (n={agg['n_old']})"
        new_s = (f"{agg['new_mean']:+.4f}" +
                 (f"±{agg['new_std_across_seeds']:.4f}(seeds)" if agg['n_new'] > 1 else "") +
                 f" [within-run std~{agg['new_within_run_std_avg']:.3f}] (n={agg['n_new']})")
        print(f"{name:40s} {old_s:>22s}   {new_s}")


# ── SANITY CHECK ──────────────────────────────────────────────────────────
def sanity_check():
    print("=" * 100)
    print("SANITY CHECK: three conventions side by side on representative runs")
    print("=" * 100)
    cases = {
        "colour, lam=5, epistemic (the cited -0.048/-0.436 case)":
            "runs/stage6/colour_world/uncertainty_measure_ablation/colour_unc_epistemic",
        "offset, lam=5, epistemic": "runs/stage6/offset_world/uncertainty_measure_ablation/unc_cmp_epistemic",
        "gossip fusion, seed 42": "runs/stage6/offset_world/gossip_comparator/gossip_cmp_fusion",
        "gossip fedavg_global, seed 42": "runs/stage6/offset_world/gossip_comparator/gossip_cmp_fedavg_global",
        "het_random_v2, seed 42": "runs/stage7/heterogeneous/het_random_v2",
    }
    for name, d in cases.items():
        mse, cert = load_cell_arrays(Path(d))
        T = mse.shape[0]
        s = run_stats(mse, cert, T - 50, T)
        ratio = abs(s["new_std"] / s["new_mean"]) if s["new_mean"] else float("nan")
        print(f"\n{name}")
        print(f"  old (averaged-then-correlated): {s['old']:+.4f}")
        print(f"  pooled_raw (n={s['pooled_n']}):            {s['pooled']:+.4f}")
        print(f"  NEW (per-timestep-then-averaged): mean={s['new_mean']:+.4f}  std={s['new_std']:.4f}  "
              f"(|std/mean|={ratio:.2f}, n_timesteps={s['new_n']}, one-sample t={s['new_t']:.2f} p={s['new_p']:.3g})")


# ── GROUP 1: gossip comparator ───────────────────────────────────────────
def group1():
    ROOT = Path("runs/stage6/offset_world/gossip_comparator")
    arms = ["fusion", "gossip_uniform", "gossip_weighted", "fedavg_global"]
    rows = []
    for arm in arms:
        per_seed = []
        for seed in SEEDS5:
            tag = f"gossip_cmp_{arm}" if seed == 42 else f"gossip_cmp_{arm}_seed{seed}"
            d = ROOT / tag
            mse, cert = load_cell_arrays(d)
            if mse is None:
                print(f"MISSING cell arrays: {d}")
                per_seed.append(None)
                continue
            T = mse.shape[0]
            per_seed.append(run_stats(mse, cert, T - 50, T))
        rows.append((arm, aggregate_across_seeds(per_seed)))
    print_group_table("GROUP 1: gossip comparator (fusion/gossip_uniform/gossip_weighted/fedavg_global), 5 seeds, last-50", rows)
    return rows


# ── GROUP 2: nig_product family ──────────────────────────────────────────
def group2():
    rows = []
    ROOT = Path("runs/stage6/offset_world/nig_product_comparator")
    for arm in ["nig_product", "nig_product_weighted"]:
        per_seed = []
        for seed in SEEDS5:
            tag = f"nig_cmp_{arm}" if seed == 42 else f"nig_cmp_{arm}_seed{seed}"
            d = ROOT / tag
            mse, cert = load_cell_arrays(d)
            if mse is None:
                print(f"MISSING cell arrays: {d}")
                per_seed.append(None)
                continue
            T = mse.shape[0]
            per_seed.append(run_stats(mse, cert, T - 50, T))
        rows.append((arm, aggregate_across_seeds(per_seed)))
    # nig_product_consensus: seed 42 only
    d = Path("runs/stage6/offset_world/nig_product_consensus_comparator/nig_cmp_nig_product_consensus")
    mse, cert = load_cell_arrays(d)
    if mse is None:
        print(f"MISSING cell arrays: {d}")
    else:
        T = mse.shape[0]
        rows.append(("nig_product_consensus (seed 42 only)", aggregate_across_seeds([run_stats(mse, cert, T - 50, T)])))
    print_group_table("GROUP 2: nig_product / nig_product_weighted (5 seeds) / nig_product_consensus (seed 42), last-50", rows)
    return rows


# ── GROUP 3: heterogeneous arms ──────────────────────────────────────────
def group3():
    ROOT = Path("runs/stage7/heterogeneous")
    rows = []
    for arm in ["het_random_v2", "het_clustered_v2"]:
        per_seed = []
        for seed in SEEDS5:
            tag = arm if seed == 42 else f"{arm}_seed{seed}"
            d = ROOT / tag
            mse, cert = load_cell_arrays(d)
            if mse is None:
                print(f"MISSING cell arrays: {d}")
                per_seed.append(None)
                continue
            T = mse.shape[0]
            per_seed.append(run_stats(mse, cert, T - 50, T))
        rows.append((arm, aggregate_across_seeds(per_seed)))
    print_group_table("GROUP 3: het_random_v2 / het_clustered_v2, 5 seeds, last-50", rows)
    return rows


# ── GROUP 4: node-failure sweep, PRE and POST windows ────────────────────
def group4():
    ROOT = Path("runs/stage6/offset_world/node_failure")
    PRE = (100, 150)
    POST = (340, 390)
    conditions_5seed = ["random_0pct", "random_10pct", "random_25pct", "random_40pct",
                        "clustered_10pct", "clustered_25pct", "clustered_40pct"]
    conditions_seed42only = ["random_50pct", "random_55pct", "random_60pct", "random_65pct"]

    rows_pre, rows_post = [], []
    for cond in conditions_5seed:
        per_seed_pre, per_seed_post = [], []
        for seed in SEEDS5:
            d = ROOT / f"nig_product_{cond}_seed{seed}"
            mse, cert = load_cell_arrays(d)
            if mse is None:
                print(f"MISSING cell arrays: {d}")
                per_seed_pre.append(None); per_seed_post.append(None)
                continue
            per_seed_pre.append(run_stats(mse, cert, *PRE))
            per_seed_post.append(run_stats(mse, cert, *POST))
        rows_pre.append((cond, aggregate_across_seeds(per_seed_pre)))
        rows_post.append((cond, aggregate_across_seeds(per_seed_post)))
    for cond in conditions_seed42only:
        d = ROOT / f"nig_product_{cond}_seed42"
        mse, cert = load_cell_arrays(d)
        if mse is None:
            print(f"MISSING cell arrays: {d}")
            continue
        rows_pre.append((cond + " (seed42 only)", aggregate_across_seeds([run_stats(mse, cert, *PRE)])))
        rows_post.append((cond + " (seed42 only)", aggregate_across_seeds([run_stats(mse, cert, *POST)])))

    print_group_table("GROUP 4a: node-failure sweep -- PRE-failure window (steps 100-150)", rows_pre)
    print_group_table("GROUP 4b: node-failure sweep -- POST-failure window (steps 340-390)", rows_post)
    return rows_pre, rows_post


# ── GROUP 5: uncertainty-measure ablation ────────────────────────────────
def group5():
    rows = []
    for world, root, prefix in [
        ("offset", Path("runs/stage6/offset_world/uncertainty_measure_ablation"), "unc_cmp_"),
        ("colour", Path("runs/stage6/colour_world/uncertainty_measure_ablation"), "colour_unc_"),
    ]:
        for measure in ["epistemic", "aleatoric", "total"]:
            d = root / f"{prefix}{measure}"
            mse, cert = load_cell_arrays(d)
            if mse is None:
                print(f"MISSING cell arrays: {d}")
                continue
            T = mse.shape[0]
            rows.append((f"{world}/{measure} (seed 42 only)",
                        aggregate_across_seeds([run_stats(mse, cert, T - 50, T)])))
    print_group_table("GROUP 5: uncertainty-measure ablation (epistemic/aleatoric/total), both worlds, seed 42, last-50", rows)
    return rows


# ── GROUP 6: lambda sweep ────────────────────────────────────────────────
def group6():
    ROOT = Path("runs/stage6/certainty_calibration_diagnostic/lam_ablation")
    rows = []
    for world in ["offset", "colour"]:
        for lam in [0.5, 1.5, 5.0, 15.0, 50.0]:
            d = ROOT / world / f"lam{lam:g}" / "diagnostic_arrays.npz"
            if not d.exists():
                print(f"MISSING: {d}")
                continue
            data = np.load(d)
            mse, cert = data["cell_mse_steps"], data["cell_cert_steps"]
            T = mse.shape[0]
            rows.append((f"{world}/lam={lam:g} (seed 42 only)",
                        aggregate_across_seeds([run_stats(mse, cert, T - 50, T)])))
    print_group_table("GROUP 6: lambda sweep, both worlds, seed 42, last-50", rows)
    return rows


if __name__ == "__main__":
    sanity_check()
    g1 = group1()
    g2 = group2()
    g3 = group3()
    g4pre, g4post = group4()
    g5 = group5()
    g6 = group6()
