# Chapter 5 Block-2 figures. Stored artefacts only; no model, no re-evaluation.
# PNG only, 6.3 in wide (LaTeX text width), dpi 200.
#
# Run: python -u experiments/section5_2/make_block2_figures.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_estimators import LAST, load, truth_for, wrap
from stage6_spatial_mesh.run_offset_experiments import build_dynamic_offset_field
from beliefmesh.node.mesh import fov_cells

STUDENT_CENTRE, FOV, GRID_G, T_STEPS = (6, 3), 7, 22, 390


def offset_trace():
    """Environment trace for trajectory panels: MEAN |offset| in degrees over
    the STUDENT's field of view, per timestep.

    Chosen by measurement. Against the frozen arm's trajectory -- which never
    trains, so its error is driven purely by how far the field has moved from
    the zero-offset condition the pretrained model assumes:
        mean |offset|, student FOV   r = 0.941
        RMS offset,    student FOV   r = 0.942
        spatial sd,    student FOV   r = 0.633
        mean |offset|, WHOLE field   r = 0.452
        |d offset / dt|              r = -0.056
    Mean and RMS over the FOV are tied; mean |offset| is used because it reads
    directly in degrees. The whole-field version is less than half as
    explanatory -- these two nodes only ever see a 7x7 window, so the field
    average is the wrong quantity. Rate of change explains nothing, so the
    frozen arm tracks WHERE the field is, not how fast it moves.
    """
    F = build_dynamic_offset_field(GRID_G, T_STEPS)
    cells = np.array(fov_cells(*STUDENT_CENTRE, FOV, GRID_G))
    m = np.zeros((GRID_G, GRID_G), bool)
    m[cells[:, 0], cells[:, 1]] = True
    return np.abs(F[:, m]).mean(axis=1)

FIGS = Path("figures")
W, DPI = 6.3, 200
INK, RED, BLUE, GREY, GREEN = "#22252a", "#c1121f", "#1f5fc1", "#8a8f98", "#1b7a3e"
SEEDS = [42, 1042, 2042, 3042, 4042]
# frozen never trains, so its cell-space value is target-rule independent
FROZEN_WHOLE, FROZEN_LAST50 = 0.09697, 0.09880


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / (name + ".png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % p)


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def metrics(d: Path):
    R = load(d)
    mse, nig = R["mse"], R["nig"]
    T = mse.shape[0]
    sl = slice(T - LAST, T)
    e = np.sqrt(mse[sl])
    nu, al, be = (nig[sl, ..., i] for i in range(3))
    k = np.isfinite(e) & np.isfinite(nu) & (al > 1.0)
    sc = np.sqrt(be[k] * (1 + nu[k]) / (nu[k] * al[k]))
    hw = sps.t.ppf(0.95, df=2 * al[k]) * sc
    m = float(np.nanmean(mse[sl]))
    return dict(whole=float(np.nanmean(mse)), last50=m,
                cov=float((e[k] <= hw).mean()),
                ratio=float(np.mean(hw) * 180 / (np.sqrt(m) * 180)))


# ---------------------------------------------------------------- 5.2
def fig_52():
    """5.2. Calibration bars ordered and grouped BY SUPERVISION SOURCE:
    frozen (no training) -> direct (ground truth) -> the two student arms
    (peer beliefs), the second of which is the adopted default. The two
    student arms are a matched pair: identical seeds, environment, lr, lam,
    wearable configuration, exclusions and node placement; their manifests
    differ only in the arm/condition label.
    """
    traj = [("frozen", "frozen", GREEN),
            ("direct", "direct", BLUE),
            ("student", "student (mode target)", "#8a5fc1"),
            ("student_sampled", "student (sampled target)", RED)]
    cal = [("frozen", "frozen", GREEN), ("direct", "direct", BLUE),
           ("student", "mode\ntarget", "#8a5fc1"),
           ("student_sampled", "sampled\ntarget", RED)]

    fig = plt.figure(figsize=(W, 4.25))
    # right margin only needs to clear the twin-axis label on the top panel;
    # the bottom row has no right-hand axis, so it can run wider.
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.935, bottom=0.145,
                          top=0.955, hspace=0.52, wspace=0.30)
    ax = fig.add_subplot(gs[0, :])

    axt = ax.twinx()
    w = 15
    sm = lambda y: np.convolve(y, np.ones(w) / w, mode="valid")
    tr = sm(offset_trace())
    xs = np.arange(len(tr)) + w // 2
    # thin dashed line, no fill: context, not a foreground series
    axt.plot(xs, tr, "--", color="#9aa0a8", lw=0.9, zorder=0)
    axt.set_ylabel("mean |offset| over FOV (deg)", fontsize=7.5, color="#7c828a")
    axt.tick_params(axis="y", labelsize=7.5, labelcolor="#7c828a")
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    for tag, lab, col in traj:
        A = []
        for sd_ in SEEDS:
            f = Path("runs/section5_2/offset/offset") / (tag + "_seed" + str(sd_)) / "node_mse_steps.npy"
            if f.exists():
                A.append(np.load(f)[:, 1])
        if not A:
            continue
        A = np.array(A)
        mu = sm(np.array(A).mean(0))
        x = np.arange(len(mu)) + w // 2
        ax.plot(x, mu, "-", color=col, lw=1.25, label=lab, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("timestep", fontsize=8)
    ax.set_ylabel("student cell-space MSE", fontsize=8)
    ax.set_title("trajectory against the moving offset field", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left", ncol=2)

    cov, rat, labs, cols = [], [], [], []
    for tag, lab, col in cal:
        cs, rs = [], []
        for sd_ in SEEDS:
            mp = Path("runs/section5_2/offset/offset") / (tag + "_seed" + str(sd_)) / "manifest.yaml"
            if not mp.exists():
                continue
            r = yaml.safe_load(open(mp))["results"]
            c = r.get("student_coverage_90") or {}
            if c.get("empirical") is None:
                continue
            cs.append(c["empirical"])
            rs.append(c["mean_half_width_deg"] /
                      (np.sqrt(r["last50_mse_cell_space"]["student"]) * 180))
        if cs:
            cov.append(ms(cs)); rat.append(ms(rs)); labs.append(lab); cols.append(col)
    x = np.arange(len(labs))

    for col_i, (vals, ref, ylab, title) in enumerate((
            (cov, 0.90, "90% coverage", "coverage"),
            (rat, 1.00, "half-width : RMS", "interval vs error"))):
        a = fig.add_subplot(gs[1, col_i])
        a.bar(x, [v[0] for v in vals], 0.66, yerr=[v[1] for v in vals], capsize=3,
              color=cols, edgecolor="white", lw=0.6)
        a.axhline(ref, color=RED, ls=":", lw=1.0)
        a.axvline(0.5, color="#c9ced6", lw=0.9, zorder=0)
        a.axvline(1.5, color="#c9ced6", lw=0.9, zorder=0)
        a.set_xticks(x)
        a.set_xticklabels(labs, fontsize=6.4)
        a.set_ylabel(ylab, fontsize=8)
        a.set_title(title, fontsize=9)
        a.tick_params(labelsize=8, pad=1.5)
        a.set_xlim(-0.6, len(x) - 0.4)
        # square bracket spanning the two student bars
        tf = a.get_xaxis_transform()
        yb, yt = -0.30, -0.255
        a.plot([1.68, 3.32], [yb, yb], color=INK, lw=0.8, clip_on=False, transform=tf)
        a.plot([1.68, 1.68], [yb, yt], color=INK, lw=0.8, clip_on=False, transform=tf)
        a.plot([3.32, 3.32], [yb, yt], color=INK, lw=0.8, clip_on=False, transform=tf)
        a.text(2.5, yb - 0.04, "student", ha="center", va="top", fontsize=7,
               transform=tf, clip_on=False)
    save(fig, "ch5_5_2_peer_supervision")


# 5.3.1's three-panel parity figure was REPLACED by a table
# (experiments/section5_2/make_531_tables.py -> table_5_3_1_parity.csv):
# the arms are indistinguishable, which a table states directly and bars do not.

# ---------------------------------------------------------------- 5.3.1b
def fig_531b():
    dirs = [Path(f"runs/chapter5_v2/s531_b2/nig_product_sampled_seed{s}") for s in SEEDS]
    per_k = defaultdict(lambda: ([], []))
    for d in dirs:
        R = load(d)
        seed = int(d.name.rsplit("_seed", 1)[1])
        T, G = R["mse"].shape[0], R["mse"].shape[1]
        sl = slice(T - LAST, T)
        tr = truth_for(seed, T, G)[-LAST:]
        fe = wrap(R["fused"][sl, ..., 0].astype(np.float64) - tr) ** 2
        nc = R["ncov"][sl]
        for k in sorted(set(int(v) for v in np.unique(nc) if v > 0)):
            m = nc == k
            per_k[k][0].append(float(np.nanmean(R["mse"][sl][m])))
            per_k[k][1].append(float(np.nanmean(fe[m])))
    ks = sorted(per_k)
    a_mu = [ms(per_k[k][0])[0] for k in ks]
    f_mu = [ms(per_k[k][1])[0] for k in ks]
    a_sd = [ms(per_k[k][0])[1] for k in ks]
    f_sd = [ms(per_k[k][1])[1] for k in ks]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.155, top=0.925)
    ax.errorbar(ks, a_mu, yerr=a_sd, fmt="o-", color=INK, ms=5, lw=1.6,
                capsize=3, label="argmax (reported convention)")
    ax.errorbar(ks, f_mu, yerr=f_sd, fmt="s--", color=GREEN, ms=5, lw=1.6,
                capsize=3, label="fused readout")
    for k, a, f in zip(ks, a_mu, f_mu):
        pct = 100 * (f - a) / a
        ax.annotate("%+.0f%%" % pct, (k, f), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7.5,
                    color=GREY if abs(pct) < 0.5 else GREEN)
    ax.annotate("control\n(identical)", (ks[0], a_mu[0]), textcoords="offset points",
                xytext=(8, 8), fontsize=7.5, color=GREY)
    ax.set_xticks(ks)
    ax.set_xlabel("covering beliefs per cell (n_cov)", fontsize=9)
    ax.set_ylabel("cell-space MSE", fontsize=9)
    ax.set_title("MSE under the argmax and fused readout estimators", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False)
    save(fig, "ch5_5_3_1b_estimator_comparison")


# ---------------------------------------------------------------- 5.5
def fig_55():
    arms = [("fusion", "runs/chapter5_v2/s5_5_gossip_fusion/*/manifest.yaml", INK, True),
            ("gossip_uniform", "runs/stage6/offset_world/gossip_comparator/gossip_cmp_gossip_uniform*/manifest.yaml", RED, False),
            ("gossip_weighted", "runs/stage6/offset_world/gossip_comparator/gossip_cmp_gossip_weighted*/manifest.yaml", BLUE, False),
            ("fedavg_global", "runs/stage6/offset_world/gossip_comparator/gossip_cmp_fedavg_global*/manifest.yaml", GREEN, False)]
    data = {}
    for name, pat, col, sampled in arms:
        w, l, c = [], [], []
        for p in sorted(glob.glob(pat)):
            d = os.path.dirname(p)
            f = d + "/cell_mse_steps.npy"
            if not os.path.exists(f):
                continue
            mse = np.load(f)
            w.append(float(np.nanmean(mse)))
            l.append(float(np.nanmean(mse[-LAST:])))
            cb = (yaml.safe_load(open(p))["results"] or {}).get("comm_bytes_total")
            if cb:
                c.append(cb)
        data[name] = (ms(w), ms(l), st.mean(c) if c else np.nan, col, sampled)
    names = [a[0] for a in arms]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.26, top=0.88, wspace=0.42)
    for ax, idx, lab in ((axes[0], 0, "whole-run MSE"), (axes[1], 1, "last-50 MSE")):
        mu = [data[n][idx][0] for n in names]
        sd = [data[n][idx][1] for n in names]
        ax.bar(range(4), mu, 0.6, yerr=sd, capsize=3,
               color=[data[n][3] for n in names], edgecolor="white", lw=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["fusion", "goss_u", "goss_w", "fedavg"], fontsize=7, rotation=30)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=8)
    ax = axes[2]
    ax.bar(range(4), [data[n][2] for n in names], 0.6,
           color=[data[n][3] for n in names], edgecolor="white", lw=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["fusion", "goss_u", "goss_w", "fedavg"], fontsize=7, rotation=30)
    ax.set_ylabel("bytes transmitted", fontsize=8)
    ax.tick_params(labelsize=8)
    axes[0].set_title("accuracy", fontsize=9)
    axes[1].set_title("steady state", fontsize=9)
    axes[2].set_title("communication", fontsize=9)
    save(fig, "ch5_5_5_parameter_exchange")


# ---------------------------------------------------------------- 5.7
def fig_57():
    G = defaultdict(list)
    for p in sorted(glob.glob("runs/chapter5_v2/s5_7_node_loss/*/manifest.yaml")):
        m = yaml.safe_load(open(p))
        r = m["results"]
        n = os.path.basename(os.path.dirname(p)).rsplit("_seed", 1)[0]
        pv = r.get("paired_vs_control") or {}
        G[n].append((r.get("dead_zone_fraction"), pv.get("mean_diff")))
    def parse(n):
        return ("clustered" if "clustered" in n else "random",
                int(n.split("_")[-1].replace("pct", "")))
    series = defaultdict(list)
    for n, v in G.items():
        kind, pct = parse(n)
        series[kind].append((pct, ms([x[0] for x in v]),
                             ms([x[1] for x in v if x[1] is not None]) if any(x[1] is not None for x in v) else None))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.88, wspace=0.32)
    for kind, col, mk in (("random", INK, "o-"), ("clustered", RED, "s--")):
        pts = sorted(series[kind])
        ax = axes[0]
        ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts],
                    yerr=[p[1][1] for p in pts], fmt=mk, color=col, ms=4, lw=1.3,
                    capsize=2.5, label=kind)
    axes[0].set_xlabel("nodes failed (%)", fontsize=8)
    axes[0].set_ylabel("dead-zone fraction", fontsize=8)
    axes[0].set_title("coverage loss", fontsize=9)
    axes[0].tick_params(labelsize=8)
    axes[0].legend(fontsize=6.8, frameon=False)
    ax = axes[1]
    for kind, col, mk in (("random", INK, "o-"), ("clustered", RED, "s--")):
        pts = [p for p in sorted(series[kind]) if p[2] is not None]
        ax.errorbar([p[0] for p in pts], [p[2][0] for p in pts],
                    yerr=[p[2][1] for p in pts], fmt=mk, color=col, ms=4, lw=1.3,
                    capsize=2.5, label=kind)
    ax.axhline(0.0, color=GREY, ls=":", lw=1.0)
    ax.set_xlabel("nodes failed (%)", fontsize=8)
    ax.set_ylabel("MSE diff vs 0% control", fontsize=8)
    ax.set_title("surviving region", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.03, 0.06, "below 0 = no worse than control", transform=ax.transAxes,
            fontsize=6.2, color=GREY)
    save(fig, "ch5_5_7_node_loss")


# ---------------------------------------------------------------- table 5.4.2
def table_542():
    rows = []
    for name, root in (("static", "runs/chapter5_v2/s5_4_2_static"),
                       ("dynamic", "runs/chapter5_v2/s5_4_2_dynamic")):
        M = [metrics(Path(os.path.dirname(p)))
             for p in sorted(glob.glob(root + "/*/manifest.yaml"))]
        rows.append((name, len(M)) + tuple(ms([m[k] for m in M])
                                           for k in ("whole", "last50", "cov", "ratio")))
    out = Path("runs/chapter5_v2/table_5_4_2.csv")
    with open(out, "w", encoding="utf8") as f:
        f.write("field,n_seeds,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,"
                "last50_mse_sd,coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd\n")
        for r in rows:
            f.write("%s,%d," % (r[0], r[1]) + ",".join("%.6f" % x for t in r[2:] for x in t) + "\n")
    print("\n=== TABLE 5.4.2 (static vs dynamic offset field, 5 seeds) ===")
    print("%-9s %3s %-21s %-21s %-17s %-17s" % ("field", "n", "whole-run MSE",
                                                "last-50 MSE", "90% coverage", "hw:RMS"))
    for r in rows:
        f2 = lambda t, p=5: "%.*f +/- %.*f" % (p, t[0], p, t[1])
        print("%-9s %3d %-21s %-21s %-17s %-17s" % (r[0], r[1], f2(r[2]), f2(r[3]),
                                                    f2(r[4], 3), f2(r[5], 3)))
    print("  wrote %s" % out)


if __name__ == "__main__":
    print("Block 2 figures:")
    fig_52(); fig_531b(); fig_55(); fig_57(); table_542()
