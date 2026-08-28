# Chapter 5 regeneration figures. Built from the dumped runs/chapter5_v2
# artefacts only -- no model, no re-evaluation.
#
# PNG only, 6.3 in wide (LaTeX text width), dpi 200.
#
# Run: python -u experiments/section5_2/make_chapter5_figures.py

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
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
import yaml
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_estimators import LAST, load

FIGS = Path("figures")
WIDTH, DPI = 6.3, 200
INK, RED, BLUE, GREY = "#22252a", "#c1121f", "#1f5fc1", "#8a8f98"
GREEN = "#1b7a3e"


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    png = FIGS / (name + ".png")
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % png)


def run_metrics(d: Path):
    """argmax-convention metrics over the last-50 window -- Chapter 5's
    reported convention."""
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
    rms = np.sqrt(m) * 180
    return dict(whole=float(np.nanmean(mse)), last50=m,
                cov=float((e[k] <= hw).mean()), ratio=float(np.mean(hw) * 180 / rms))


def collect(pattern, keyfn):
    G = defaultdict(list)
    for p in sorted(glob.glob(pattern)):
        d = Path(os.path.dirname(p))
        m = yaml.safe_load(open(p))
        G[keyfn(d, m)].append((run_metrics(d), m))
    return G


def ms(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def fig_density():
    """5.4.1: everything against MEAN PER-CELL COVERAGE, the swept axis."""
    G = collect("runs/chapter5_v2/s5_4_1_overlap/*/manifest.yaml",
                lambda d, m: d.name.rsplit("_seed", 1)[0])
    pts = sorted(((v[0][1]["mean_cell_coverage"], v) for v in G.values()),
                 key=lambda t: t[0])
    x = [p[0] for p in pts]
    nodes = [p[1][0][1]["n_nodes"] for p in pts]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35))
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.22, top=0.90, wspace=0.42)
    for ax, key, lab, col in ((axes[0], "whole", "whole-run MSE", INK),
                              (axes[1], "cov", "90% coverage", BLUE),
                              (axes[2], "ratio", "half-width : RMS", GREEN)):
        mu = [ms([r[key] for r, _ in v])[0] for _, v in pts]
        sd = [ms([r[key] for r, _ in v])[1] for _, v in pts]
        ax.errorbar(x, mu, yerr=sd, fmt="o-", color=col, ms=4, lw=1.3, capsize=2.5)
        ax.set_xscale("log")
        # the log locator's minor ticks overprint the four explicit labels
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(x)
        ax.set_xticklabels(["%.2f" % v for v in x], fontsize=7)
        ax.set_xlabel("mean per-cell coverage", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=8)
    axes[1].axhline(0.90, color=RED, ls=":", lw=1.0)
    axes[2].axhline(1.0, color=RED, ls=":", lw=1.0)
    axes[2].text(x[0], 1.02, "calibrated", fontsize=6.5, color=RED, va="bottom")
    axes[0].set_title("accuracy", fontsize=9)
    axes[1].set_title("coverage", fontsize=9)
    axes[2].set_title("interval vs error", fontsize=9)
    save(fig, "ch5_5_4_1_overlap_density")


def fig_heterogeneity():
    """5.6: the arms separate under heterogeneity but not under homogeneity."""
    het = collect("runs/chapter5_v2/s5_6_het/*/manifest.yaml",
                  lambda d, m: d.name.rsplit("_seed", 1)[0].replace("het_", ""))
    hom = collect("runs/chapter5_v2/s531_b2/*/manifest.yaml",
                  lambda d, m: d.name.rsplit("_seed", 1)[0])
    hom = {k.replace("_sampled", ""): v for k, v in hom.items()
           if k.endswith("_sampled")}
    arms = ["nig_product", "naive", "certainty"]
    cols = {"nig_product": INK, "naive": RED, "certainty": BLUE}

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.26, top=0.88, wspace=0.40)
    w = 0.35
    # last-50 FIRST: it carries the argument -- nig_product is flat across the
    # homogeneous and heterogeneous conditions while the other arms degrade.
    for ax, key, lab in ((axes[0], "last50", "last-50 MSE"),
                         (axes[1], "whole", "whole-run MSE"),
                         (axes[2], "ratio", "half-width : RMS")):
        for i, arm in enumerate(arms):
            for j, (src, hatch, tag) in enumerate(((hom, "", "homog"),
                                                   (het, "///", "heterog"))):
                if arm not in src:
                    continue
                mu, sd = ms([r[key] for r, _ in src[arm]])
                ax.bar(i + (j - 0.5) * w, mu, w, yerr=sd, capsize=2,
                       color=cols[arm], alpha=0.95 if j == 0 else 0.45,
                       hatch=hatch, edgecolor="white", linewidth=0.5,
                       label=tag if i == 0 else None)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(["nig_prod", "naive", "certainty"], fontsize=7.5)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=8)
    axes[2].axhline(1.0, color=RED, ls=":", lw=1.0)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    axes[0].set_title("steady state", fontsize=9)
    axes[1].set_title("accuracy", fontsize=9)
    axes[2].set_title("interval vs error", fontsize=9)
    save(fig, "ch5_5_6_heterogeneity")


if __name__ == "__main__":
    print("Chapter 5 figures:")
    fig_density()
    fig_heterogeneity()
