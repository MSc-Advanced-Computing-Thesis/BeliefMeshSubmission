"""Section 5.6 -- Uncertainty-Guided Measurement.

Regenerates, from artefacts/5.6_uncertainty_guided_measurement/
{routing_product_fusion, routing_average_fusion}/ (random and
uncertainty-guided wearable routing under each fusion rule, 5 seeds each):

    Figure 5.9  fig_5_9_routing.png
    and the numbers quoted in the prose: table_5_9_routing.csv (+ .txt)

--rerun re-runs the 20 experiments (GPU, ~4 h) through
experiment_routing_product_fusion.py and experiment_routing_average_fusion.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, FIGURES, TABLES, SEEDS  # noqa: E402
from _shared.artefacts import run_dirs  # noqa: E402
from _shared.averaged_readout import arm_summary  # noqa: E402
from _shared.calibration_band import BAND, draw as cal_band  # noqa: E402
from _shared.style import WIDTH, DPI, RED, GREY, fmt  # noqa: E402

SECTION_ART = ARTEFACTS / "5.6_uncertainty_guided_measurement"
PROD_COLOR, AVG_COLOR = "#0F6E56", "#E6821B"     # as in the fusion-rule illustration
ARMS = [  # (rule, policy, sub-root, run prefix, colour, hatch)
    ("Product fusion", "random", "routing_product_fusion", "random", PROD_COLOR, ""),
    ("Product fusion", "guided", "routing_product_fusion", "uncertainty_guided", PROD_COLOR, "///"),
    ("Average fusion", "random", "routing_average_fusion", "random", AVG_COLOR, ""),
    ("Average fusion", "guided", "routing_average_fusion", "uncertainty_guided", AVG_COLOR, "///"),
]
X = np.array([0.0, 0.82, 2.38, 3.20])


def collect(art: Path):
    A = []
    for rule, pol, sub, prefix, col, hat in ARMS:
        r = arm_summary(run_dirs(art / sub / (prefix + "_seed*")))
        if r is None:
            raise SystemExit("no runs for %s/%s" % (sub, prefix))
        A.append((rule, pol, col, hat, r))
    return A


def figure_5_9(A, fig_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9))
    fig.subplots_adjust(left=0.098, right=0.995, bottom=0.30, top=0.86, wspace=0.44)
    ax = axes[0]
    GC = np.array([0.0, 1.62])
    PAIR_OFF, BAR_OFF, BW = 0.40, 0.115, 0.21
    for gi, (rule_i, pol_i) in enumerate(((0, 1), (2, 3))):
        for mi, key in enumerate(("whole", "last50")):
            pc = GC[gi] + (mi - 0.5) * 2 * PAIR_OFF
            for bi, ai in enumerate((rule_i, pol_i)):
                a = A[ai]
                ax.bar(pc + (bi - 0.5) * 2 * BAR_OFF, a[4][key][0], BW, yerr=a[4][key][1], capsize=2.2,
                       color=a[2], hatch=a[3], edgecolor="white", lw=0.5)
    ax.set_ylabel("cell-space MSE", fontsize=8)
    ax.set_title("accuracy", fontsize=9, pad=11)
    ax.set_ylim(0, max(a[4]["whole"][0] + a[4]["whole"][1] for a in A) * 1.16)
    ax.set_xlim(GC[0] - PAIR_OFF - 0.35, GC[-1] + PAIR_OFF + 0.35)
    ax.set_xticks([])
    for gi in range(2):
        for mi, lab in enumerate(("whole-run", "last-50")):
            ax.annotate(lab, xy=(GC[gi] + (mi - 0.5) * 2 * PAIR_OFF, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=5.6)
    for ax_, key, lab, title in ((axes[1], "cov", "90% coverage", "coverage"),
                                 (axes[2], "ratio", "hw : RMS", "hw : RMS")):
        ax_.bar(X, [a[4][key][0] for a in A], 0.62, yerr=[a[4][key][1] for a in A], capsize=2.5,
                color=[a[2] for a in A], hatch=[a[3] for a in A], edgecolor="white", lw=0.5)
        if key == "cov":
            ax_.axhline(0.90, color=RED, ls=":", lw=1.0)
            ax_.set_ylim(0, 1.12)
        else:
            cal_band(ax_, on_top=True)
            ax_.set_ylim(0, max(a[4]["ratio"][0] + a[4]["ratio"][1] for a in A) * 1.10)
        ax_.set_ylabel(lab, fontsize=8)
        ax_.set_title(title, fontsize=9, pad=11)
    for ax_ in axes[1:]:
        ax_.set_xticks([X[:2].mean(), X[2:].mean()])
        ax_.set_xticklabels(["Product\nfusion", "Average\nfusion"], fontsize=7.4)
        ax_.tick_params(labelsize=8, length=0)
        ax_.set_xlim(X[0] - 0.62, X[-1] + 0.62)
    for gi, rule in enumerate(("Product\nfusion", "Average\nfusion")):
        axes[0].annotate(rule, xy=(GC[gi], 0), xycoords=("data", "axes fraction"), xytext=(0, -14),
                         textcoords="offset points", ha="center", va="top", fontsize=7.4)
    fig.legend(handles=[Patch(facecolor=PROD_COLOR, ec="white", label="Product fusion"),
                        Patch(facecolor=AVG_COLOR, ec="white", label="Average fusion"),
                        Patch(facecolor=GREY, ec="white", label="random routing"),
                        Patch(facecolor=GREY, ec="white", hatch="///", label="uncertainty-guided routing"),
                        Line2D([], [], color=RED, ls=":", lw=1.0, label="90% nominal coverage (centre panel)"),
                        Patch(facecolor=RED, alpha=0.16, ec="none",
                              label="empirical 90%% hw:RMS band %.2f-%.2f (right panel)" % BAND)],
               loc="lower center", ncol=3, fontsize=6.2, frameon=False, handlelength=1.5, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.005))
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / "fig_5_9_routing.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % p)


def table(A, tab_dir: Path):
    lines = ["MATCHED ROUTING UNDER BOTH FUSION RULES (cell space, averaged-NIG readout, last-50 window, 5 seeds, mean +/- sd)",
             "%-16s %-8s %3s %-21s %-21s %-15s %-15s" % ("fusion rule", "routing", "n", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS")]
    for rule, pol, col, hat, r in A:
        lines.append("%-16s %-8s %3d %-21s %-21s %-15s %-15s"
                     % (rule, pol, r["n"], fmt(r["whole"]), fmt(r["last50"]), fmt(r["cov"], 3), fmt(r["ratio"], 3)))
    for i in (0, 2):
        r, g = A[i][4], A[i + 1][4]
        lines.append("  %s: guided vs random  whole-run %+.1f%%  last-50 %+.1f%%  coverage %.3f -> %.3f  hw:RMS %.3f -> %.3f"
                     % (A[i][0], 100 * (g["whole"][0] - r["whole"][0]) / r["whole"][0],
                        100 * (g["last50"][0] - r["last50"][0]) / r["last50"][0],
                        r["cov"][0], g["cov"][0], r["ratio"][0], g["ratio"][0]))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_5_9_routing.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    with open(tab_dir / "table_5_9_routing.csv", "w", encoding="utf8") as f:
        f.write("fusion_rule,routing,n_seeds,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,last50_mse_sd,"
                "coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd\n")
        for rule, pol, col, hat, r in A:
            f.write("%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n"
                    % (rule, pol, r["n"], *r["whole"], *r["last50"], *r["cov"], *r["ratio"]))
    print("\n".join(lines))
    print("  wrote %s" % (tab_dir / "table_5_9_routing.csv"))


def rerun(seeds: str, art: Path):
    py = sys.executable
    subprocess.run([py, "-u", str(HERE / "experiment_routing_product_fusion.py"), "--seeds", seeds,
                    "--root", str(art / "routing_product_fusion")], check=True, cwd=str(HERE))
    subprocess.run([py, "-u", str(HERE / "experiment_routing_average_fusion.py"), "--seeds", seeds,
                    "--root", str(art / "routing_average_fusion")], check=True, cwd=str(HERE))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rerun", action="store_true", help="re-run the 20 experiments first (GPU, ~4 h)")
    a = ap.parse_args()
    if a.rerun:
        rerun(a.seeds, Path(a.artefacts))
    A = collect(Path(a.artefacts))
    table(A, Path(a.tables))
    figure_5_9(A, Path(a.figures))
