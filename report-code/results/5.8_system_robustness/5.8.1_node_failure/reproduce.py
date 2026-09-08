"""Section 5.8.1 -- Node Failure.

Regenerates, from artefacts/5.8_system_robustness/5.8.1_node_failure/node_loss/
(random and clustered node loss at 10-60 % of the mesh plus the 0 % control,
5 seeds each, 55 runs):

    Figure 5.11 fig_5_11_coverage_maps.png   surviving coverage at 40 % loss,
                                             seed 42, random beside clustered
    Figure 5.12 fig_5_12_node_loss.png       dead-zone fraction, surviving-region
                                             MSE difference, coverage and hw:RMS
                                             against the fraction of nodes failed

Figure 5.12's second to fourth panels read two derived CSVs (node_loss_paired_
averaged.csv, node_loss_calibration.csv) that analysis_node_loss_paired.py and
analysis_node_loss_calibration.py compute from the runs; --recompute-analysis
rebuilds them (a few minutes), otherwise the stored CSVs are used.

--rerun re-runs the 55 node-failure experiments (GPU, ~9 h) through
experiment_node_loss.py, then recomputes the analysis CSVs.
"""

from __future__ import annotations

import argparse
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1]
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, FIGURES  # noqa: E402
from _shared.artefacts import load_array, manifest, run_dirs  # noqa: E402
from _shared.calibration_band import draw as cal_band  # noqa: E402
from _shared.style import WIDTH, DPI, INK, RED, GREY, ms  # noqa: E402
from beliefmesh.simulation.assets import ENVIRONMENT_DIR  # noqa: E402

SECTION_ART = ARTEFACTS / "5.8_system_robustness" / "5.8.1_node_failure"


def read_csv(p: Path) -> dict:
    with open(p, encoding="utf8") as f:
        hdr = f.readline().strip().split(",")
        rows = [line.strip().split(",") for line in f if line.strip()]
    return {r[0]: dict(zip(hdr[1:], r[1:])) for r in rows}


def figure_5_12(art: Path, fig_dir: Path):
    paired = read_csv(art / "node_loss_paired_averaged.csv")
    cal = read_csv(art / "node_loss_calibration.csv")
    G = defaultdict(list)
    for d in run_dirs(art / "node_loss" / "*"):
        m = manifest(d)
        G[d.name.rsplit("_seed", 1)[0]].append(m["results"].get("dead_zone_fraction"))
    series = defaultdict(list)
    for cond, dz in G.items():
        kind = "clustered" if "clustered" in cond else "random"
        pct = int(cond.split("_")[-1].replace("pct", ""))
        p = paired.get(cond)
        c = cal.get(cond)
        series[kind].append((pct, ms(dz),
                             (float(p["averaged_mean_diff"]), float(p["averaged_sd_across_seeds"])) if p else None,
                             dict(cov=(float(c["coverage_failed"]), float(c["coverage_failed_sd"]), float(c["coverage_control"])),
                                  hw=(float(c["hw_rms_failed"]), float(c["hw_rms_failed_sd"]), float(c["hw_rms_control"]))) if c else None))
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.45))   # wider than the 6.3 in standard, as in the report
    fig.subplots_adjust(left=0.068, right=0.995, bottom=0.285, top=0.90, wspace=0.46)
    for kind, col, mk in (("random", INK, "o-"), ("clustered", RED, "s--")):
        pts = sorted(series[kind])
        axes[0].errorbar([p[0] for p in pts], [100 * p[1][0] for p in pts], yerr=[100 * p[1][1] for p in pts],
                         fmt=mk, color=col, ms=4, lw=1.3, capsize=2.5, label=kind)
    axes[0].set_xlabel("nodes failed (%)", fontsize=8)
    axes[0].set_ylabel("dead-zone (%)", fontsize=8, labelpad=2)
    axes[0].set_title("cell coverage loss", fontsize=9)
    ax = axes[1]
    for kind, col, mk in (("random", INK, "o-"), ("clustered", RED, "s--")):
        pts = [p for p in sorted(series[kind]) if p[2] is not None]
        ax.errorbar([p[0] for p in pts], [1e3 * p[2][0] for p in pts], yerr=[1e3 * p[2][1] for p in pts],
                    fmt=mk, color=col, ms=4, lw=1.3, capsize=2.5)
    ax.axhline(0.0, color=GREY, ls=":", lw=1.0)
    ax.set_xlabel("nodes failed (%)", fontsize=8)
    ax.set_ylabel(r"MSE diff ($\times 10^{-3}$)", fontsize=8, labelpad=2)
    ax.set_title("surviving performance", fontsize=9)
    for ax, key, ylab, title in ((axes[2], "cov", "90% coverage", "coverage"), (axes[3], "hw", "hw : RMS", "hw : RMS")):
        ctrl = [p[3][key][2] for k in ("random", "clustered") for p in series[k] if p[3]]
        if key == "hw":
            cal_band(ax, on_top=False)
        else:
            ax.axhline(0.90, color=RED, ls=":", lw=1.0)
        ax.axhline(st.mean(ctrl), color=GREY, ls=":", lw=1.0)
        for kind, col, mk in (("random", INK, "o-"), ("clustered", RED, "s--")):
            pts = [p for p in sorted(series[kind]) if p[3] is not None]
            ax.errorbar([p[0] for p in pts], [p[3][key][0] for p in pts], yerr=[p[3][key][1] for p in pts],
                        fmt=mk, color=col, ms=4, lw=1.3, capsize=2.5)
        ax.set_xlabel("nodes failed (%)", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8, labelpad=2)
        ax.set_title(title, fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.legend(handles=[Line2D([], [], color=INK, marker="o", ls="-", ms=4, lw=1.3, label="random loss"),
                        Line2D([], [], color=RED, marker="s", ls="--", ms=4, lw=1.3, label="clustered loss"),
                        Line2D([], [], color=GREY, ls=":", lw=1.0, label="0% control")],
               loc="lower center", ncol=3, frameon=False, fontsize=7, handlelength=2.0, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.015))
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / "fig_5_12_node_loss.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % p)
    for kind in ("random", "clustered"):
        print("  %s: " % kind + ", ".join("%d%%: dead-zone %.1f%%" % (p[0], 100 * p[1][0]) for p in sorted(series[kind])))


def figure_5_11(art: Path, fig_dir: Path):
    centres = np.load(ENVIRONMENT_DIR / "node_centres.npy")
    panels = [("random", "random loss"), ("clustered", "clustered loss")]
    maps, dead, failed = {}, {}, {}
    for kind, _ in panels:
        d = art / "node_loss" / ("nig_product_%s_40pct_seed42" % kind)
        maps[kind] = load_array(d, "coverage_count")
        dead[kind] = int((maps[kind] == 0).sum())
        failed[kind] = set(manifest(d)["failure_ids"])
    LEVELS = list(range(1, 10))
    base = plt.get_cmap("YlGnBu")
    cmap = ListedColormap([base(0.12 + 0.80 * i / (len(LEVELS) - 1)) for i in range(len(LEVELS))])
    cmap.set_bad("#4a4a4a")
    norm = BoundaryNorm(np.arange(len(LEVELS) + 1) - 0.5, len(LEVELS))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.05))
    fig.subplots_adjust(left=0.055, right=0.865, bottom=0.175, top=0.855, wspace=0.13)
    for ax, (kind, lab) in zip(axes, panels):
        idx = np.searchsorted(LEVELS, maps[kind])
        M = np.ma.masked_where(maps[kind] == 0, idx)
        im = ax.imshow(M, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
        alive = [i for i in range(len(centres)) if i not in failed[kind]]
        gone = sorted(failed[kind])
        ax.plot(centres[alive, 0], centres[alive, 1], "o", ms=3.6, mfc="white", mec=INK, mew=0.8, ls="none")
        ax.plot(centres[gone, 0], centres[gone, 1], "X", ms=6.4, mfc=RED, mec="white", mew=0.9, ls="none", zorder=5)
        ax.set_title("%s, %d dead cells" % (lab, dead[kind]), fontsize=9)
        ax.set_xticks([0, 7, 14, 21]); ax.set_yticks([0, 7, 14, 21])
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("cell column", fontsize=8)
    axes[0].set_ylabel("cell row", fontsize=8)
    cax = fig.add_axes([0.878, 0.175, 0.022, 0.68])
    cb = fig.colorbar(im, cax=cax, ticks=np.arange(len(LEVELS)))
    cb.ax.set_yticklabels([str(v) for v in LEVELS])
    cb.set_label("covering surviving nodes", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.legend(handles=[Patch(facecolor="#4a4a4a", label="dead cell (0 nodes)"),
                        Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=0.8, ms=3.6, label="surviving node"),
                        Line2D([], [], marker="X", ls="none", mfc=RED, mec="white", mew=0.9, ms=6.4, label="failed node")],
               loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.46, -0.012))
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / "fig_5_11_coverage_maps.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("  wrote %s   (dead cells: random %d, clustered %d)" % (p, dead["random"], dead["clustered"]))


def recompute_analysis():
    py = sys.executable
    for s in ("analysis_node_loss_paired.py", "analysis_node_loss_calibration.py"):
        print("\n=== %s ===" % s, flush=True)
        subprocess.run([py, "-u", str(HERE / s)], check=True, cwd=str(HERE))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--recompute-analysis", action="store_true", help="rebuild the two derived CSVs from the runs")
    ap.add_argument("--rerun", action="store_true", help="re-run the 55 node-failure experiments first (GPU, ~9 h)")
    a = ap.parse_args()
    if a.rerun:
        subprocess.run([sys.executable, "-u", str(HERE / "experiment_node_loss.py"), "--root",
                        str(Path(a.artefacts) / "node_loss")], check=True, cwd=str(HERE))
    if a.rerun or a.recompute_analysis:
        recompute_analysis()
    figure_5_11(Path(a.artefacts), Path(a.figures))
    figure_5_12(Path(a.artefacts), Path(a.figures))
