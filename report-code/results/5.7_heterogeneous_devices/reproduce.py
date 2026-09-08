"""Section 5.7 -- Heterogeneous Devices.

Regenerates, from artefacts/5.7_heterogeneous_devices/:

    Table 5.5   table_5_5_mixed_vs_uniform.csv (+ .txt)
                the mixed-capacity mesh (mixed_capacity/nig_product/) against
                uniform narrow / baseline / wide meshes (uniform_{narrow,
                baseline,wide}/), Product fusion, 5 seeds. All four were run
                by the same script under the same settings; the uniform
                baseline mesh is statistically indistinguishable from the
                Section 5.3.2 Product fusion arm (0.02516 whole-run in both).
    Figure 5.10 fig_5_10_heterogeneity.png
                Naive, Certainty, Product fusion and Average fusion on the
                homogeneous mesh (Section 5.3.2 arms) and on the mixed mesh
                (mixed_capacity/{naive,certainty,nig_product,avgfusion}/).

--rerun re-runs the mixed-capacity arms and the uniform narrow / wide meshes
(GPU, ~6 h) through experiment_mixed_capacity.py; the homogeneous baseline
arms belong to Section 5.3.2.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, FIGURES, TABLES, SEEDS  # noqa: E402
from _shared.artefacts import run_dirs  # noqa: E402
from _shared.averaged_readout import arm_summary  # noqa: E402
from _shared.arm_names import name as arm_name  # noqa: E402
from _shared.calibration_band import draw as cal_band, ensure_visible as cal_ylim  # noqa: E402
from _shared.style import INK, RED, BLUE, ORANGE, DPI, fmt  # noqa: E402

SECTION_ART = ARTEFACTS / "5.7_heterogeneous_devices"
HOMOG_ART = ARTEFACTS / "5.3_belief_aggregation" / "5.3.2_mesh_scale"
ARMS = ["naive", "certainty", "nig_product", "avgfusion"]
COLS = {"nig_product": INK, "naive": RED, "certainty": BLUE, "avgfusion": ORANGE}


def homogeneous(homog: Path) -> dict:
    return {"naive": arm_summary(run_dirs(homog / "aggregation_arms" / "naive_sampled_seed*")),
            "certainty": arm_summary(run_dirs(homog / "aggregation_arms" / "certainty_sampled_seed*")),
            "nig_product": arm_summary(run_dirs(homog / "aggregation_arms" / "nig_product_sampled_seed*")),
            "avgfusion": arm_summary(run_dirs(homog / "average_fusion" / "avg_training_seed*"))}


def heterogeneous(art: Path) -> dict:
    return {a: arm_summary(run_dirs(art / "mixed_capacity" / a / "*")) for a in ARMS}


def table_5_5(art: Path, homog: Path, tab_dir: Path):
    rows = [("Mixed capacity", arm_summary(run_dirs(art / "mixed_capacity" / "nig_product" / "*"))),
            ("Uniform narrow", arm_summary(run_dirs(art / "uniform_narrow" / "*"))),
            ("Uniform baseline", arm_summary(run_dirs(art / "uniform_baseline" / "*"))),
            ("Uniform wide", arm_summary(run_dirs(art / "uniform_wide" / "*")))]
    lines = ["TABLE 5.5 -- MIXED CAPACITY AGAINST UNIFORM MESHES (Product fusion, averaged-NIG readout, last-50 window, 5 seeds)",
             "%-18s %3s %-21s %-21s %-15s %-15s" % ("configuration", "n", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS")]
    for lab, r in rows:
        lines.append("%-18s %3d %-21s %-21s %-15s %-15s"
                     % (lab, r["n"], fmt(r["whole"]), fmt(r["last50"]), fmt(r["cov"], 3), fmt(r["ratio"], 3)))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_5_5_mixed_vs_uniform.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    with open(tab_dir / "table_5_5_mixed_vs_uniform.csv", "w", encoding="utf8") as f:
        f.write("configuration,n_seeds,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,last50_mse_sd,"
                "coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd\n")
        for lab, r in rows:
            f.write("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (lab, r["n"], *r["whole"], *r["last50"], *r["cov"], *r["ratio"]))
    print("\n".join(lines))
    print("  wrote %s" % (tab_dir / "table_5_5_mixed_vs_uniform.csv"))


def figure_5_10(hom: dict, het: dict, fig_dir: Path, tab_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.55))    # wider than the 6.3 in standard, as in the report
    fig.subplots_adjust(left=0.082, right=0.995, bottom=0.145, top=0.885, wspace=0.42)
    w = 0.35
    for ax, key, lab, title in ((axes[0], "whole", "whole-run MSE", "whole-run"),
                                (axes[1], "last50", "last-50 MSE", "last-50"),
                                (axes[2], "cov", "90% coverage", "coverage"),
                                (axes[3], "ratio", "hw : RMS", "hw : RMS")):
        for i, arm in enumerate(ARMS):
            for j, (src, hatch) in enumerate(((hom, ""), (het, "///"))):
                r = src.get(arm)
                if not r:
                    continue
                ax.bar(i + (j - 0.5) * w, r[key][0], w, yerr=r[key][1], capsize=2, color=COLS[arm],
                       alpha=0.95 if j == 0 else 0.45, hatch=hatch, edgecolor="white", linewidth=0.5)
        if key == "cov":
            ax.axhline(0.90, color=RED, ls=":", lw=1.0)
            ax.set_ylim(0, 1.10)
        ax.set_xticks([])
        ax.set_ylabel(lab, fontsize=8, labelpad=2)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(-0.55, len(ARMS) - 0.45)
    cal_band(axes[3], on_top=True); cal_ylim(axes[3])
    axes[0].legend(handles=[Patch(facecolor="#9aa0a8", ec="white", label="homogeneous"),
                            Patch(facecolor="#9aa0a8", ec="white", alpha=0.45, hatch="///", label="heterogeneous")],
                   fontsize=6.4, frameon=False, loc="upper left", handlelength=1.4, labelspacing=0.35)
    fig.legend(handles=[Patch(facecolor=COLS[a], ec="white", label=arm_name(a)) for a in ARMS],
               loc="lower center", ncol=4, frameon=False, fontsize=6.8, handlelength=1.5, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.01))
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / "fig_5_10_heterogeneity.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % p)
    # the numbers behind the figure
    lines = ["FIGURE 5.10 -- aggregation rules on the homogeneous and mixed-capacity meshes (5 seeds)",
             "%-16s %-13s %-21s %-21s %-15s %-15s" % ("arm", "mesh", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS")]
    for arm in ARMS:
        for lab, src in (("homogeneous", hom), ("heterogeneous", het)):
            r = src[arm]
            lines.append("%-16s %-13s %-21s %-21s %-15s %-15s"
                         % (arm_name(arm), lab, fmt(r["whole"]), fmt(r["last50"]), fmt(r["cov"], 3), fmt(r["ratio"], 3)))
        h, e = hom[arm], het[arm]
        lines.append("  heterogeneous vs homogeneous: whole-run %+.0f%%, last-50 %+.0f%%"
                     % (100 * (e["whole"][0] - h["whole"][0]) / h["whole"][0], 100 * (e["last50"][0] - h["last50"][0]) / h["last50"][0]))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "figure_5_10_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


def rerun(seeds: str, art: Path):
    subprocess.run([sys.executable, "-u", str(HERE / "experiment_mixed_capacity.py")], check=True, cwd=str(HERE))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--homogeneous", default=str(HOMOG_ART), help="Section 5.3.2 artefacts (homogeneous arms)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rerun", action="store_true", help="re-run the mixed and uniform experiments first (GPU, ~6 h)")
    a = ap.parse_args()
    if a.rerun:
        rerun(a.seeds, Path(a.artefacts))
    table_5_5(Path(a.artefacts), Path(a.homogeneous), Path(a.tables))
    figure_5_10(homogeneous(Path(a.homogeneous)), heterogeneous(Path(a.artefacts)), Path(a.figures), Path(a.tables))
