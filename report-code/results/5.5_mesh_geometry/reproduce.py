"""Section 5.5 -- Mesh Geometry.

Regenerates, from the stored artefacts:

    Figure 5.8  fig_5_8_mesh_density.png
                accuracy, coverage and hw:RMS against mean per-cell coverage
                for the 4x4, 5x5, 6x6 and 10x10 meshes
                (artefacts/5.5_mesh_geometry/mesh_density_sweep/, 5 seeds each)
    Table 5.4   table_5_4_hop_calibration.csv (+ .txt)
                hw:RMS, half-width and RMS error by hop distance from the
                wearables, on the reference 6x6 mesh (the Section 5.3.2 Product
                fusion runs, artefacts/5.3_belief_aggregation/5.3.2_mesh_scale/
                aggregation_arms/nig_product_sampled_seed*)

--rerun re-runs the 20 mesh-density experiments (GPU, ~4 h) through
experiment_mesh_density.py --profile chapter5. The hop table's runs belong to
Section 5.3.2 and are re-run from there.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, NullLocator

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, FIGURES, TABLES, SEEDS  # noqa: E402
from _shared.artefacts import load_array, manifest, run_dirs  # noqa: E402
from _shared.averaged_readout import load_run, run_metrics  # noqa: E402
from _shared.calibration_band import draw as cal_band, ensure_visible as cal_ylim  # noqa: E402
from _shared.style import WIDTH, DPI, INK, RED, BLUE, GREY, GREEN, ms, save  # noqa: E402
from _shared.truth import LAST  # noqa: E402

SECTION_ART = ARTEFACTS / "5.5_mesh_geometry" / "mesh_density_sweep"
HOP_RUNS = ARTEFACTS / "5.3_belief_aggregation" / "5.3.2_mesh_scale" / "aggregation_arms"
REFERENCE_COVERAGE = 3.64      # the 6x6 mesh used throughout Chapter 5


def figure_5_8(art: Path, fig_dir: Path):
    G = defaultdict(list)
    for d in run_dirs(art / "*"):
        m = manifest(d)
        G[d.name.rsplit("_seed", 1)[0]].append((run_metrics(d, LAST), m))
    pts = sorted(((v[0][1]["mean_cell_coverage"], v) for v in G.values()), key=lambda t: t[0])
    x = [p[0] for p in pts]
    ref = min(x, key=lambda v: abs(v - REFERENCE_COVERAGE))

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.45))
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.235, top=0.845, wspace=0.42)
    ax = axes[0]
    for key, lab, col, mk in (("whole", "whole-run", INK, "o-"), ("last50", "last-50", GREEN, "s--")):
        mu = [ms([r[key] for r, _ in v])[0] for _, v in pts]
        sd = [ms([r[key] for r, _ in v])[1] for _, v in pts]
        ax.errorbar(x, mu, yerr=sd, fmt=mk, color=col, ms=4, lw=1.3, capsize=2.5, label=lab)
    ax.set_ylabel("cell-space MSE", fontsize=8)
    ax.set_title("accuracy", fontsize=9, pad=11)
    ax.legend(fontsize=6.6, frameon=False)
    for ax_, key, lab, col, ref_line, title in (
            (axes[1], "cov", "90% coverage", BLUE, 0.90, "coverage"),
            (axes[2], "ratio", "hw : RMS", GREEN, None, "hw : RMS")):
        mu = [ms([r[key] for r, _ in v])[0] for _, v in pts]
        sd = [ms([r[key] for r, _ in v])[1] for _, v in pts]
        ax_.errorbar(x, mu, yerr=sd, fmt="o-", color=col, ms=4, lw=1.3, capsize=2.5)
        if ref_line is None:
            cal_band(ax_); cal_ylim(ax_)
        else:
            ax_.axhline(ref_line, color=RED, ls=":", lw=1.0)
        ax_.set_ylabel(lab, fontsize=8)
        ax_.set_title(title, fontsize=9, pad=11)
    for ax_ in axes:
        ax_.set_xscale("log")
        ax_.xaxis.set_minor_locator(NullLocator())
        ax_.xaxis.set_minor_formatter(NullFormatter())
        ax_.set_xticks(x)
        ax_.set_xticklabels(["%.2f" % v for v in x], fontsize=7)
        ax_.set_xlabel("mean per-cell coverage", fontsize=8)
        ax_.tick_params(labelsize=8)
        ax_.axvline(ref, color=GREY, ls="-", lw=0.9, alpha=0.55, zorder=0)
        ax_.annotate("ch.5 config", xy=(ref, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2.5),
                     textcoords="offset points", fontsize=6.2, color=GREY, va="bottom", ha="center")
    save(fig, "fig_5_8_mesh_density", fig_dir)
    print("  mean per-cell coverage -> whole-run MSE: " + ", ".join(
        "%.2f: %.5f" % (p[0], ms([r["whole"] for r, _ in p[1]])[0]) for p in pts))


def table_5_4(hop_root: Path, tab_dir: Path):
    """hw:RMS by hop under the averaged-NIG readout, last-50 window, 5 seeds."""
    HOPS = [0, 1, 2, 3]
    acc = {h: [] for h in HOPS}
    cells = {h: [] for h in HOPS}
    for d in run_dirs(hop_root / "nig_product_sampled_seed*"):
        seed = int(manifest(d)["env_seed"])
        R = load_run(d, seed)
        hop = load_array(d, "cell_hop_steps")
        e2, hw = R["averaged"]
        T = R["T"]
        sl = slice(T - LAST, T)
        for h in HOPS:
            ok = np.isfinite(e2[sl]) & np.isfinite(hw[sl]) & (hop[sl] == h)
            if ok.sum() < 8:
                continue
            m = float(np.mean(e2[sl][ok]))
            hwd, rms = float(np.mean(hw[sl][ok])) * 180, float(np.sqrt(m)) * 180
            acc[h].append((hwd / rms, hwd, rms))
            cells[h].append(ok.sum() / LAST)
    lines = ["TABLE 5.4 -- CALIBRATION BY HOP FROM GROUND TRUTH (6x6 mesh, 3 wearables, averaged-NIG readout, last-50 window, 5 seeds)",
             "%-4s %10s %-17s %-17s %-17s %6s" % ("hop", "cells/step", "hw : RMS", "half-width (deg)", "RMS error (deg)", "seeds")]
    rows = []
    for h in HOPS:
        if not acc[h]:
            continue
        r = ms([a[0] for a in acc[h]]); w = ms([a[1] for a in acc[h]]); e = ms([a[2] for a in acc[h]])
        rows.append((h, np.mean(cells[h]), r, w, e, len(acc[h])))
        lines.append("%-4d %10.0f %-17s %-17s %-17s %6d"
                     % (h, np.mean(cells[h]), "%.3f +/- %.3f" % r, "%.2f +/- %.2f" % w, "%.2f +/- %.2f" % e, len(acc[h])))
    lines.append("  (hops with fewer than five contributing seeds are shown for completeness; the report tabulates hops 0-2)")
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_5_4_hop_calibration.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    with open(tab_dir / "table_5_4_hop_calibration.csv", "w", encoding="utf8") as f:
        f.write("hop,cells_per_step,hw_rms_mean,hw_rms_sd,half_width_deg_mean,half_width_deg_sd,rms_error_deg_mean,rms_error_deg_sd,n_seeds\n")
        for h, c, r, w, e, n in rows:
            f.write("%d,%.1f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%d\n" % (h, c, *r, *w, *e, n))
    print("\n".join(lines))
    print("  wrote %s" % (tab_dir / "table_5_4_hop_calibration.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART), help="mesh-density run root")
    ap.add_argument("--hop-runs", default=str(HOP_RUNS), help="6x6 Product fusion runs for Table 5.4")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rerun", action="store_true", help="re-run the mesh-density sweep first (GPU, ~4 h)")
    a = ap.parse_args()
    if a.rerun:
        subprocess.run([sys.executable, "-u", str(HERE / "experiment_mesh_density.py"), "--profile", "chapter5",
                        "--seeds", a.seeds, "--root", a.artefacts], check=True, cwd=str(HERE))
    figure_5_8(Path(a.artefacts), Path(a.figures))
    table_5_4(Path(a.hop_runs), Path(a.tables))
