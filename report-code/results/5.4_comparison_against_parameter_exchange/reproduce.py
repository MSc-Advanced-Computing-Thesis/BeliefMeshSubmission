"""Section 5.4 -- Comparison Against Parameter Exchange.

Regenerates, from artefacts/5.4_comparison_against_parameter_exchange/
five_environments/ (5 environments x 4 mechanisms x 5 seeds = 100 runs):

    Table 5.3   table_5_3_parameter_exchange.csv (+ .txt)
                accuracy and per-node payload per timestep, pooled over the
                25 runs of each mechanism
    Figure 5.7  fig_5_7_exchange_trajectories.png
                cell-space MSE per timestep in field2 (the one environment in
                which gossip finishes below fusion on last-50) and field3 (one in
                which fusion finishes clearly lowest); --fields overrides the pair,
                --fields auto picks the least- and most-converged environments
                (figure_5_7_orderings.txt lists the spread and ordering per field)
    (Appendix D's five-environment figure reuses this section's readout; see
    appendix_D_parameter_exchange_trajectories/reproduce.py)

Each output is written twice. The plain file scores every run against the
offset field it was generated with. The *_as_published file scores every run
against field0, which is how the report's Table 5.3 and Figure 5.7 were
produced; it is retained so the published numbers can be reproduced and the
difference inspected (see exchange_readout.py).

--rerun re-runs the 100 mesh experiments (GPU, ~16 h) through
experiment_parameter_exchange.py.
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
from matplotlib.ticker import LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, FIGURES, TABLES  # noqa: E402
from _shared.arm_names import name as arm_name  # noqa: E402
from _shared.style import WIDTH, DPI, INK, RED, BLUE, GREEN, fmt  # noqa: E402
from exchange_readout import (FIELDS, MODES, N_NODES, collect, ms, offset_trace,  # noqa: E402
                              pooled)

SECTION_ART = ARTEFACTS / "5.4_comparison_against_parameter_exchange" / "five_environments"
SMOOTH = 15
OFFC = "#9aa0a8"
STYLE = {"fusion": (INK, "-"), "gossip_uniform": (RED, "--"),
         "gossip_weighted": (BLUE, "-."), "fedavg_global": (GREEN, ":")}
# panel titles name the environment only; nothing about which mechanism leads
TITLE = {f: f.split("_")[0] for f in FIELDS}
OFF_LIM = (5.0, 35.0)
LEGEND_ORDER = ["fedavg_global", "gossip_uniform", "gossip_weighted", "fusion"]


def table_5_3(C: dict, tab_dir: Path, suffix: str):
    lines = ["TABLE 5.3 -- PARAMETER EXCHANGE ACROSS 5 ENVIRONMENTS x 5 SEEDS "
             "(averaged-NIG readout, last-50 window, n = 25 runs per mechanism)%s" % suffix,
             "%-18s %3s %-21s %-21s %-15s %-15s %14s %14s"
             % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS",
                "tx bytes/node", "rx bytes/node")]
    rows = []
    for m in MODES:
        v = pooled(C, m)
        tx = 1_560_080 if m != "fusion" else None
        rx_per_step = ms([x["comm_step"] for x in v])[0] / N_NODES
        # bytes received per node per timestep; the transmitted belief payload of
        # a 7x7 field of view is 49 cells x 4 parameters x 4 bytes = 784 bytes
        tx = 784 if m == "fusion" else tx
        rows.append((m, len(v), ms([x["whole"] for x in v]), ms([x["last50"] for x in v]),
                     ms([x["cov"] for x in v]), ms([x["ratio"] for x in v]), tx, rx_per_step))
        lines.append("%-18s %3d %-21s %-21s %-15s %-15s %14s %14s"
                     % (arm_name(m), len(v), fmt(rows[-1][2]), fmt(rows[-1][3]), fmt(rows[-1][4], 3),
                        fmt(rows[-1][5], 3), "{:,}".format(tx), "{:,.0f}".format(rx_per_step)))
    b = dict((r[0], r) for r in rows)["fusion"]
    for m in MODES[1:]:
        r = dict((x[0], x) for x in rows)[m]
        lines.append("  Product fusion vs %s: whole-run %+.1f%%, last-50 %+.1f%%"
                     % (arm_name(m), 100 * (b[2][0] - r[2][0]) / r[2][0], 100 * (b[3][0] - r[3][0]) / r[3][0]))
    lines.append("\nPER ENVIRONMENT (5 seeds per cell)")
    for f in FIELDS:
        lines.append("--- %s" % f)
        for m in MODES:
            v = C[(f, m)]
            lines.append("  %-18s whole %s   last-50 %s" % (arm_name(m), fmt(ms([x["whole"] for x in v])),
                                                              fmt(ms([x["last50"] for x in v]))))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / ("table_5_3_parameter_exchange%s.txt" % suffix)).write_text("\n".join(lines) + "\n", encoding="utf8")
    with open(tab_dir / ("table_5_3_parameter_exchange%s.csv" % suffix), "w", encoding="utf8") as fh:
        fh.write("arm,n_runs,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,last50_mse_sd,"
                 "coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd,transmitted_bytes_per_node_step,"
                 "received_bytes_per_node_step\n")
        for m, n, w, l, c, r, tx, rx in rows:
            fh.write("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.1f\n"
                     % (arm_name(m), n, *w, *l, *c, *r, tx, rx))
    print("\n".join(lines[:8]))
    print("  wrote %s" % (tab_dir / ("table_5_3_parameter_exchange%s.csv" % suffix)))


sm = lambda y: np.convolve(y, np.ones(SMOOTH) / SMOOTH, mode="valid")


def trace(C, field, mode):
    return np.nanmean(np.stack([r["trace"] for r in C[(field, mode)]]), axis=0)


def panel(ax, C, field, *, ylab, rlab, xlab):
    o = sm(offset_trace(field))
    axt = ax.twinx()
    axt.plot(np.arange(len(o)) + SMOOTH // 2, o, "--", color=OFFC, lw=0.9, zorder=0)
    axt.set_ylim(*OFF_LIM)
    axt.tick_params(axis="y", labelsize=6.8, labelcolor="#7c828a")
    if rlab:
        axt.set_ylabel("mean |offset| (deg)", fontsize=7.2, color="#7c828a", labelpad=2)
    else:
        axt.set_yticklabels([])
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    for mode in MODES:
        col, ls = STYLE[mode]
        s = sm(trace(C, field, mode))
        ax.plot(np.arange(len(s)) + SMOOTH // 2, s, ls, color=col, lw=1.2, label=arm_name(mode), zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(TITLE[field], fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    if ylab:
        ax.set_ylabel("cell-space MSE", fontsize=8)
    if xlab:
        ax.set_xlabel("timestep", fontsize=8)


def ordered(ax):
    h, l = ax.get_legend_handles_labels()
    by = dict(zip(l, h))
    labs = [arm_name(m) for m in LEGEND_ORDER]
    return [by[x] for x in labs], labs


def last50_spread(C: dict, field: str) -> float:
    """max / min of the four mechanisms' seed-mean last-50 MSE: 1.0 means the
    variants have converged onto one another by the end of the run."""
    v = [ms([x["last50"] for x in C[(field, m)]])[0] for m in MODES]
    return max(v) / min(v)


def ordering(C: dict, field: str) -> list[str]:
    return sorted(MODES, key=lambda m: ms([x["last50"] for x in C[(field, m)]])[0])


def report_orderings(C: dict, tab_dir: Path, suffix: str):
    lines = ["LAST-50 ORDERING BY ENVIRONMENT (seed-mean last-50 MSE, ascending)%s" % suffix]
    for f in FIELDS:
        vals = ["%s %.5f" % (arm_name(m), ms([x["last50"] for x in C[(f, m)]])[0]) for m in ordering(C, f)]
        lines.append("  %-6s spread %.2fx   " % (TITLE[f], last50_spread(C, f)) + " < ".join(vals))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / ("figure_5_7_orderings%s.txt" % suffix)).write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


def choose_pair(C: dict):
    """The environment where the variants converge most (smallest spread) and
    the one where they converge least (largest spread)."""
    sp = {f: last50_spread(C, f) for f in FIELDS}
    return min(sp, key=sp.get), max(sp, key=sp.get)


def figure_5_7(C: dict, fig_dir: Path, suffix: str, fields=None):
    fields = fields or choose_pair(C)
    print("  Figure 5.7 panels: %s (spread %.2fx) and %s (spread %.2fx)"
          % (TITLE[fields[0]], last50_spread(C, fields[0]), TITLE[fields[1]], last50_spread(C, fields[1])))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    fig.subplots_adjust(left=0.082, right=0.918, bottom=0.255, top=0.885, wspace=0.62)
    for ax, f in zip(axes, fields):
        panel(ax, C, f, ylab=True, rlab=True, xlab=True)
    h, l = ordered(axes[0])
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7,
               handlelength=2.0, columnspacing=1.4, bbox_to_anchor=(0.5, 0.035))
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / ("fig_5_7_exchange_trajectories%s.png" % suffix)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % p)


def figure_D_1(C: dict, fig_dir: Path, suffix: str):
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 5.6))
    fig.subplots_adjust(left=0.088, right=0.912, bottom=0.115, top=0.945, wspace=0.60, hspace=0.55)
    flat = axes.ravel()
    for i, f in enumerate(FIELDS):
        panel(flat[i], C, f, ylab=(i % 2 == 0), rlab=(i % 2 == 1), xlab=(i >= 3))
    flat[5].axis("off")
    h, l = ordered(flat[0])
    flat[5].legend(h, l, loc="center", frameon=False, fontsize=7.5, handlelength=2.2, labelspacing=0.9)
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / ("fig_D_1_exchange_trajectories_all%s.png" % suffix)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % p)


DEFAULT_FIELDS = ("field2_seed23", "field3_seed37")


def regenerate(art: Path, fig_dir: Path, tab_dir: Path, with_appendix_d: bool = False, fields=DEFAULT_FIELDS):
    for truth, suffix in (("own", ""), ("field0", "_as_published")):
        print("\n=== readout: truth=%s ===" % truth)
        C = collect(art, truth)
        table_5_3(C, tab_dir, suffix)
        report_orderings(C, tab_dir, suffix)
        figure_5_7(C, fig_dir, suffix, fields)
        if with_appendix_d:
            figure_D_1(C, fig_dir, suffix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--with-appendix-d", action="store_true", help="also write Figure D.1")
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS),
                    help="the two environments of Figure 5.7, comma-separated, or 'auto'")
    ap.add_argument("--rerun", action="store_true", help="re-run the 100 experiments first (GPU, ~16 h)")
    a = ap.parse_args()
    if a.rerun:
        subprocess.run([sys.executable, "-u", str(HERE / "experiment_parameter_exchange.py"),
                        "--root", a.artefacts], check=True, cwd=str(HERE))
    fields = None if a.fields == "auto" else tuple(a.fields.split(","))
    regenerate(Path(a.artefacts), Path(a.figures), Path(a.tables), a.with_appendix_d, fields)
