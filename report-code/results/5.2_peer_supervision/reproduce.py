"""Section 5.2 -- Peer Supervision as a Training Signal.

Regenerates, from artefacts/5.2_peer_supervision/two_node_offset_world/
(frozen, direct, student with mode target, student with sampled target;
5 seeds each on the two-node offset world):

    Figure 5.4  fig_5_4_peer_supervision.png
    and the numbers quoted in the prose: table_5_4_peer_supervision.csv (+ .txt)

--rerun re-runs the 20 two-node experiments (GPU, ~1 h) through
experiment_peer_supervision.py --world offset.
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

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, FIGURES, TABLES, SEEDS  # noqa: E402
from _shared.artefacts import load_array, manifest  # noqa: E402
from _shared.calibration_band import draw as cal_band, ensure_visible as cal_ylim  # noqa: E402
from _shared.style import WIDTH, DPI, INK, RED, BLUE, GREEN, PURPLE, ms  # noqa: E402
from beliefmesh.node.mesh import fov_cells  # noqa: E402
from beliefmesh.simulation.offset_fields import build_dynamic_offset_field  # noqa: E402

SECTION_ART = ARTEFACTS / "5.2_peer_supervision" / "two_node_offset_world"
STUDENT_CENTRE, FOV, GRID_G, T_STEPS = (6, 3), 7, 22, 390
ARMS = [("frozen", "frozen", GREEN), ("direct", "direct", BLUE),
        ("student", "student (mode target)", PURPLE), ("student_sampled", "student (sampled target)", RED)]
CAL_LABELS = {"frozen": "frozen", "direct": "direct", "student": "mode\ntarget", "student_sampled": "sampled\ntarget"}


def offset_trace():
    """Mean |offset| in degrees over the STUDENT's field of view, per timestep."""
    F = build_dynamic_offset_field(GRID_G, T_STEPS)
    cells = np.array(fov_cells(*STUDENT_CENTRE, FOV, GRID_G))
    m = np.zeros((GRID_G, GRID_G), bool)
    m[cells[:, 0], cells[:, 1]] = True
    return np.abs(F[:, m]).mean(axis=1)


def collect(art: Path, seeds):
    """Per arm: student MSE trajectories (node index 1) and manifest results."""
    out = {}
    for tag, lab, col in ARMS:
        traj, res = [], []
        for s in seeds:
            d = art / ("%s_seed%d" % (tag, s))
            if not (d / "manifest.yaml").exists():
                continue
            traj.append(load_array(d, "node_mse_steps")[:, 1])
            res.append(manifest(d)["results"])
        out[tag] = (np.array(traj), res)
    return out


def figure_5_4(D: dict, fig_dir: Path):
    fig = plt.figure(figsize=(WIDTH, 4.25))
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.935, bottom=0.145, top=0.955, hspace=0.52, wspace=0.30)
    ax = fig.add_subplot(gs[0, :])
    axt = ax.twinx()
    w = 15
    sm = lambda y: np.convolve(y, np.ones(w) / w, mode="valid")
    tr = sm(offset_trace())
    axt.plot(np.arange(len(tr)) + w // 2, tr, "--", color="#9aa0a8", lw=0.9, zorder=0)
    axt.set_ylabel("mean |offset| over FOV (deg)", fontsize=7.5, color="#7c828a")
    axt.tick_params(axis="y", labelsize=7.5, labelcolor="#7c828a")
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    for tag, lab, col in ARMS:
        A = D[tag][0]
        if not len(A):
            continue
        mu = sm(A.mean(0))
        ax.plot(np.arange(len(mu)) + w // 2, mu, "-", color=col, lw=1.25, label=lab, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("timestep", fontsize=8)
    ax.set_ylabel("student cell-space MSE", fontsize=8)
    ax.set_title("trajectory against the moving offset field", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left", ncol=2)

    cov, rat, labs, cols = [], [], [], []
    for tag, lab, col in ARMS:
        cs, rs = [], []
        for r in D[tag][1]:
            c = r.get("student_coverage_90") or {}
            if c.get("empirical") is None:
                continue
            cs.append(c["empirical"])
            rs.append(c["mean_half_width_deg"] / (np.sqrt(r["last50_mse_cell_space"]["student"]) * 180))
        if cs:
            cov.append(ms(cs)); rat.append(ms(rs)); labs.append(CAL_LABELS[tag]); cols.append(col)
    x = np.arange(len(labs))
    for col_i, (vals, ref, ylab, title) in enumerate(((cov, 0.90, "90% coverage", "coverage"),
                                                       (rat, 1.00, "hw : RMS", "hw : RMS"))):
        a = fig.add_subplot(gs[1, col_i])
        a.bar(x, [v[0] for v in vals], 0.66, yerr=[v[1] for v in vals], capsize=3, color=cols, edgecolor="white", lw=0.6)
        if abs(ref - 1.0) < 1e-9:
            cal_band(a); cal_ylim(a)
        else:
            a.axhline(ref, color=RED, ls=":", lw=1.0)
        a.axvline(0.5, color="#c9ced6", lw=0.9, zorder=0)
        a.axvline(1.5, color="#c9ced6", lw=0.9, zorder=0)
        a.set_xticks(x)
        a.set_xticklabels(labs, fontsize=6.4)
        a.set_ylabel(ylab, fontsize=8)
        a.set_title(title, fontsize=9)
        a.tick_params(labelsize=8, pad=1.5)
        a.set_xlim(-0.6, len(x) - 0.4)
        tf = a.get_xaxis_transform()
        yb, yt = -0.30, -0.255
        a.plot([1.68, 3.32], [yb, yb], color=INK, lw=0.8, clip_on=False, transform=tf)
        a.plot([1.68, 1.68], [yb, yt], color=INK, lw=0.8, clip_on=False, transform=tf)
        a.plot([3.32, 3.32], [yb, yt], color=INK, lw=0.8, clip_on=False, transform=tf)
        a.text(2.5, yb - 0.04, "student", ha="center", va="top", fontsize=7, transform=tf, clip_on=False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / "fig_5_4_peer_supervision.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % p)


def table(D: dict, tab_dir: Path):
    rows = []
    for tag, lab, col in ARMS:
        res = D[tag][1]
        if not res:
            continue
        last = [r["last50_mse_cell_space"]["student"] for r in res]
        cov = [r["student_coverage_90"]["empirical"] for r in res]
        rat = [r["student_coverage_90"]["mean_half_width_deg"] / (np.sqrt(r["last50_mse_cell_space"]["student"]) * 180) for r in res]
        by = {}
        for k in ("supervised", "unsupervised", "shared", "exclusive"):
            v = [r.get("last50_mse_by_region", {}).get(k) for r in res]
            if all(x is not None for x in v):
                by[k] = ms(v)
        rows.append((tag, lab, ms(last), ms(cov), ms(rat), by))
    lines = ["SECTION 5.2 -- student node on the two-node offset world (last-50 window, %d seeds)" % len(D["direct"][1]),
             "%-26s %-21s %-15s %-15s" % ("supervision", "last-50 FOV MSE", "90% coverage", "hw : RMS")]
    for tag, lab, l, c, r, by in rows:
        lines.append("%-26s %-21s %-15s %-15s" % (lab, "%.5f +/- %.5f" % l, "%.3f +/- %.3f" % c, "%.3f +/- %.3f" % r))
    R = dict((t, l) for t, _, l, _, _, _ in rows)
    if all(k in R for k in ("frozen", "direct", "student", "student_sampled")):
        gap = R["frozen"][0] - R["direct"][0]
        for t in ("student", "student_sampled"):
            lines.append("  %s closes %.1f%% of the frozen-to-direct gap" % (t, 100 * (R["frozen"][0] - R[t][0]) / gap))
    for tag, lab, l, c, r, by in rows:
        if by:
            lines.append("  %s by region: " % lab + ", ".join("%s %.5f" % (k, v[0]) for k, v in by.items()))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_5_4_peer_supervision.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    with open(tab_dir / "table_5_4_peer_supervision.csv", "w", encoding="utf8") as f:
        f.write("supervision,last50_fov_mse_mean,last50_fov_mse_sd,coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd\n")
        for tag, lab, l, c, r, by in rows:
            f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (lab, *l, *c, *r))
    print("\n".join(lines))
    print("  wrote %s" % (tab_dir / "table_5_4_peer_supervision.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rerun", action="store_true", help="re-run the 20 two-node experiments first (GPU, ~1 h)")
    a = ap.parse_args()
    if a.rerun:
        subprocess.run([sys.executable, "-u", str(HERE / "experiment_peer_supervision.py"), "--world", "offset",
                        "--arms", "frozen,direct,student,student_sampled"], check=True, cwd=str(HERE))
    D = collect(Path(a.artefacts), [int(s) for s in a.seeds.split(",")])
    table(D, Path(a.tables))
    figure_5_4(D, Path(a.figures))
