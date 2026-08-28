# Section 5.2 figures and consolidated summary, built from the dumped
# runs/section5_2 artefacts only. No model, no re-evaluation.
#
# PNG only, 6.3 in wide, dpi 200 (Christian's convention).
#
# Run: python -u experiments/section5_2/make_figures.py

from __future__ import annotations

import glob
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

R = Path("runs/section5_2")
FIGS = Path("figures")
WIDTH, DPI = 6.3, 200
ARMS = ["frozen", "student", "direct"]
COL = {"frozen": "#8a8f98", "student": "#c1121f", "direct": "#1f5fc1"}
SEEDS = [42, 1042, 2042, 3042, 4042]
STUDENT = 1
LAST_N = 50


def traj(world: str, cond: str, arm: str) -> np.ndarray:
    """(n_seeds, T) student-node cell-space MSE per timestep."""
    out = []
    for s in SEEDS:
        f = R / world / cond / f"{arm}_seed{s}" / "node_mse_steps.npy"
        if f.exists():
            out.append(np.load(f)[:, STUDENT])
    return np.array(out)


def smooth(y, w=15):
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def collect(world: str):
    rows = []
    for p in glob.glob(str(R / world / "*" / "*" / "manifest.yaml")):
        m = yaml.safe_load(open(p))
        r = m["results"]
        rows.append({
            "cond": Path(p).parents[1].name, "arm": m["arm"], "seed": m["seed"],
            "whole": r["whole_run_mse_cell_space"]["student"],
            "last50": r["last50_mse_cell_space"]["student"],
            "anchor_whole": r["whole_run_mse_cell_space"]["anchor"],
            "cert_r": r["student_cert_mse_r_per_timestep"]["mean"],
            "cov90": (r["student_coverage_90"] or {}).get("empirical", float("nan")),
            "epi_own": r["uncertainty_source"]["r_student_epi_vs_own_error"],
            "epi_anchor": r["uncertainty_source"]["r_student_epi_vs_anchor_epi"],
        })
    return rows


def agg(rows, cond, arm, key):
    v = [r[key] for r in rows if r["cond"] == cond and r["arm"] == arm
         and r[key] is not None and np.isfinite(r[key])]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else (np.nan, np.nan, 0)


def panel_traj(ax, world, cond, title):
    for arm in ARMS:
        t = traj(world, cond, arm)
        if not len(t):
            continue
        m = smooth(t.mean(0))
        sd = smooth(t.std(0, ddof=1))
        x = np.arange(len(m)) + 7
        ax.plot(x, m, "-", color=COL[arm], lw=1.2, label=arm)
        ax.fill_between(x, np.maximum(m - sd, 1e-5), m + sd,
                        color=COL[arm], alpha=0.16, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("timestep", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)


def main():
    off = collect("offset")
    col = collect("colour")

    # ── figure: trajectories + coverage ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35))
    fig.subplots_adjust(left=0.082, right=0.995, bottom=0.235, top=0.885,
                        wspace=0.34)
    panel_traj(axes[0], "offset", "offset", "Offset world")
    axes[0].set_ylabel("student cell-space MSE", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False)
    panel_traj(axes[1], "colour", "v0.0", "Colour world (blue)")

    ax = axes[2]
    order = ["none", "v0.5", "v1.0", "v0.0"]
    xs = np.arange(len(order))
    for arm in ("student", "direct"):
        m = [agg(col, c, arm, "cov90")[0] for c in order]
        sd = [agg(col, c, arm, "cov90")[1] for c in order]
        ax.errorbar(xs, m, yerr=sd, fmt="o-", color=COL[arm], ms=3.5, lw=1.2,
                    capsize=2, label=arm)
    ax.axhline(0.90, color="#22252a", ls=":", lw=1.0, label="nominal")
    ax.set_xticks(xs)
    ax.set_xticklabels(["none", "purple", "red", "blue"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("90% interval coverage", fontsize=8)
    ax.set_xlabel("colour condition", fontsize=8)
    ax.set_title("Student calibration", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    FIGS.mkdir(parents=True, exist_ok=True)
    png = FIGS / "section5_2_peer_supervision.png"
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    print("  wrote %s" % png)

    # ── consolidated summary ──────────────────────────────────────────────
    out = {"convention": "cell space: mean over (step, cell in node FOV) of "
                         "that node's squared wrapped error; 5 seeds "
                         "(42/1042/2042/3042/4042), mean +/- sd",
           "note_v1_5_degenerate": "colour condition v1.5 is the IDENTITY "
                                   "transform (apply_filter_from_env_value "
                                   "gives r_w=g_w=b_w=1.0 at v=1.5), so it "
                                   "duplicates 'none'. Four distinct colour "
                                   "conditions, not five.",
           "worlds": {}}
    for world, rows, conds in (("offset", off, ["offset"]),
                               ("colour", col, ["none", "v0.5", "v1.0", "v0.0", "v1.5"])):
        w = {}
        for c in conds:
            e = {}
            for arm in ARMS:
                e[arm] = {k: dict(zip(("mean", "sd", "n"), agg(rows, c, arm, k)))
                          for k in ("whole", "last50", "cert_r", "cov90",
                                    "epi_own", "epi_anchor")}
            f_, s_, d_ = (e["frozen"]["whole"]["mean"], e["student"]["whole"]["mean"],
                          e["direct"]["whole"]["mean"])
            e["gap_closed_whole_pct"] = float(100 * (f_ - s_) / (f_ - d_))
            fl, sl, dl = (e["frozen"]["last50"]["mean"], e["student"]["last50"]["mean"],
                          e["direct"]["last50"]["mean"])
            e["gap_closed_last50_pct"] = float(100 * (fl - sl) / (fl - dl))
            w[c] = e
        out["worlds"][world] = w
    with open(R / "summary.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    print("  wrote %s" % (R / "summary.yaml"))

    for world in ("offset", "colour"):
        print("\n=== %s ===" % world)
        for c, e in out["worlds"][world].items():
            print("  %-6s gap closed: whole %5.1f%%  last50 %5.1f%%   "
                  "student cov90 %.3f  direct cov90 %.3f"
                  % (c, e["gap_closed_whole_pct"], e["gap_closed_last50_pct"],
                     e["student"]["cov90"]["mean"], e["direct"]["cov90"]["mean"]))


if __name__ == "__main__":
    main()
