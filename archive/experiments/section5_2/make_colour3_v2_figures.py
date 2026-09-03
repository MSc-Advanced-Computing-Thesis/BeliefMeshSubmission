# Section 5.3 colour three-node V2: geometry figure, evidence-vs-error figure,
# and the compressed results table. Stored artefacts only; no runs.
# PNG only, dpi 200.
#
# Run: python -u experiments/section5_2/make_colour3_v2_figures.py

from __future__ import annotations

import glob
import os
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from averaged_readout import (load_run, metrics, averaged_params, truth_for_run,
                              wrap_np)
from beliefmesh.node.mesh import fov_cells
from stage6_spatial_mesh.cert_mse_metrics import per_timestep_then_averaged

G, FOV, LAST, DPI = 22, 7, 50, 200
A, B, C = (8, 9), (13, 9), (10, 12)
ARMS = ["frozen", "naive", "certainty", "nig_product", "nig_product_avgtrain"]
from arm_names import name as _arm_name
SHORT = {k: _arm_name(k, wrapped=True) for k in ARMS}
ROOT = "runs/chapter5_v2/s5_3_colour_three_node_v2"
FIGS = Path("figures")
INK, RED, BLUE, GREY, GREEN = "#22252a", "#c1121f", "#1f5fc1", "#8a8f98", "#1b7a3e"

a = set(fov_cells(*A, FOV, G))
b = set(fov_cells(*B, FOV, G))
c = set(fov_cells(*C, FOV, G))
SHARED = np.array(sorted(c & (a | b)))


def colour_field2d():
    shared = c & (a | b)
    lo = min(cc for _, cc in shared); hi = max(cc for _, cc in shared)
    v = np.clip((hi - np.arange(G, dtype=float)) / float(hi - lo), 0.0, 1.0)
    grid = np.tile(v[None, :], (G, 1))
    for r, cc in (a - b - c):
        grid[r, cc] = 1.0
    for r, cc in (b - a - c):
        grid[r, cc] = 0.0
    return grid


def mask_of(cells):
    m = np.zeros((G, G), bool)
    for r, cc in cells:
        m[r, cc] = True
    return m


def rect_of(cells, ax, colour, ls, lw, label=None, z=4):
    """Outline ONLY -- no fill. All three marked regions are exact rectangles
    (checked), so a single Rectangle each is faithful. Filling them would hide
    the colour gradient, which is what the figure exists to show."""
    rs = [x[0] for x in cells]; cs = [x[1] for x in cells]
    ax.add_patch(Rectangle((min(cs) - 0.5, min(rs) - 0.5),
                           max(cs) - min(cs) + 1, max(rs) - min(rs) + 1,
                           fill=False, edgecolor=colour, ls=ls, lw=lw,
                           zorder=z, label=label))


def fig_geometry():
    fig, ax = plt.subplots(figsize=(6.3, 3.05))
    fig.subplots_adjust(left=0.082, right=0.475, bottom=0.155, top=0.965)
    im = ax.imshow(colour_field2d(), cmap="RdBu_r", vmin=0.0, vmax=1.0,
                   origin="upper", interpolation="nearest", zorder=0)

    # the three FOVs: thin, neutral, so they do not compete with the regions
    for ctr, lab in ((A, "A"), (B, "B"), (C, "C")):
        rect_of(fov_cells(*ctr, FOV, G), ax, "#3a3f46", "-", 0.9, z=3)
        ax.text(ctr[0], ctr[1], lab, ha="center", va="center", fontsize=8,
                color="white", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.14", fc="#22252a", ec="none"))
    # the three marked regions: outline only, distinct colour AND line style
    rect_of(a & b, ax, "#111317", "-", 2.2, "A$\cap$B: excluded from both anchors")
    rect_of(c & (a | b), ax, "#1b7a3e", "--", 2.2, "C's supervision (28 cells)")
    rect_of(c - a - b, ax, "#6b3fa0", ":", 2.4, "C exclusive: no supervision")

    for cells in (a - b - c, b - a - c):
        arr = np.array(sorted(cells))
        ax.plot(arr[:, 1], arr[:, 0], marker=".", ls="none", ms=1.9,
                color="white", mec="#22252a", mew=0.2, zorder=6,
                label=("anchor wearable confined"
                       if cells == (a - b - c) else None))

    ax.set_xlim(2.5, 18.5); ax.set_ylim(17.5, 3.5)
    ax.set_xticks([4, 8, 12, 16]); ax.set_yticks([4, 8, 12, 16])
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("cell column", fontsize=8)
    ax.set_ylabel("cell row", fontsize=8)

    cax = fig.add_axes([0.492, 0.155, 0.017, 0.72])
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_yticklabels(["0.0 blue", "0.25", "0.50", "0.75", "1.0 red"],
                          fontsize=6.6)
    cb.ax.set_title("filter\nvalue", fontsize=7, pad=5)
    cb.outline.set_visible(False)

    handles = [
        plt.Line2D([], [], color="#111317", ls="-", lw=2.2,
                   label="A$\cap$B: excluded from both"),
        plt.Line2D([], [], color="#1b7a3e", ls="--", lw=2.2,
                   label="C's supervision (28 cells)"),
        plt.Line2D([], [], color="#6b3fa0", ls=":", lw=2.4,
                   label="C exclusive: unsupervised"),
        plt.Line2D([], [], color="#3a3f46", ls="-", lw=0.9, label="field of view"),
        plt.Line2D([], [], marker=".", ls="none", ms=6, color="white",
                   mec="#22252a", mew=0.5, label="anchor wearable confined")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.575, 0.55),
               fontsize=7.2, frameon=False, handlelength=2.4, labelspacing=0.85)
    p = FIGS / "ch5_5_3_colour3_geometry.png"
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print("  wrote %s" % p)


def per_run(d, seed):
    d = Path(d)
    R = load_run(d, seed); T = R["T"]
    Cm = mask_of(c)
    m = metrics(R, last=LAST, mask=Cm)["averaged"]
    bel = np.load(d / "cell_beliefs_steps.npy").astype(np.float64)
    nc = np.load(d / "cell_ncov_steps.npy").astype(int)
    gam, nu_s, al_s, be_s = averaged_params(bel, nc)
    epi = be_s / (np.maximum(nu_s, 1e-6) * np.maximum(al_s - 1.0, 1e-6))
    cert = 1.0 / (1.0 + np.minimum(epi, 10.0))
    r_cert, _, _, _ = per_timestep_then_averaged(
        np.where(Cm[None], R["averaged"][0], np.nan),
        np.where(Cm[None], cert, np.nan), T - LAST, T)
    tr = truth_for_run(d, seed, T, G)[-LAST:]
    sub = bel[-LAST:][:, SHARED[:, 0], SHARED[:, 1], :, :]
    nu, g = sub[..., 1], sub[..., 0]
    err = np.abs(wrap_np(g - tr[:, SHARED[:, 0], SHARED[:, 1]][..., None])) * 180
    fin = np.isfinite(nu) & np.isfinite(err)
    anch = np.zeros(nu.shape, bool)
    for k, cell in enumerate(map(tuple, SHARED)):
        ids = [i for i, s in enumerate((a, b, c)) if cell in s]
        for j, nid in enumerate(ids):
            if nid in (0, 1):
                anch[:, k, j] = True
    ma = fin & anch
    n = np.isfinite(g).sum(-1); ok = n >= 2
    gf = np.isfinite(g)
    nucv = float(np.nanmean(np.where(ok, np.nanstd(np.where(gf, nu, np.nan), -1) /
                 np.maximum(np.nanmean(np.where(gf, nu, np.nan), -1), 1e-12), np.nan)))
    x = np.where(gf, wrap_np(g - g[..., 0][..., None]), np.nan)
    gs = float(np.nanmean(np.where(ok, np.nanmax(x, -1) - np.nanmin(x, -1), np.nan))) * 180
    return dict(**m, r_cert=r_cert, nucv=nucv, gs=gs,
                r_anch=float(np.corrcoef(nu[ma], err[ma])[0, 1]),
                r_all=float(np.corrcoef(nu[fin], err[fin])[0, 1]))


def collect():
    D = {}
    for arm in ARMS:
        V = []
        for p in sorted(glob.glob("%s/%s_seed*/manifest.yaml" % (ROOT, arm))):
            d = Path(os.path.dirname(p))
            V.append(per_run(d, int(yaml.safe_load(open(p))["env_seed"])))
        if V:
            D[arm] = V
    return D


def fig_evidence(D, outname="ch5_5_3_colour3_evidence_vs_error", arms=None):
    arms = list(ARMS) if arms is None else list(arms)
    ms = lambda V, k: (st.mean([v[k] for v in V]),
                       st.stdev([v[k] for v in V]) if len(V) > 1 else 0.0)
    mu = [ms(D[a_], "r_anch")[0] for a_ in arms]
    sd = [ms(D[a_], "r_anch")[1] for a_ in arms]
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.235, top=0.885)
    x = np.arange(len(arms))
    cols = ([GREY] + [INK, RED, BLUE, GREEN])[:len(arms)]
    ax.bar(x, mu, 0.58, yerr=sd, capsize=3.5, color=cols,
           edgecolor="white", lw=0.6)
    ax.axhline(0.0, color=INK, lw=1.0)
    # frozen is the control: separate it from the four trained arms
    ax.axvline(0.5, color=GREY, ls="--", lw=0.9, zorder=0)
    ax.annotate("control", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, 2.5), textcoords="offset points", ha="center",
                va="bottom", fontsize=7, color=GREY)
    ax.annotate("trained arms", xy=((len(arms) + 1) / 2.0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, 2.5), textcoords="offset points", ha="center",
                va="bottom", fontsize=7, color=GREY)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[a_] for a_ in arms], fontsize=7.5)
    ax.set_ylabel("r(contributor $\\nu$, contributor error)", fontsize=8)
    ax.set_title("evidence against error, anchor contributors on the student's shared cells",
                 fontsize=9, pad=10)
    ax.tick_params(labelsize=8)
    p = FIGS / (outname + ".png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print("  wrote %s" % p)
    print("\n  exact values plotted (5-seed mean +/- sd):")
    for a_, m_, s_ in zip(arms, mu, sd):
        print("    %-24s %+.3f +/- %.3f" % (a_, m_, s_))


def table(D):
    ms = lambda V, k: (st.mean([v[k] for v in V]),
                       st.stdev([v[k] for v in V]) if len(V) > 1 else 0.0)
    print("\n" + "=" * 92)
    print("TABLE EXTRACT -- Section 5.3 colour three-node v2 (5 seeds, mean +/- sd)")
    print("=" * 92)
    print("%-24s %-20s %-15s %-15s %-18s"
          % ("arm", "last-50 MSE", "90% coverage", "hw : RMS", "r(nu, err) anchors"))
    rows = []
    for a_ in ARMS:
        V = D[a_]
        f = lambda k, p=5: "%.*f +/- %.*f" % (p, ms(V, k)[0], p, ms(V, k)[1])
        print("%-24s %-20s %-15s %-15s %-18s"
              % (a_, f("last50"), f("cov", 3), f("ratio", 3),
                 "%+.3f +/- %.3f" % ms(V, "r_anch")))
        rows.append((a_, ms(V, "last50"), ms(V, "cov"), ms(V, "ratio"),
                     ms(V, "r_anch"), ms(V, "r_cert"), ms(V, "nucv"), ms(V, "gs")))
    out = Path("runs/chapter5_v2/table_5_3_colour3_v2.csv")
    with open(out, "w", encoding="utf8") as f:
        f.write("arm,last50_mse_mean,last50_mse_sd,coverage90_mean,coverage90_sd,"
                "hw_rms_mean,hw_rms_sd,r_nu_err_anchors_mean,r_nu_err_anchors_sd\n")
        for r in rows:
            f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n"
                    % (r[0], r[1][0], r[1][1], r[2][0], r[2][1],
                       r[3][0], r[3][1], r[4][0], r[4][1]))
    print("  wrote %s" % out)

    print("\nFOR THE SECTION PROSE")
    print("  %-24s %-22s %-22s" % ("arm", "nu-CV", "gamma spread (deg)"))
    for r in rows:
        print("  %-24s %-22s %-22s"
              % (r[0], "%.3f +/- %.3f" % r[6], "%.2f +/- %.2f" % r[7]))
    print("    offset-world reference: nu-CV 0.101, gamma spread 2-5 deg")
    print("\n  cert-MSE r (WITHIN-node ordering) vs r(nu,err) (CROSS-contributor):")
    print("  %-24s %-22s %-22s" % ("arm", "cert-MSE r (within)", "r(nu,err) (cross)"))
    for r in rows:
        print("  %-24s %-22s %-22s"
              % (r[0], "%+.3f +/- %.3f" % r[5], "%+.3f +/- %.3f" % r[4]))


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    fig_geometry()
    D = collect()
    fig_evidence(D)
    table(D)
