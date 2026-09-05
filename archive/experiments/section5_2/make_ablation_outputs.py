# Section 5.9 ablation outputs: summary table, matched-routing figure, and the
# Section 5.6 uniform-vs-mixed table.
#
# All numbers under the AVERAGED NIG readout (Chapter 5's convention).
# Readout only over stored beliefs. PNG only, 6.3 in wide, dpi 200.
#
# Run: python -u experiments/section5_2/make_ablation_outputs.py

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_band import draw as _cal_band, ensure_visible as _cal_ylim
from averaged_readout import load_run, metrics

LAST = 50
BASE = "runs/chapter5_v2"
OUT = Path("runs/chapter5_v2")
FIGS = Path("figures")
W, DPI = 6.3, 200
INK, RED, BLUE, GREY, GREEN = "#22252a", "#c1121f", "#1f5fc1", "#8a8f98", "#1b7a3e"


def arm(pattern):
    M = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        d = Path(os.path.dirname(p))
        sd = int(yaml.safe_load(open(p))["env_seed"])
        M.append(metrics(load_run(d, sd), last=LAST)["averaged"])
    if not M:
        return None
    ms = lambda k: (st.mean([m[k] for m in M]),
                    st.stdev([m[k] for m in M]) if len(M) > 1 else 0.0)
    return dict(n=len(M), whole=ms("whole"), last50=ms("last50"),
                cov=ms("cov"), ratio=ms("ratio"))


def fam(pats):
    return [(lab, arm(p)) for lab, p in pats]


def spread(rows, key="whole"):
    v = [r[key][0] for _, r in rows if r]
    sds = [r[key][1] for _, r in rows if r]
    return max(v) - min(v), min(v), max(v), st.mean(sds)


# ---------------------------------------------------------------- summary
def summary_table():
    LAM = fam([(v, "%s/s5_9_lam_%s/*/manifest.yaml" % (BASE, v.replace(".", "p")))
               for v in ("1.0", "2.5", "5.0", "10.0", "25.0")])
    RHO = fam([(r.replace("p", "."),
                "%s/s5_9_consensus_rho/consensus_rho%s_seed*/manifest.yaml" % (BASE, r))
               for r in ("0p01", "0p05", "0p1", "0p2", "0p5", "0p8", "0p99")])
    WGT = fam([("nig_product", "%s/s531_b2/nig_product_sampled_seed*/manifest.yaml" % BASE),
               ("weighted", "%s/s5_9_nig_weighted/*/manifest.yaml" % BASE)])
    UNC = fam([(m, "%s/s5_9_uncertainty_%s/*/manifest.yaml" % (BASE, m))
               for m in ("epistemic", "aleatoric", "total")])
    TMP = fam([("het ON", "%s/s5_9_temper_het_temper/**/manifest.yaml" % BASE),
               ("het OFF", "%s/s5_9_temper_het_notemper/**/manifest.yaml" % BASE)])

    fams = [("lambda (regularisation)", LAM, 5, "1.0 - 25.0"),
            ("consensus rho", RHO, 7, "0.01 - 0.99"),
            ("weighted fusion", WGT, 2, "off / on"),
            ("uncertainty measure", UNC, 3, "epistemic/aleatoric/total"),
            ("gradient tempering", TMP, 2, "off / on")]

    print("=" * 108)
    print("TABLE -- SECTION 5.9 ABLATION SUMMARY  (cell space, averaged NIG readout, 5 seeds)")
    print("=" * 108)
    print("%-24s %4s %-26s %-24s %-13s %s"
          % ("family", "arms", "range swept", "whole-run MSE range",
             "seed sd", "verdict"))
    rows = []
    for name, F, n, rng in fams:
        sp, lo, hi, sd = spread(F)
        null = sp < 2 * sd
        verdict = ("NULL (spread %.5f < 2 x seed sd %.5f)" % (sp, sd) if null
                   else "EFFECT (spread %.5f = %.1f x seed sd)" % (sp, sp / sd))
        print("%-24s %4d %-26s %-24s %-13s %s"
              % (name, n, rng, "%.5f - %.5f" % (lo, hi), "%.5f" % sd, verdict))
        rows.append((name, n, rng, lo, hi, sd, sp, "null" if null else "effect"))
    print()
    print("  NULL means the whole-run MSE spread across every arm in the family is")
    print("  smaller than twice the within-arm seed standard deviation, i.e. the")
    print("  parameter moves the result less than re-seeding does.")
    print()
    print("  lambda detail (the only non-null):")
    print("  %-10s %-21s %-21s %-13s %s"
          % ("lam", "whole-run MSE", "last-50 MSE", "coverage", "hw:RMS"))
    for lab, r in LAM:
        print("  %-10s %-21s %-21s %-13s %.3f +/- %.3f"
              % (lab, "%.5f +/- %.5f" % r["whole"], "%.5f +/- %.5f" % r["last50"],
                 "%.3f +/- %.3f" % r["cov"], r["ratio"][0], r["ratio"][1]))

    with open(OUT / "table_5_9_ablation_summary.csv", "w", encoding="utf8") as f:
        f.write("family,n_arms,range_swept,whole_run_mse_min,whole_run_mse_max,"
                "mean_seed_sd,spread,verdict\n")
        for r in rows:
            f.write("%s,%d,%s,%.6f,%.6f,%.6f,%.6f,%s\n"
                    % (r[0], r[1], r[2].replace(",", ";"), r[3], r[4], r[5], r[6], r[7]))
    print("\n  wrote %s" % (OUT / "table_5_9_ablation_summary.csv"))
    return LAM


# ---------------------------------------------------------------- 5.6 table
def uniform_table():
    rows = [("heterogeneous (mixed)", arm("%s/s5_6_het/het_nig_product*/manifest.yaml" % BASE)),
            ("uniform narrow", arm("%s/s5_6_homog_narrow/**/manifest.yaml" % BASE)),
            ("uniform baseline", arm("%s/s5_6_homog_baseline/**/manifest.yaml" % BASE)),
            ("uniform wide", arm("%s/s5_6_homog_wide/**/manifest.yaml" % BASE))]
    ref = rows[0][1]["whole"][0]
    print("\n" + "=" * 108)
    print("TABLE -- SECTION 5.6  MIXED vs UNIFORM CAPACITY")
    print("nig_product, all arms matched (same script, world, geometry, lam, lr,")
    print("colour filter, seeds). Cell space, averaged NIG readout, 5 seeds.")
    print("=" * 108)
    print("%-24s %3s %-21s %-21s %-15s %-15s %10s"
          % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage",
             "hw : RMS", "vs mixed"))
    for lab, r in rows:
        pct = 100 * (r["whole"][0] - ref) / ref
        print("%-24s %3d %-21s %-21s %-15s %-15s %+9.2f%%"
              % (lab, r["n"], "%.5f +/- %.5f" % r["whole"],
                 "%.5f +/- %.5f" % r["last50"], "%.3f +/- %.3f" % r["cov"],
                 "%.3f +/- %.3f" % r["ratio"], pct))
    best = min(rows[1:], key=lambda x: x[1]["whole"][0])
    print("\n  best uniform arm: %s (%.5f) vs heterogeneous (%.5f)"
          % (best[0], best[1]["whole"][0], ref))
    print("  -> the mixture %s the best uniform configuration"
          % ("BEATS" if ref < best[1]["whole"][0] else "does NOT beat"))
    with open(OUT / "table_5_6_uniform_vs_mixed.csv", "w", encoding="utf8") as f:
        f.write("arm,n_seeds,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,"
                "last50_mse_sd,coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd,"
                "whole_run_pct_vs_mixed\n")
        for lab, r in rows:
            f.write("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n"
                    % (lab, r["n"], r["whole"][0], r["whole"][1], r["last50"][0],
                       r["last50"][1], r["cov"][0], r["cov"][1], r["ratio"][0],
                       r["ratio"][1], 100 * (r["whole"][0] - ref) / ref))
    print("  wrote %s" % (OUT / "table_5_6_uniform_vs_mixed.csv"))


# ---------------------------------------------------------------- routing fig
def fig_routing():
    """Matched routing: the only non-null ablation. Policy is the only
    difference between the arms -- verified by kwargs capture before the runs."""
    A = [("random", arm("%s/s5_9_routing_matched/random_seed*/manifest.yaml" % BASE), INK),
         ("uncertainty\nguided",
          arm("%s/s5_9_routing_matched/uncertainty_guided_seed*/manifest.yaml" % BASE), GREEN)]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.4))
    fig.subplots_adjust(left=0.108, right=0.995, bottom=0.175, top=0.845, wspace=0.44)

    ax = axes[0]
    x = np.arange(2)
    w = 0.36
    for j, (key, hatch, lab) in enumerate((("whole", "", "whole-run"),
                                           ("last50", "///", "last-50"))):
        ax.bar(x + (j - 0.5) * w, [a[1][key][0] for a in A], w,
               yerr=[a[1][key][1] for a in A], capsize=3,
               color=[a[2] for a in A], alpha=0.95 if j == 0 else 0.45,
               hatch=hatch, edgecolor="white", lw=0.5, label=lab)
    ax.set_ylabel("cell-space MSE", fontsize=8)
    ax.set_title("accuracy", fontsize=9, pad=11)
    # headroom so the legend clears the bars rather than sitting on them
    ax.set_ylim(0, max(a[1]["whole"][0] + a[1]["whole"][1] for a in A) * 1.42)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2,
              handlelength=1.4, columnspacing=1.0)

    for ax_, key, lab, ref, title in ((axes[1], "cov", "90% coverage", 0.90, "coverage"),
                                      (axes[2], "ratio", "hw : RMS", 1.0,
                                       "hw : RMS")):
        ax_.bar(x, [a[1][key][0] for a in A], 0.55,
                yerr=[a[1][key][1] for a in A], capsize=3,
                color=[a[2] for a in A], edgecolor="white", lw=0.5)
        if abs(ref - 1.0) < 1e-9:
            _cal_band(ax_); _cal_ylim(ax_)
        else:
            ax_.axhline(ref, color=RED, ls=":", lw=1.0)
        ax_.set_ylabel(lab, fontsize=8)
        ax_.set_title(title, fontsize=9, pad=11)

    for ax_ in axes:
        ax_.set_xticks(x)
        ax_.set_xticklabels([a[0] for a in A], fontsize=7.5)
        ax_.tick_params(labelsize=8)
    FIGS.mkdir(exist_ok=True)
    p = FIGS / "ch5_5_9_routing_matched.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("\n  wrote %s" % p)

    r, g = A[0][1], A[1][1]
    print("  last-50 MSE  %.5f -> %.5f  (%+.1f%%)"
          % (r["last50"][0], g["last50"][0],
             100 * (g["last50"][0] - r["last50"][0]) / r["last50"][0]))
    print("  whole-run    %.5f -> %.5f  (%+.1f%%)"
          % (r["whole"][0], g["whole"][0],
             100 * (g["whole"][0] - r["whole"][0]) / r["whole"][0]))
    print("  coverage     %.3f -> %.3f     hw:RMS %.3f -> %.3f"
          % (r["cov"][0], g["cov"][0], r["ratio"][0], g["ratio"][0]))


if __name__ == "__main__":
    summary_table()
    uniform_table()
    fig_routing()
