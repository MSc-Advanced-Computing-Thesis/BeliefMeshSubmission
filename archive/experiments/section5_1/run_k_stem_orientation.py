# K (exploratory) -- is the population-level elevated band explained by
# variation in the natural slant of handwritten sevens?
#
# Hypothesis: the model fails when the digit's stem reaches a particular
# orientation IN IMAGE SPACE. MNIST sevens are written at varying natural
# slants, so that fixed stem orientation corresponds to a different APPLIED
# rotation for each instance, and individually narrow failures smear into the
# ~30-degree band the population panel shows.
#
# Test: re-bin the existing per-sample error/epistemic against ABSOLUTE STEM
# ORIENTATION (natural slant + applied rotation, wrapped) instead of against
# applied rotation, and ask whether the band narrows and sharpens.
#
# Read-only with respect to every existing artefact. The only model output used
# is runs/section5_1/r1a/per_sample_baseline.csv; nothing is re-evaluated.
#
# Run: python -u experiments/section5_1/run_k_stem_orientation.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import read_dump, write_rows_csv
from make_figures import BLUE, FIGS, GREY, INK, RED, WIDTH

from beliefmesh.data.digits import get_digit7_splits
from beliefmesh.data.filters import make_green_digits

SRC = Path("runs/section5_1/r1a/per_sample_baseline.csv")
OUT = Path("runs/section5_1/k_stem_orientation")
BIN_DEG = 15.0
ELEVATION_FACTOR = 2.0        # a bin is "elevated" at >= 2x the median bin
STABILITY_N = 120             # instances used for the method-stability test
STABILITY_ANGLES = np.arange(0.0, 360.0, 30.0)

_TO_TENSOR = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def ink_weight(img: torch.Tensor) -> np.ndarray:
    """Per-pixel ink mass from a prepared (green digit on white) render.
    A soft weight rather than a threshold: antialiased stroke edges carry
    partial mass, so the moment estimates do not jump when a boundary pixel
    crosses an arbitrary cutoff."""
    return np.clip(1.0 - img.mean(dim=0).numpy(), 0.0, 1.0)


def slant_principal_axis(w: np.ndarray) -> float:
    """Orientation of the principal axis of the ink mass, in degrees, in the
    same sense as TF.rotate's applied angle.

    Second central moments give an AXIS (mod 180). The 180-degree ambiguity is
    resolved with the third moment along that axis: a seven is mass-heavy at
    the bar end, so the sign of the skewness picks a consistent end."""
    ys, xs = np.nonzero(w > 0)
    m = w[ys, xs]
    # image row index grows downward; negate so the angle is in the usual
    # counter-clockwise-positive sense TF.rotate uses
    x = xs - np.average(xs, weights=m)
    y = -(ys - np.average(ys, weights=m))
    mxx = np.average(x * x, weights=m)
    myy = np.average(y * y, weights=m)
    mxy = np.average(x * y, weights=m)
    theta = 0.5 * np.arctan2(2 * mxy, mxx - myy)          # mod pi
    proj = x * np.cos(theta) + y * np.sin(theta)
    if np.average(proj ** 3, weights=m) < 0:              # point the heavy end
        theta += np.pi
    return float(np.degrees(theta))


def slant_dominant_stroke(w: np.ndarray) -> float:
    """Direction from the ink centroid to the ink-weighted far extremity --
    a crude 'dominant stroke' estimate, carried only as the comparison the
    task asks for."""
    ys, xs = np.nonzero(w > 0)
    m = w[ys, xs]
    cx, cy = np.average(xs, weights=m), np.average(ys, weights=m)
    x, y = xs - cx, -(ys - cy)
    r = np.hypot(x, y)
    far = r >= np.quantile(r, 0.9)                        # outer 10% of ink
    return float(np.degrees(np.arctan2(np.average(y[far], weights=m[far]),
                                       np.average(x[far], weights=m[far]))))


def wrap180(a):
    return ((np.asarray(a, dtype=float) + 180.0) % 360.0) - 180.0


def circ_sd_deg(a):
    """Circular standard deviation, degrees."""
    r = np.abs(np.mean(np.exp(1j * np.radians(np.asarray(a)))))
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(r, 1e-12)))))


def load_instances(mnist_root="data/mnist"):
    base = datasets.MNIST(root=mnist_root, train=True, download=True)
    sevens = base.data[base.targets == 7]
    _, eval_idx = get_digit7_splits(root=mnist_root)
    return sevens, np.asarray(eval_idx)


def prepared(sevens, pos):
    return make_green_digits(TF.invert(_TO_TENSOR(sevens[pos])))


def stability(sevens, eval_idx):
    """Which estimator is more stable? For each instance, render at known
    applied rotations and check that (estimate - applied) is constant. The
    method whose residual has the smaller circular sd is the one whose output
    genuinely tracks the digit's orientation."""
    rng = np.random.default_rng(42)
    chosen = rng.choice(eval_idx, size=STABILITY_N, replace=False)
    res = {"principal_axis": [], "dominant_stroke": []}
    for pos in chosen:
        img = prepared(sevens, int(pos))
        for name, fn in (("principal_axis", slant_principal_axis),
                         ("dominant_stroke", slant_dominant_stroke)):
            est = [fn(ink_weight(TF.rotate(img, float(a), fill=1.0)))
                   for a in STABILITY_ANGLES]
            res[name].append(circ_sd_deg(wrap180(np.array(est) - STABILITY_ANGLES)))
    return {k: {"median_circular_sd_deg": float(np.median(v)),
                "mean_circular_sd_deg": float(np.mean(v)),
                "p90_circular_sd_deg": float(np.percentile(v, 90)),
                "n_instances": STABILITY_N}
            for k, v in res.items()}


def bin_table(x_deg, err, epi, label):
    edges = np.arange(-180.0, 180.0 + BIN_DEG, BIN_DEG)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (x_deg >= lo) & (x_deg < hi)
        if not m.any():
            continue
        rows.append({
            "binning": label, "bin_lo_deg": float(lo), "bin_hi_deg": float(hi),
            "bin_centre_deg": float((lo + hi) / 2), "n": int(m.sum()),
            "mean_abs_error_deg": float(np.mean(err[m])),
            "p90_abs_error_deg": float(np.percentile(err[m], 90)),
            "max_abs_error_deg": float(np.max(err[m])),
            "mean_epistemic": float(np.mean(epi[m])),
        })
    med = float(np.median([r["mean_abs_error_deg"] for r in rows]))
    med_e = float(np.median([r["mean_epistemic"] for r in rows]))
    for r in rows:
        r["ae_ratio_to_median"] = r["mean_abs_error_deg"] / med
        r["epi_ratio_to_median"] = r["mean_epistemic"] / med_e
        r["elevated"] = bool(r["ae_ratio_to_median"] >= ELEVATION_FACTOR)
    return rows, med, med_e


def band_stats(rows, med, med_e):
    """Same definition under both binnings: elevated = bin mean absolute error
    at or above ELEVATION_FACTOR x the median bin. Width is the total angular
    extent of elevated bins; the longest contiguous run is reported too, since
    a band that fragments is not a narrower band."""
    elev = [r for r in rows if r["elevated"]]
    idx = [i for i, r in enumerate(rows) if r["elevated"]]
    runs, cur = [], []
    for i in idx:
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = [i]
    if cur:
        runs.append(cur)
    longest = max(runs, key=len) if runs else []
    peak = max(rows, key=lambda r: r["mean_abs_error_deg"])
    return {
        "n_elevated_bins": len(elev),
        "total_elevated_width_deg": len(elev) * BIN_DEG,
        "longest_contiguous_run_bins": len(longest),
        "longest_contiguous_width_deg": len(longest) * BIN_DEG,
        "contiguous_range_deg": ([rows[longest[0]]["bin_lo_deg"],
                                  rows[longest[-1]]["bin_hi_deg"]] if longest else None),
        "n_separate_runs": len(runs),
        "median_bin_mean_abs_error_deg": med,
        "median_bin_mean_epistemic": med_e,
        "peak_bin": [peak["bin_lo_deg"], peak["bin_hi_deg"]],
        "peak_mean_abs_error_deg": peak["mean_abs_error_deg"],
        "peak_to_median_ratio_error": peak["mean_abs_error_deg"] / med,
        "peak_mean_epistemic": peak["mean_epistemic"],
        "peak_to_median_ratio_epistemic": peak["mean_epistemic"] / med_e,
        "elevated_bins": [[r["bin_lo_deg"], r["bin_hi_deg"]] for r in elev],
    }


def figure(rows_applied, rows_stem):
    """Bottom panel style of the existing top panel, for the stem binning,
    with the applied binning drawn faint behind it for comparison."""
    g = lambda rs, k: np.array([r[k] for r in rs], dtype=float)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    fig.subplots_adjust(left=0.090, right=0.700, bottom=0.175, top=0.975)
    ax.plot(g(rows_applied, "bin_centre_deg"), g(rows_applied, "mean_abs_error_deg"),
            "-", color=GREY, lw=1.0, zorder=2)
    ax.plot(g(rows_stem, "bin_centre_deg"), g(rows_stem, "mean_abs_error_deg"),
            "o-", color=INK, ms=3.5, lw=1.3, zorder=4)
    ax.plot(g(rows_stem, "bin_centre_deg"), g(rows_stem, "p90_abs_error_deg"),
            "^--", color=RED, ms=3, lw=1.0, zorder=3)
    ax.set_xlabel("absolute stem orientation (deg)", fontsize=8)
    ax.set_ylabel("absolute circular error (deg)", fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_xticks(range(-180, 181, 45))
    ax.tick_params(labelsize=8)
    axt = ax.twinx()
    axt.plot(g(rows_stem, "bin_centre_deg"), g(rows_stem, "mean_epistemic"),
             "s:", color=BLUE, ms=3, lw=1.1, zorder=5)
    axt.set_ylabel("mean epistemic uncertainty", color=BLUE, fontsize=8)
    axt.tick_params(axis="y", labelcolor=BLUE, labelsize=8)
    from matplotlib.lines import Line2D
    axt.legend(handles=[
        Line2D([], [], marker="o", ls="-", color=INK, ms=3.5, lw=1.3,
               label="bin mean (stem)"),
        Line2D([], [], marker="^", ls="--", color=RED, ms=3, lw=1.0,
               label="p90 (stem)"),
        Line2D([], [], marker="s", ls=":", color=BLUE, ms=3, lw=1.1,
               label="mean epistemic (stem)"),
        Line2D([], [], ls="-", color=GREY, lw=1.0,
               label="bin mean (applied),\nfor comparison"),
    ], fontsize=6.8, frameon=False, loc="center left",
        bbox_to_anchor=(1.135, 0.5), handlelength=1.9, handletextpad=0.6,
        labelspacing=0.65, borderpad=0.0)
    FIGS.mkdir(parents=True, exist_ok=True)
    png = FIGS / "section5_1_stem_orientation.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("  wrote %s" % png)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = read_dump(SRC)
    sevens, eval_idx = load_instances()
    assert len(eval_idx) == len(d["sample_index"]), "dump/split length mismatch"

    print("stability test (%d instances x %d applied rotations) ..."
          % (STABILITY_N, len(STABILITY_ANGLES)))
    stab = stability(sevens, eval_idx)
    for k, v in stab.items():
        print("  %-16s median circular sd of (estimate - applied) = %6.2f deg"
              % (k, v["median_circular_sd_deg"]))
    chosen = min(stab, key=lambda k: stab[k]["median_circular_sd_deg"])
    fn = {"principal_axis": slant_principal_axis,
          "dominant_stroke": slant_dominant_stroke}[chosen]
    print("  -> using %s" % chosen)

    # natural slant at zero applied rotation, per held-out instance
    order = np.argsort(d["sample_index"])
    # wrap180 here, not only in the summary: slant_principal_axis can return
    # up to +270 after the third-moment flip, and an unwrapped value silently
    # falls outside any [-180, 180) analysis downstream.
    slant = wrap180([fn(ink_weight(prepared(sevens, int(p)))) for p in eval_idx])
    applied = d["true_angle_deg"][order]
    err = np.abs(d["circular_error_deg"])[order]
    epi = d["epistemic"][order]
    stem = wrap180(slant + applied)

    write_rows_csv([{"sample_index": int(d["sample_index"][order][i]),
                     "eval_position": int(eval_idx[i]),
                     "natural_slant_deg": float(slant[i]),
                     "applied_rotation_deg": float(applied[i]),
                     "abs_stem_orientation_deg": float(stem[i]),
                     "abs_circular_error_deg": float(err[i]),
                     "epistemic": float(epi[i])}
                    for i in range(len(slant))], OUT / "per_sample_stem.csv")

    rows_applied, med_a, mede_a = bin_table(applied, err, epi, "applied")
    rows_stem, med_s, mede_s = bin_table(stem, err, epi, "stem")
    write_rows_csv(rows_applied + rows_stem, OUT / "binned_both.csv")
    band_a = band_stats(rows_applied, med_a, mede_a)
    band_s = band_stats(rows_stem, med_s, mede_s)

    sl = slant
    slant_dist = {
        "n": int(len(sl)),
        "circular_sd_deg": circ_sd_deg(sl),
        "mean_deg": float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(sl)))))),
        "p5_deg": float(np.percentile(sl, 5)), "p25_deg": float(np.percentile(sl, 25)),
        "median_deg": float(np.median(sl)), "p75_deg": float(np.percentile(sl, 75)),
        "p95_deg": float(np.percentile(sl, 95)),
        "iqr_deg": float(np.percentile(sl, 75) - np.percentile(sl, 25)),
        "p5_to_p95_span_deg": float(np.percentile(sl, 95) - np.percentile(sl, 5)),
    }

    summary = {
        "source": str(SRC), "bin_width_deg": BIN_DEG,
        "elevation_criterion": "bin mean abs error >= %.1fx the median bin"
                               % ELEVATION_FACTOR,
        "slant_estimator": {"chosen": chosen, "stability": stab,
                            "why": "lower circular sd of (estimate - applied "
                                   "rotation) means the estimate tracks the "
                                   "digit's actual orientation rather than "
                                   "responding to rasterisation"},
        "slant_distribution": slant_dist,
        "band_under_applied_rotation": band_a,
        "band_under_stem_orientation": band_s,
    }
    with open(OUT / "summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)

    print("\nslant distribution: circular sd %.1f deg, IQR %.1f deg, p5-p95 span %.1f deg"
          % (slant_dist["circular_sd_deg"], slant_dist["iqr_deg"],
             slant_dist["p5_to_p95_span_deg"]))
    print("\n%-22s %10s %10s" % ("", "applied", "stem"))
    for k in ("n_elevated_bins", "total_elevated_width_deg",
              "longest_contiguous_width_deg", "n_separate_runs",
              "peak_to_median_ratio_error", "peak_to_median_ratio_epistemic"):
        print("%-22s %10.2f %10.2f" % (k, band_a[k], band_s[k]))
    print("peak bin  applied %s   stem %s" % (band_a["peak_bin"], band_s["peak_bin"]))

    figure(rows_applied, rows_stem)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
