# J -- angle-binned error and uncertainty on the MESH digit instance.
#
# F binned error/epistemic by true angle over the 1,253-sample held-out
# POPULATION. The spatially structured mesh experiments do not use that
# population: with per_cell_digits=False (the default, and what every reported
# spatial run used) GridEnvironment renders every cell from ONE fixed digit-7
# instance -- base.data[base.targets == 7][0], i.e. position 0 of the digit-7
# subset. This re-runs F's analysis on that single instance, to establish
# whether the 120-150 and +/-165-180 exclusion ranges are correctly targeted
# for the image they actually apply to.
#
# NO SAMPLING VARIANCE HERE. One fixed image, deterministic evenly-spaced
# angles, a frozen checkpoint, no dropout: re-running produces bit-identical
# output. The 20-seed protocol exists to average out RotatedDigitDataset's
# per-pass rotation re-roll, which has no analogue in this design -- there is
# nothing to average. These artefacts are therefore NOT single-pass results of
# the kind the protocol was introduced to replace, and the summary says so.
#
# Rendering matches GridEnvironment.get_cell_input exactly: invert +
# make_green_digits once, then TF.rotate(fill=1.0) per angle, and NO colour
# filter (apply_colour_filter=False equivalent). Read-only; nothing on a
# reported code path is touched.
#
# Run: python -u experiments/section5_1/run_j_single_instance.py

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

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

from analyse import write_rows_csv
from dump import load_model
from figure_sweeps import BLUE, GREY, INK, RED, WIDTH, save

from beliefmesh.data.digits import get_digit7_splits
from beliefmesh.data.filters import make_green_digits
from beliefmesh.metrics.circular import circular_diff

OUT = Path(ART + "/5.1_baseline_model_and_calibration/j_single_instance")
N_RENDERS = 720          # 30 per 15-degree bin, evenly spaced over [-180, 180)
BIN_DEG = 15.0
BATCH = 64
MESH_EXCLUDED = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]

_TO_TENSOR = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def locate_instance(mnist_root: str = "data/mnist") -> dict:
    """The single fixed instance GridEnvironment uses, and which partition of
    the seed-42 80/20 split it belongs to."""
    base = datasets.MNIST(root=mnist_root, train=True, download=True)
    mask = (base.targets == 7)
    subset_positions = np.nonzero(mask.numpy())[0]
    train_idx, eval_idx = get_digit7_splits(root=mnist_root)
    in_eval = bool(0 in set(eval_idx.tolist()))
    return {
        "subset_position": 0,
        "mnist_train_index": int(subset_positions[0]),
        "digit7_subset_size": int(len(subset_positions)),
        "partition": "held-out" if in_eval else "training",
        "note": "GridEnvironment(per_cell_digits=False) uses "
                "base.data[base.targets == 7][0]; the frozen baseline "
                "checkpoint was trained on the training partition, so this "
                "instance was SEEN during Stage 0 pretraining.",
    }, base.data[mask][0]


def in_excluded(lo: float, hi: float) -> bool:
    return any(lo < ehi and hi > elo for elo, ehi in MESH_EXCLUDED)


def evaluate_instance(base_image: torch.Tensor, device) -> dict:
    """Render the instance at N_RENDERS evenly-spaced angles and run the
    frozen baseline checkpoint over them."""
    model = load_model("baseline", device)
    prepared = make_green_digits(TF.invert(_TO_TENSOR(base_image)))
    angles = np.linspace(-180.0, 180.0, N_RENDERS, endpoint=False)

    rows = {k: [] for k in ("gamma", "nu", "alpha", "beta")}
    with torch.no_grad():
        for s in range(0, len(angles), BATCH):
            chunk = angles[s:s + BATCH]
            imgs = torch.stack([TF.rotate(prepared, float(a), fill=1.0)
                                for a in chunk]).to(device)
            g, n, a, b = model(imgs)
            for k, v in zip(("gamma", "nu", "alpha", "beta"), (g, n, a, b)):
                rows[k].append(v.cpu().numpy().astype(np.float64))

    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["true_angle_deg"] = angles
    target = torch.tensor(angles / 180.0, dtype=torch.float32)
    out["circular_error_deg"] = circular_diff(
        torch.tensor(out["gamma"], dtype=torch.float32), target).numpy().astype(np.float64) * 180.0

    am1 = out["alpha"] - 1.0
    out["alpha_floor_bound"] = (am1 <= 0).astype(np.int64)
    out["epistemic"] = out["beta"] / (out["nu"] * am1)
    out["aleatoric"] = out["beta"] / am1
    out["total"] = out["beta"] * (1.0 + out["nu"]) / (out["nu"] * am1)
    out["n_params"] = sum(p.numel() for p in model.parameters())
    return out


def bin_table(d) -> list[dict]:
    ang, err = d["true_angle_deg"], np.abs(d["circular_error_deg"])
    edges = np.arange(-180.0, 180.0 + BIN_DEG, BIN_DEG)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (ang >= lo) & (ang < hi)
        if not m.any():
            continue
        rows.append({
            "bin_lo_deg": float(lo), "bin_hi_deg": float(hi),
            "bin_centre_deg": float((lo + hi) / 2), "n": int(m.sum()),
            "mean_abs_error_deg": float(np.mean(err[m])),
            "median_abs_error_deg": float(np.median(err[m])),
            "p90_abs_error_deg": float(np.percentile(err[m], 90)),
            "max_abs_error_deg": float(np.max(err[m])),
            "mean_epistemic": float(np.mean(d["epistemic"][m])),
            "mean_aleatoric": float(np.mean(d["aleatoric"][m])),
            "n_alpha_floor_bound": int(d["alpha_floor_bound"][m].sum()),
            "n_err_over_90deg": int((err[m] > 90).sum()),
            "mesh_excluded_overlap": in_excluded(lo, hi),
        })
    return rows


def make_figure(d, rows):
    """Same format as F's figure so the two can sit side by side.
    Saved as PNG only via make_figures.save."""
    b = {k: np.array([r[k] for r in rows], dtype=np.float64)
         for k in ("bin_centre_deg", "mean_abs_error_deg", "p90_abs_error_deg",
                   "mean_epistemic")}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    for i, (lo, hi) in enumerate(MESH_EXCLUDED):
        ax.axvspan(lo, hi, color="#d8dce2", alpha=0.55, lw=0, zorder=0,
                   label="mesh-excluded range" if i == 0 else None)
    ax.scatter(d["true_angle_deg"], np.abs(d["circular_error_deg"]), s=4,
               color=GREY, alpha=0.45, lw=0, zorder=2, label="render")
    ax.plot(b["bin_centre_deg"], b["mean_abs_error_deg"], "o-", color=INK,
            ms=3.5, lw=1.3, zorder=4, label="bin mean")
    ax.plot(b["bin_centre_deg"], b["p90_abs_error_deg"], "^--", color=RED,
            ms=3, lw=1.0, zorder=3, label="bin p90")
    ax.set_xlabel("true angle (deg)")
    ax.set_ylabel("absolute circular error (deg)")
    ax.set_xlim(-180, 180)
    ax.set_xticks(range(-180, 181, 45))
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)

    ax2 = ax.twinx()
    ax2.plot(b["bin_centre_deg"], b["mean_epistemic"], "s:", color=BLUE, ms=3,
             lw=1.1, zorder=5, label="bin mean epistemic")
    ax2.set_ylabel("mean epistemic uncertainty", color=BLUE, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=BLUE, labelsize=7)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper left",
              ncol=2)
    fig.tight_layout()
    save(fig, "section5_1_error_by_true_angle_mesh_instance")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, base_image = locate_instance()
    print("mesh instance: subset position %d = MNIST train index %d, partition=%s"
          % (meta["subset_position"], meta["mnist_train_index"], meta["partition"]))

    d = evaluate_instance(base_image, device)
    OUT.mkdir(parents=True, exist_ok=True)
    per_render = [{
        "render_index": i,
        "true_angle_deg": float(d["true_angle_deg"][i]),
        "gamma": float(d["gamma"][i]),
        "circular_error_deg": float(d["circular_error_deg"][i]),
        "nu": float(d["nu"][i]), "alpha": float(d["alpha"][i]),
        "beta": float(d["beta"][i]),
        "epistemic": float(d["epistemic"][i]),
        "aleatoric": float(d["aleatoric"][i]),
        "total": float(d["total"][i]),
        "alpha_floor_bound": int(d["alpha_floor_bound"][i]),
    } for i in range(len(d["true_angle_deg"]))]
    write_rows_csv(per_render, OUT / "per_render.csv")

    rows = bin_table(d)
    write_rows_csv(rows, OUT / "error_by_true_angle.csv")

    # ── elevation defined against the MEDIAN bin, not an absolute threshold ──
    ae = np.array([r["mean_abs_error_deg"] for r in rows])
    ep = np.array([r["mean_epistemic"] for r in rows])
    med_ae, med_ep = float(np.median(ae)), float(np.median(ep))
    for r in rows:
        r["ae_ratio_to_median"] = r["mean_abs_error_deg"] / med_ae
        r["epi_ratio_to_median"] = r["mean_epistemic"] / med_ep
    elevated = [r for r in rows if r["ae_ratio_to_median"] >= 2.0]

    def covers(rng):
        lo, hi = rng
        hits = [r for r in elevated if r["bin_lo_deg"] < hi and r["bin_hi_deg"] > lo]
        return {"range": list(rng), "contains_elevated_bins": bool(hits),
                "elevated_bins_inside": [[r["bin_lo_deg"], r["bin_hi_deg"]] for r in hits]}

    mirror = [r for r in rows if -135.0 <= r["bin_lo_deg"] < -105.0]
    peak = max(rows, key=lambda r: r["mean_abs_error_deg"])

    summary = {
        "instance": meta,
        "n_renders": N_RENDERS,
        "renders_per_bin": N_RENDERS // 24,
        "bin_width_deg": BIN_DEG,
        "render_pipeline": "invert -> make_green_digits -> TF.rotate(fill=1.0); "
                           "no colour filter, no target offset -- matches "
                           "GridEnvironment.get_cell_input with "
                           "apply_colour_filter=False",
        "sampling_variance": {
            "applies": False,
            "note": "Single fixed image, deterministic evenly-spaced angles, "
                    "frozen checkpoint, model.eval(): output is bit-identical "
                    "on re-run. The 20-seed protocol averages "
                    "RotatedDigitDataset's per-pass angle re-roll, which does "
                    "not exist in this design. These are NOT single-pass "
                    "results of the kind that protocol replaced.",
        },
        "median_bin": {"mean_abs_error_deg": med_ae, "mean_epistemic": med_ep},
        "elevation_criterion": "mean_abs_error_deg >= 2x the median bin",
        "peak_bin": {"bin": [peak["bin_lo_deg"], peak["bin_hi_deg"]],
                     "mean_abs_error_deg": peak["mean_abs_error_deg"],
                     "ae_ratio_to_median": peak["ae_ratio_to_median"],
                     "mean_epistemic": peak["mean_epistemic"],
                     "epi_ratio_to_median": peak["epi_ratio_to_median"]},
        "elevated_bins": [
            {"bin": [r["bin_lo_deg"], r["bin_hi_deg"]],
             "mean_abs_error_deg": r["mean_abs_error_deg"],
             "ae_ratio_to_median": r["ae_ratio_to_median"],
             "mean_epistemic": r["mean_epistemic"],
             "epi_ratio_to_median": r["epi_ratio_to_median"],
             "mesh_excluded_overlap": r["mesh_excluded_overlap"]}
            for r in elevated],
        "mesh_excluded_ranges_assessment": [covers(t) for t in MESH_EXCLUDED],
        "elevated_bins_outside_all_exclusions": [
            [r["bin_lo_deg"], r["bin_hi_deg"]] for r in elevated
            if not r["mesh_excluded_overlap"]],
        "population_band_from_F": [90.0, 135.0],
        "mirror_bins_minus105_to_minus135": [
            {"bin": [r["bin_lo_deg"], r["bin_hi_deg"]],
             "mean_abs_error_deg": r["mean_abs_error_deg"],
             "ae_ratio_to_median": r["ae_ratio_to_median"],
             "mean_epistemic": r["mean_epistemic"],
             "epi_ratio_to_median": r["epi_ratio_to_median"]} for r in mirror],
        "alpha_floor_bound_total": int(d["alpha_floor_bound"].sum()),
        "n_params": int(d["n_params"]),
    }
    with open(OUT / "summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)

    hdr = ("%16s %5s %8s %8s %8s %11s %11s %6s %5s"
           % ("bin", "n", "meanAE", "p90", "max", "epi", "ale", "xmed", "excl"))
    print("\n" + hdr)
    for r in rows:
        print("[%+6.0f,%+6.0f) %5d %8.2f %8.2f %8.2f %11.6f %11.6f %6.2f %5s"
              % (r["bin_lo_deg"], r["bin_hi_deg"], r["n"],
                 r["mean_abs_error_deg"], r["p90_abs_error_deg"],
                 r["max_abs_error_deg"], r["mean_epistemic"],
                 r["mean_aleatoric"], r["ae_ratio_to_median"],
                 "YES" if r["mesh_excluded_overlap"] else ""))
    print("\nmedian bin: meanAE=%.2f deg  epistemic=%.6f" % (med_ae, med_ep))
    print("alpha-floor bindings total: %d / %d" % (summary["alpha_floor_bound_total"], N_RENDERS))

    make_figure(d, rows)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Half-degree rotation sweep of the mesh digit instance (GPU, ~2 min)")
    _p.parse_args()
    main()
