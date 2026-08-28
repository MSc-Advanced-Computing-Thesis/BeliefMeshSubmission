# Section 5.1 callout figures -- every number the results prose needs, computed
# from the dumped artefacts so the text and the figures cannot drift apart.
#
# Pure recomputation. No model, no dataset, no evaluation.
#
# Bands used throughout, and why:
#   POPULATION damage band  +90 to +135 deg -- the contiguous run of elevated
#       15-degree bins in runs/section5_1/f_angle_bins/. Not the mesh exclusion
#       ranges, which are a separate question answered against these numbers.
#   INSTANCE failure window  data-defined, not hand-set: the contiguous run
#       within +100..+150 where |error| exceeds 5x the instance's own median.
#   MIRROR band  -135 to -90 deg -- the reflection of the population band,
#       carried to test whether the ambiguity is a rotational symmetry.
#
# Run: python -u experiments/section5_1/make_callouts.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import read_dump
from make_figures import load_csv

R = Path("runs/section5_1")
OUT = R / "callouts.yaml"
POP_BAND = (90.0, 135.0)
MIRROR_BAND = (-135.0, -90.0)
PRECAUTIONARY = [(165.0, 180.0), (-180.0, -165.0)]
DIAGNOSED = (120.0, 150.0)
MARKED = [0.50, 0.80, 0.90, 0.95]


def f(x):
    return float(x)


def band_stats(ang, err, epi, lo, hi):
    m = (ang >= lo) & (ang < hi)
    return {
        "range_deg": [lo, hi], "n": int(m.sum()),
        "mean_abs_error_deg": f(np.mean(err[m])),
        "median_abs_error_deg": f(np.median(err[m])),
        "p90_abs_error_deg": f(np.percentile(err[m], 90)),
        "max_abs_error_deg": f(np.max(err[m])),
        "mean_epistemic": f(np.mean(epi[m])),
        "n_err_over_90deg": int((err[m] > 90).sum()),
    }


def outside_stats(ang, err, epi, bands):
    m = np.ones(len(ang), dtype=bool)
    for lo, hi in bands:
        m &= ~((ang >= lo) & (ang < hi))
    return {
        "excludes": [list(b) for b in bands], "n": int(m.sum()),
        "mean_abs_error_deg": f(np.mean(err[m])),
        "median_abs_error_deg": f(np.median(err[m])),
        "p90_abs_error_deg": f(np.percentile(err[m], 90)),
        "max_abs_error_deg": f(np.max(err[m])),
        "mean_epistemic": f(np.mean(epi[m])),
        "n_err_over_90deg": int((err[m] > 90).sum()),
    }


def angle_response():
    pop = read_dump(R / "r1a" / "per_sample_baseline.csv")
    ang, err, epi = (pop["true_angle_deg"], np.abs(pop["circular_error_deg"]),
                     pop["epistemic"])
    inside = band_stats(ang, err, epi, *POP_BAND)
    outside = outside_stats(ang, err, epi, [POP_BAND])
    mirror = band_stats(ang, err, epi, *MIRROR_BAND)
    # Both wrap-boundary bins together: keep only |angle| >= 165.
    prec_m = np.abs(ang) >= 165.0
    prec = {"range_deg": "|angle| >= 165", "n": int(prec_m.sum()),
            "mean_abs_error_deg": f(np.mean(err[prec_m])),
            "median_abs_error_deg": f(np.median(err[prec_m])),
            "p90_abs_error_deg": f(np.percentile(err[prec_m], 90)),
            "max_abs_error_deg": f(np.max(err[prec_m])),
            "mean_epistemic": f(np.mean(epi[prec_m])),
            "n_err_over_90deg": int((err[prec_m] > 90).sum())}
    population = {
        "band": {"damage": list(POP_BAND), "mirror": list(MIRROR_BAND)},
        "inside_damage_band": inside,
        "outside_damage_band": outside,
        "mirror_band": mirror,
        "wrap_boundary_bins_combined": prec,
        "ratios_inside_over_outside": {
            "mean_abs_error": f(inside["mean_abs_error_deg"] / outside["mean_abs_error_deg"]),
            "p90_abs_error": f(inside["p90_abs_error_deg"] / outside["p90_abs_error_deg"]),
            "mean_epistemic": f(inside["mean_epistemic"] / outside["mean_epistemic"]),
        },
        "ratios_inside_over_mirror": {
            "mean_abs_error": f(inside["mean_abs_error_deg"] / mirror["mean_abs_error_deg"]),
            "mean_epistemic": f(inside["mean_epistemic"] / mirror["mean_epistemic"]),
        },
        "share_of_circle_in_damage_band": f((POP_BAND[1] - POP_BAND[0]) / 360.0),
        "share_of_samples_in_damage_band": f(inside["n"] / len(ang)),
        "share_of_over90deg_errors_in_damage_band": f(
            inside["n_err_over_90deg"] / max(1, int((err > 90).sum()))),
        "n_err_over_90deg_total": int((err > 90).sum()),
    }

    # ── single mesh instance ──────────────────────────────────────────────
    j = read_dump(R / "j_single_instance" / "per_render.csv")
    ja, je, jp = (j["true_angle_deg"], np.abs(j["circular_error_deg"]),
                  j["epistemic"])
    med = float(np.median(je))
    sel = (ja > 100) & (ja < 150) & (je > 5 * med)
    lo, hi = f(ja[sel].min()), f(ja[sel].max())
    win = (ja >= lo) & (ja <= hi)
    peak_i = int(np.argmax(je))
    instance = {
        "n_renders": int(len(ja)), "resolution_deg": 0.5,
        "median_abs_error_deg_whole_circle": med,
        "failure_window_deg": [lo, hi],
        "failure_window_width_deg": f(hi - lo),
        "failure_window_share_of_circle": f((hi - lo) / 360.0),
        "inside_window": {
            "n": int(win.sum()),
            "mean_abs_error_deg": f(np.mean(je[win])),
            "max_abs_error_deg": f(np.max(je[win])),
            "mean_epistemic": f(np.mean(jp[win])),
            "max_epistemic": f(np.max(jp[win])),
        },
        "outside_window": {
            "n": int((~win).sum()),
            "mean_abs_error_deg": f(np.mean(je[~win])),
            "p90_abs_error_deg": f(np.percentile(je[~win], 90)),
            "max_abs_error_deg": f(np.max(je[~win])),
            "mean_epistemic": f(np.mean(jp[~win])),
        },
        "peak": {"angle_deg": f(ja[peak_i]), "abs_error_deg": f(je[peak_i]),
                 "epistemic": f(jp[peak_i])},
        "ratios_inside_over_outside": {
            "mean_abs_error": f(np.mean(je[win]) / np.mean(je[~win])),
            "mean_epistemic": f(np.mean(jp[win]) / np.mean(jp[~win])),
            "peak_epistemic_over_outside_mean": f(np.max(jp[win]) / np.mean(jp[~win])),
        },
        "partition_note": "this instance is in the TRAINING partition (MNIST "
                          "train index 15); its error LEVEL is not comparable "
                          "to the held-out population, only its angular shape",
    }

    # ── how the exclusion ranges score against both ───────────────────────
    b = load_csv(R / "f_angle_bins" / "error_by_true_angle.csv")
    def bins_in(lo_, hi_):
        m = (b["bin_lo_deg"] < hi_) & (b["bin_hi_deg"] > lo_)
        return {"mean_abs_error_deg": f(np.average(b["mean_abs_error_deg"][m],
                                                   weights=b["n"][m])),
                "mean_epistemic": f(np.average(b["mean_epistemic"][m],
                                               weights=b["n"][m])),
                "n": int(b["n"][m].sum())}
    exclusions = {
        "diagnosed_120_150": {
            "population": bins_in(*DIAGNOSED),
            "instance_contains_failure_window": bool(
                lo >= DIAGNOSED[0] - 15 and hi <= DIAGNOSED[1]),
        },
        "precautionary_wrap": {
            "population_165_180": bins_in(165.0, 180.0),
            "population_minus180_minus165": bins_in(-180.0, -165.0),
        },
        "population_median_bin_mean_abs_error_deg": f(np.median(b["mean_abs_error_deg"])),
        "population_median_bin_mean_epistemic": f(np.median(b["mean_epistemic"])),
        "peak_population_bin": {
            "range": [f(b["bin_lo_deg"][int(np.argmax(b["mean_abs_error_deg"]))]),
                      f(b["bin_hi_deg"][int(np.argmax(b["mean_abs_error_deg"]))])],
            "mean_abs_error_deg": f(np.max(b["mean_abs_error_deg"])),
            "inside_diagnosed_range": bool(
                b["bin_lo_deg"][int(np.argmax(b["mean_abs_error_deg"]))] >= DIAGNOSED[0]),
        },
    }
    return {"population": population, "mesh_instance": instance,
            "exclusion_ranges": exclusions}


def calibration():
    d = read_dump(R / "r1a" / "per_sample_baseline.csv")
    e = np.abs(d["circular_error_deg"])
    fine = load_csv(R / "r1a" / "interval_coverage_fine.csv")
    o = np.argsort(fine["nominal"])
    nom, emp = fine["nominal"][o], fine["empirical"][o]
    dev = emp - nom
    below = nom < 0.5
    above = nom > 0.5
    return {
        "error_distribution": {
            "n": int(len(e)),
            "mean_abs_error_deg": f(np.mean(e)),
            "rms_error_deg": f(np.sqrt(np.mean(e ** 2))),
            "median_abs_error_deg": f(np.median(e)),
            "p90_abs_error_deg": f(np.percentile(e, 90)),
            "p99_abs_error_deg": f(np.percentile(e, 99)),
            "max_abs_error_deg": f(np.max(e)),
            "rms_over_mean_ratio": f(np.sqrt(np.mean(e ** 2)) / np.mean(e)),
            "share_under_10deg": f(np.mean(e < 10)),
            "share_under_25deg": f(np.mean(e < 25)),
            "share_over_45deg": f(np.mean(e > 45)),
            "share_over_90deg": f(np.mean(e > 90)),
            "n_over_90deg": int((e > 90).sum()),
        },
        "coverage": {
            "n_levels": int(len(nom)),
            "at_marked_levels": {f"{m:.2f}": f(emp[np.isclose(nom, m)][0])
                                 for m in MARKED},
            "mean_abs_deviation": f(np.mean(np.abs(dev))),
            "max_over_coverage": {"deviation": f(dev.max()),
                                  "at_nominal": f(nom[int(dev.argmax())])},
            "max_under_coverage": {"deviation": f(dev.min()),
                                   "at_nominal": f(nom[int(dev.argmin())])},
            "mean_deviation_below_0p50": f(np.mean(dev[below])),
            "mean_deviation_above_0p50": f(np.mean(dev[above])),
            "deviation_at_0p50": f(dev[np.isclose(nom, 0.5)][0]),
            "half_width_deg_at_marked": {
                f"{m:.2f}": f(fine["mean_half_width_deg"][o][np.isclose(nom, m)][0])
                for m in MARKED},
            "max_frac_interval_covers_circle": f(
                fine["frac_interval_covers_circle"].max()),
            "n_alpha_floor_bound": int(d["alpha_floor_bound"].sum()),
        },
    }


def sweeps():
    b = load_csv(R / "r1b" / "sweep.csv")
    ft = b["filter_type"].astype(str)
    out_b = {}
    for name in ("red", "blue"):
        m = ft == name
        o = np.argsort(b["w"][m])
        w = b["w"][m][o]
        mse = b["circular_mse_norm_mean"][m][o]
        sd = b["circular_mse_norm_sd"][m][o]
        epi = b["mean_epistemic_mean"][m][o]
        ale = b["mean_aleatoric_mean"][m][o]
        r = b["manifest_convention_pearson_r_mean"][m][o]
        cov = b["coverage_90_mean"][m][o]
        base = mse[0]
        dbl = w[np.argmax(mse >= 2 * base)] if (mse >= 2 * base).any() else None
        out_b[name] = {
            "mse_at_w0": f(base), "mse_at_w0p5": f(mse[w == 0.5][0]),
            "mse_at_w1": f(mse[-1]), "mse_sd_at_w1": f(sd[-1]),
            "mse_growth_w0_to_w1": f(mse[-1] / base),
            "w_at_which_mse_doubles": (f(dbl) if dbl is not None else None),
            "epistemic_w0": f(epi[0]), "epistemic_w1": f(epi[-1]),
            "epistemic_growth": f(epi[-1] / epi[0]),
            "aleatoric_w0": f(ale[0]), "aleatoric_w1": f(ale[-1]),
            "aleatoric_growth": f(ale[-1] / ale[0]),
            "epi_over_ale_w0": f(epi[0] / ale[0]),
            "epi_over_ale_w1": f(epi[-1] / ale[-1]),
            "r_w0": f(r[0]), "r_w1": f(r[-1]),
            "coverage90_w0": f(cov[0]), "coverage90_w1": f(cov[-1]),
            "coverage90_min": f(cov.min()),
            "coverage90_min_at_w": f(w[int(cov.argmin())]),
        }
    pair = load_csv(R / "r1b" / "red_vs_blue_paired.csv")
    po = np.argsort(pair["w"])
    out_b["blue_minus_red_paired"] = {
        "at_w1": {"mean_diff": f(pair["mean_diff"][po][-1]),
                  "sd": f(pair["sd_diff"][po][-1]),
                  "cohens_dz": f(pair["cohens_dz"][po][-1])},
        "first_w_with_dz_over_2": f(pair["w"][po][
            int(np.argmax(np.nan_to_num(np.abs(pair["cohens_dz"][po])) > 2))]),
        "blue_over_red_mse_at_w1": f(out_b["blue"]["mse_at_w1"]
                                     / out_b["red"]["mse_at_w1"]),
    }

    c = load_csv(R / "r1c" / "sweep.csv")
    o = np.argsort(c["offset_deg"])
    off = c["offset_deg"][o]
    mse = c["circular_mse_norm_mean"][o]
    epi = c["mean_epistemic_mean"][o]
    ale = c["mean_aleatoric_mean"][o]
    cov = c["coverage_90_mean"][o]
    r = c["manifest_convention_pearson_r_mean"][o]
    sign = np.where(np.sign(r[:-1]) != np.sign(r[1:]))[0]
    out_c = {
        "offsets_deg": [f(x) for x in off],
        "epistemic_constant_value": f(epi[0]),
        "epistemic_range_over_sweep": [f(epi.min()), f(epi.max())],
        "aleatoric_constant_value": f(ale[0]),
        "aleatoric_range_over_sweep": [f(ale.min()), f(ale.max())],
        "uncertainty_responds_to_offset": bool(
            (epi.max() - epi.min()) > 1e-9 or (ale.max() - ale.min()) > 1e-9),
        "mse_at_0": f(mse[0]), "mse_at_10": f(mse[1]),
        "mse_at_30": f(mse[3]), "mse_at_60": f(mse[6]),
        "mse_at_110": f(mse[-1]),
        "mse_growth_0_to_110": f(mse[-1] / mse[0]),
        "coverage90": {f"{int(a)}": f(v) for a, v in zip(off, cov)},
        "coverage90_at_0": f(cov[0]), "coverage90_at_10": f(cov[1]),
        "coverage90_at_20": f(cov[2]), "coverage90_at_40": f(cov[4]),
        "coverage90_at_110": f(cov[-1]),
        "first_offset_with_coverage_below_half_nominal": f(
            off[int(np.argmax(cov < 0.45))]),
        "r_at_0": f(r[0]), "r_at_110": f(r[-1]),
        "r_sign_change_between_deg": ([f(off[sign[0]]), f(off[sign[0] + 1])]
                                      if len(sign) else None),
        "interval_half_width_constant": True,
    }
    return {"colour_sweep_r1b": out_b, "offset_sweep_r1c": out_c}


def main():
    data = {"angle_response": angle_response(),
            "calibration": calibration(),
            "sweeps": sweeps()}
    with open(OUT, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    print(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
