"""Section 5.1 -- Baseline Model and Calibration.

Regenerates, from the stored evaluation dumps in
artefacts/5.1_baseline_model_and_calibration/:

    Figure 5.1  fig_5_1_angle_response.png    (held-out population beside the
                                               mesh digit instance, error and
                                               epistemic uncertainty by angle)
    Figure 5.2  fig_5_2_calibration.png       (error distribution, interval
                                               coverage against nominal)
    Figure 5.3  fig_5_3a_colour_sweep.png, fig_5_3b_offset_sweep.png

and prints the quantities quoted in the section's prose (table_5_1_prose.txt).

--rerun re-evaluates the pretrained checkpoints (GPU, ~1 h) via the experiment
scripts in this directory, in the order r1a -> angle bins -> single instance
-> r1b -> r1c, writing fresh dumps into the artefact directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, FIGURES, TABLES  # noqa: E402

SECTION_ART = ARTEFACTS / "5.1_baseline_model_and_calibration"
RERUN_ORDER = ["experiment_r1a_held_out.py", "analysis_angle_bins.py",
               "experiment_single_instance.py", "experiment_r1b_colour_sweep.py",
               "experiment_r1c_offset_sweep.py"]


def prose_numbers(out_dir: Path) -> Path:
    """The numbers Section 5.1 quotes, recomputed from the dumps."""
    import numpy as np
    import yaml
    from analyse import read_dump
    from figure_sweeps import load_csv

    lines = []
    j = read_dump(SECTION_ART / "j_single_instance" / "per_render.csv")
    err = np.abs(j["circular_error_deg"])
    ang = j["true_angle_deg"]
    inside = ((ang >= 120) & (ang <= 150)) | (ang >= 165) | (ang <= -165)
    lines.append("Mesh digit instance, outside the excluded ranges: mean abs error %.2f deg, p90 %.2f deg"
                 % (err[~inside].mean(), np.percentile(err[~inside], 90)))
    k = int(np.argmax(err))
    lines.append("  maximum error %.2f deg at %.1f deg; epistemic there %.4f"
                 % (err[k], ang[k], j["epistemic"][k]))
    for a in (124.0, 126.0, 129.0):
        i = int(np.argmin(np.abs(ang - a)))
        lines.append("  at %+.0f deg: epistemic %.4f, abs error %.2f deg" % (ang[i], j["epistemic"][i], err[i]))

    ref = yaml.safe_load(open(SECTION_ART / "r1a_held_out_evaluation" / "summary.yaml"))
    rp = ref["reference_passes"]
    lines.append("Held-out population (20 passes): mean abs %.2f +/- %.2f deg, RMS %.2f +/- %.2f deg"
                 % (rp["mean_abs_error_deg"]["mean"], rp["mean_abs_error_deg"]["sd"],
                    rp["rms_error_deg"]["mean"], rp["rms_error_deg"]["sd"]))
    lines.append("  epistemic-vs-|error| Pearson r %.4f +/- %.4f"
                 % (rp["manifest_convention_pearson_r"]["mean"], rp["manifest_convention_pearson_r"]["sd"]))
    dec = load_csv(SECTION_ART / "r1a_held_out_evaluation" / "rms_by_uncertainty_decile.csv")
    lines.append("  RMS error by uncertainty decile: lowest %.1f deg, highest %.1f deg"
                 % (dec["rms_error_deg"][0], dec["rms_error_deg"][-1]))
    cov = load_csv(SECTION_ART / "r1a_held_out_evaluation" / "interval_coverage_fine.csv")
    dev = cov["empirical"] - cov["nominal"]
    i50 = int(np.argmin(np.abs(cov["nominal"] - 0.5)))
    lines.append("  coverage at nominal 0.50: %.4f; max deviation above the median %+.3f at nominal %.2f"
                 % (cov["empirical"][i50], dev[cov["nominal"] > 0.5].max(),
                    cov["nominal"][cov["nominal"] > 0.5][np.argmax(dev[cov["nominal"] > 0.5])]))
    b = load_csv(SECTION_ART / "f_angle_bins" / "error_by_true_angle.csv")
    lines.append("  population epistemic peak %.4f (bin centre %.0f deg), %.1fx the median bin"
                 % (b["mean_epistemic"].max(), b["bin_centre_deg"][np.argmax(b["mean_epistemic"])],
                    b["mean_epistemic"].max() / np.median(b["mean_epistemic"])))

    s = load_csv(SECTION_ART / "r1b_colour_sweep" / "sweep.csv")
    ft = s["filter_type"].astype(str)
    for name in ("blue", "red"):
        m = (ft == name)
        w1 = m & (np.abs(s["w"] - 1.0) < 1e-9)
        w0 = m & (np.abs(s["w"]) < 1e-9)
        lines.append("Colour sweep, full %s filter: MSE %.5f; epistemic/aleatoric ratio %.2f at w=0 -> %.2f at w=1"
                     % (name, s["circular_mse_norm_mean"][w1][0],
                        s["mean_epistemic_mean"][w0][0] / s["mean_aleatoric_mean"][w0][0],
                        s["mean_epistemic_mean"][w1][0] / s["mean_aleatoric_mean"][w1][0]))
    o = load_csv(SECTION_ART / "r1c_offset_sweep" / "sweep.csv")
    order = np.argsort(o["offset_deg"])
    x, mse, c90 = o["offset_deg"][order], o["circular_mse_norm_mean"][order], o["coverage_90_mean"][order]
    i40 = int(np.argmin(np.abs(x - 40)))
    lines.append("Offset sweep: MSE %.5f at 0 deg -> %.5f at %.0f deg (%.1fx); coverage90 %.3f -> %.3f at 40 deg -> %.3f at %.0f deg"
                 % (mse[0], mse[-1], x[-1], mse[-1] / mse[0], c90[0], c90[i40], c90[-1], x[-1]))
    lines.append("  epistemic range across offsets %.6f-%.6f, aleatoric %.6f-%.6f (constant to 6 s.f. expected)"
                 % (o["mean_epistemic_mean"].min(), o["mean_epistemic_mean"].max(),
                    o["mean_aleatoric_mean"].min(), o["mean_aleatoric_mean"].max()))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "section_5_1_prose_numbers.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))
    print("  wrote %s" % p)
    return p


def regenerate(fig_dir: Path, tab_dir: Path):
    import figure_5_1_angle_response as F51
    import figure_5_2_calibration as F52
    import figure_sweeps as FS
    F51.FIGS = fig_dir
    F52.FIGS = fig_dir
    FS.FIGS = fig_dir
    F51.main()
    F52.main()
    FS.fig3_colour_sweep()
    FS.fig4_offset_sweep()
    prose_numbers(tab_dir)


def rerun():
    py = sys.executable
    for script in RERUN_ORDER:
        print("\n=== %s ===" % script, flush=True)
        subprocess.run([py, "-u", str(HERE / script)], check=True, cwd=str(HERE))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES), help="output directory for figures")
    ap.add_argument("--tables", default=str(TABLES), help="output directory for tables / prose numbers")
    ap.add_argument("--rerun", action="store_true",
                    help="re-evaluate the checkpoints before regenerating (GPU, ~1 h)")
    a = ap.parse_args()
    if a.rerun:
        rerun()
    regenerate(Path(a.figures), Path(a.tables))
