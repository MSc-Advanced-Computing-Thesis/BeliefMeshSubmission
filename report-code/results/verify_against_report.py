"""Compare the regenerated tables and figure numbers against the values printed
in the report (MSc_Advanced_Computing_Final_Report, v47).

Reads results/tables/*.csv and *.txt written by make_all.py and writes
VERIFICATION.md at the repository root: one row per reported quantity, the
report's value, the regenerated value, and whether they agree to the report's
printed precision.

Run after make_all.py:  python results/verify_against_report.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent
TAB = RESULTS / "tables"
FIG = RESULTS / "figures"
OUT = RESULTS.parent / "VERIFICATION.md"

rows = []   # (item, quantity, report value, regenerated value, status, note)


def read_csv(name):
    with open(TAB / name, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def check(item, quantity, report, got, tol=None, note=""):
    """report and got are floats; agree if |diff| <= tol (default: half a unit of
    the report's last printed digit)."""
    if got is None:
        rows.append((item, quantity, report, "missing", "MISSING", note)); return
    if tol is None:
        s = ("%r" % report)
        dec = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** (-dec) + 1e-12
    ok = abs(float(got) - float(report)) <= tol
    rows.append((item, quantity, report, got, "match" if ok else "DIFFERS", note))


def find(rows_, key, value):
    for r in rows_:
        if r[key] == value:
            return r
    return None


def num(s):
    return float(re.sub(r"[^0-9.+-eE]", "", s))


def main():
    # ---- Table 5.1 (two-contributor cells)
    try:
        t = read_csv("table_5_1_three_node_aggregation.csv")
        REP = {"Frozen": (0.08146, 0.08114, 0.837, 1.408, -0.738), "Naive": (0.00949, 0.00070, 0.973, 1.906, -0.384),
               "Certainty": (0.00949, 0.00072, 0.972, 1.889, -0.381), "Product fusion": (0.00951, 0.00071, 0.973, 1.910, -0.372),
               "Average fusion": (0.00988, 0.00071, 0.980, 1.983, -0.376)}
        for arm, v in REP.items():
            r = find(t, "arm", arm)
            for k, rv in zip(("whole_run_mse_mean", "last50_mse_mean", "coverage90_mean", "hw_rms_mean", "r_nu_error_mean"), v):
                check("Table 5.1", "%s %s" % (arm, k), rv, float(r[k]) if r else None)
    except FileNotFoundError as e:
        rows.append(("Table 5.1", "-", "-", "-", "MISSING", str(e)))

    # ---- Table 5.2
    try:
        t = read_csv("table_5_2_aggregation_mesh_scale.csv")
        REP = {"Frozen": (0.09697, 0.09878, 0.815, 1.349), "Naive": (0.02522, 0.01451, 0.723, 1.067),
               "Certainty": (0.02521, 0.01458, 0.725, 1.064), "Product fusion": (0.02516, 0.01468, 0.720, 1.046),
               "Average fusion": (0.03431, 0.01368, 0.981, 3.053)}
        for arm, v in REP.items():
            r = find(t, "arm", arm)
            for k, rv in zip(("whole_run_mse_mean", "last50_mse_mean", "coverage90_mean", "hw_rms_mean"), v):
                check("Table 5.2", "%s %s" % (arm, k), rv, float(r[k]) if r else None)
    except FileNotFoundError as e:
        rows.append(("Table 5.2", "-", "-", "-", "MISSING", str(e)))

    # ---- Table 5.3 (as published, and the corrected readout)
    REP = {"FedAvg": (0.02971, 0.02168, 1560080, 1560080), "Gossip uniform": (0.02942, 0.02143, 1560080, 10350731),
           "Gossip weighted": (0.02968, 0.02243, 1560080, 10350731), "Product fusion": (0.03641, 0.02433, 784, 4988)}
    for suffix, label in (("_as_published", "Table 5.3 (as published)"), ("", "Table 5.3 (each run vs its own field)")):
        try:
            t = read_csv("table_5_3_parameter_exchange%s.csv" % suffix)
            for arm, v in REP.items():
                r = find(t, "arm", arm)
                note = "" if suffix else "report scored all environments against field0; see BUILD_LOG.md"
                check(label, "%s whole_run_mse" % arm, v[0], float(r["whole_run_mse_mean"]) if r else None, note=note)
                check(label, "%s last50_mse" % arm, v[1], float(r["last50_mse_mean"]) if r else None, note=note)
                if not suffix:
                    check(label, "%s transmitted bytes" % arm, v[2], float(r["transmitted_bytes_per_node_step"]) if r else None, tol=0.5)
                    check(label, "%s received bytes" % arm, v[3], float(r["received_bytes_per_node_step"]) if r else None, tol=0.5,
                          note="report's received figure for Product fusion (4,988) and FedAvg (1,560,080) are not the per-node manifest means; see BUILD_LOG.md")
        except FileNotFoundError as e:
            rows.append((label, "-", "-", "-", "MISSING", str(e)))

    # ---- Table 5.4
    try:
        t = read_csv("table_5_4_hop_calibration.csv")
        REP = {"0": (127, 1.714, 22.35, 13.57), "1": (279, 1.169, 22.46, 20.12), "2": (76, 0.732, 23.16, 32.26)}
        for hop, v in REP.items():
            r = find(t, "hop", hop)
            check("Table 5.4", "hop %s cells/step" % hop, v[0], float(r["cells_per_step"]) if r else None, tol=0.5)
            for k, rv in zip(("hw_rms_mean", "half_width_deg_mean", "rms_error_deg_mean"), v[1:]):
                check("Table 5.4", "hop %s %s" % (hop, k), rv, float(r[k]) if r else None)
    except FileNotFoundError as e:
        rows.append(("Table 5.4", "-", "-", "-", "MISSING", str(e)))

    # ---- Table 5.5
    try:
        t = read_csv("table_5_5_mixed_vs_uniform.csv")
        REP = {"Mixed capacity": (0.05073, 0.01913, 0.843, 1.463), "Uniform narrow": (0.04955, 0.02601, 0.768, 1.006),
               "Uniform baseline": (0.02516, 0.01468, 0.719, 1.044), "Uniform wide": (0.03475, 0.02029, 0.746, 1.102)}
        for arm, v in REP.items():
            r = find(t, "configuration", arm)
            for k, rv in zip(("whole_run_mse_mean", "last50_mse_mean", "coverage90_mean", "hw_rms_mean"), v):
                check("Table 5.5", "%s %s" % (arm, k), rv, float(r[k]) if r else None)
    except FileNotFoundError as e:
        rows.append(("Table 5.5", "-", "-", "-", "MISSING", str(e)))

    # ---- Table 5.6 / 5.7
    try:
        t = read_csv("table_5_6_miscalibration.csv")
        for arm, v in {"Baseline": (0.02516, 0.01468, 0.720, 1.046), "Miscalibrated": (0.02912, 0.01698, 0.766, 1.237)}.items():
            r = find(t, "condition", arm)
            for k, rv in zip(("whole_run_mse_mean", "last50_mse_mean", "coverage90_mean", "hw_rms_mean"), v):
                check("Table 5.6", "%s %s" % (arm, k), rv, float(r[k]) if r else None)
        t = read_csv("table_5_7_miscalibration_by_distance.csv")
        REP = {"0-2": (236, 0.00627, 1.307, 1.427, 41.7), "2-4": (88, 0.00074, 1.185, 1.430, 24.0), "4-6": (58, -0.00273, 0.978, 1.188, 15.2),
               "6-9": (58, -0.00482, 0.836, 0.972, 8.9), "9+": (43, -0.00739, 1.058, 1.273, 3.4)}
        for d, v in REP.items():
            r = find(t, "distance_cells", d)
            check("Table 5.7", "%s cells" % d, v[0], float(r["cells"]) if r else None, tol=0.5)
            for k, rv in zip(("mse_difference_mean", "hw_rms_baseline", "hw_rms_miscalibrated", "half_width_change_pct"), v[1:]):
                check("Table 5.7", "%s %s" % (d, k), rv, float(r[k]) if r else None)
    except FileNotFoundError as e:
        rows.append(("Table 5.6/5.7", "-", "-", "-", "MISSING", str(e)))

    # ---- Table 5.8
    try:
        t = read_csv("table_5_8_ablation_summary.csv")
        REP = {"Regularisation weight lambda": (0.02369, 0.03087, 0.00165, "Effect"), "Consensus rho": (0.02521, 0.02530, 0.00166, "Null"),
               "Weighted fusion": (0.02516, 0.02524, 0.00173, "Null"), "Uncertainty measure": (0.02514, 0.02517, 0.00168, "Null"),
               "Gradient tempering": (0.02066, 0.02067, 0.01077, "Null")}
        for fam, v in REP.items():
            r = find(t, "family", fam)
            note = "report scored the five environment-paired runs against field0; see BUILD_LOG.md" if fam == "Gradient tempering" else ""
            check("Table 5.8", "%s whole-run min" % fam, v[0], float(r["whole_run_mse_min"]) if r else None, note=note)
            check("Table 5.8", "%s whole-run max" % fam, v[1], float(r["whole_run_mse_max"]) if r else None, note=note)
            check("Table 5.8", "%s seed sd" % fam, v[2], float(r["mean_seed_sd"]) if r else None, note=note)
            rows.append(("Table 5.8", "%s outcome" % fam, v[3], r["outcome"] if r else "missing",
                         "match" if r and r["outcome"] == v[3] else "DIFFERS", ""))
    except FileNotFoundError as e:
        rows.append(("Table 5.8", "-", "-", "-", "MISSING", str(e)))

    # ---- Table C.2
    try:
        t = read_csv("table_C_2_deployment_performance.csv")
        REP = {"Whole-run MSE": (0.01858, 0.02516, 0.00173, 0.02307, 0.02790), "Last-50 MSE": (0.01100, 0.01468, 0.00299, 0.01122, 0.01785),
               "Coverage": (0.815, 0.720, 0.064, 0.658, 0.827), "hw:RMS": (1.541, 1.046, 0.273, 0.876, 1.521)}
        for m, v in REP.items():
            r = find(t, "measure", m)
            for k, rv in zip(("deployment_66x66", "reference_mean", "reference_sd", "reference_min", "reference_max"), v):
                check("Table C.2", "%s %s" % (m, k), rv, float(r[k]) if r else None)
    except FileNotFoundError as e:
        rows.append(("Table C.2", "-", "-", "-", "MISSING", str(e)))
    # Table C.1: geometry only (wall times are from the run logs)
    try:
        t = read_csv("table_C_1_deployment_configuration.csv")
        for m, v in {"Deployment area (cells)": ("484", "4356"), "Cells per node": ("49", "441"), "Mean coverage depth": ("3.64", "3.64"),
                     "Mean overlap degree": ("15.0", "15.0"), "Wearables": ("3", "27")}.items():
            r = find(t, "measure", m)
            got = (r["reference_22x22"], r["deployment_66x66"]) if r else None
            rows.append(("Table C.1", m, "%s / %s" % v, "%s / %s" % got if got else "missing",
                         "match" if got == v else "DIFFERS", ""))
        rows.append(("Table C.1", "wall time, stored arrays", "9.2 min / 67.3 min; 39.6 MB / 376.1 MB", "from run logs; 39.7 / 357.0 MB without disagreement log",
                     "n/a", "process wall time is not stored with the artefacts; array sizes exclude the disagreement log"))
    except FileNotFoundError as e:
        rows.append(("Table C.1", "-", "-", "-", "MISSING", str(e)))

    # ---- Table E.1
    try:
        t = read_csv("table_E_1_heterogeneous_capacity.csv")
        REP = {"Naive": (0.02653, 0.02938, 0.981, 3.072), "Certainty": (0.02563, 0.02674, 0.980, 3.080),
               "Product fusion": (0.01706, 0.01285, 0.859, 1.543), "Average fusion": (0.01695, 0.01306, 0.973, 2.631)}
        for arm, v in REP.items():
            r = find(t, "arm", arm)
            for k, rv in zip(("whole_run_mse_mean", "last50_mse_mean", "coverage90_mean", "hw_rms_mean"), v):
                check("Table E.1", "%s %s" % (arm, k), rv, float(r[k]) if r else None)
    except FileNotFoundError as e:
        rows.append(("Table E.1", "-", "-", "-", "MISSING", str(e)))

    # ---- Table F.1
    try:
        t = read_csv("table_F_1_lambda_sweep.csv")
        REP = {"1.0": (0.02369, 0.01455, 0.676, 0.902), "2.5": (0.02405, 0.01468, 0.694, 0.949), "5.0": (0.02515, 0.01478, 0.721, 1.046),
               "10.0": (0.02700, 0.01474, 0.810, 1.380), "25.0": (0.03087, 0.01464, 0.977, 3.337)}
        for lam, v in REP.items():
            r = find(t, "lambda", lam)
            for k, rv in zip(("whole_run_mse_mean", "last50_mse_mean", "coverage90_mean", "hw_rms_mean"), v):
                check("Table F.1", "lambda %s %s" % (lam, k), rv, float(r[k]) if r else None)
    except FileNotFoundError as e:
        rows.append(("Table F.1", "-", "-", "-", "MISSING", str(e)))

    # ---- Section 5.2 numbers (Figure 5.4)
    try:
        t = read_csv("table_5_4_peer_supervision.csv")
        REP = {"frozen": (0.02578, 0.10, 0.44), "direct": (0.00282, 0.90, 1.79), "student (mode target)": (0.00335, 0.14, 0.18),
               "student (sampled target)": (0.00403, 0.92, 1.95)}
        for arm, v in REP.items():
            r = find(t, "supervision", arm)
            note = "the report quotes coverage and hw:RMS to two decimals, which appear truncated rather than rounded"
            check("Fig 5.4 / Sec 5.2", "%s last-50 MSE" % arm, v[0], float(r["last50_fov_mse_mean"]) if r else None)
            check("Fig 5.4 / Sec 5.2", "%s coverage" % arm, v[1], float(r["coverage90_mean"]) if r else None, tol=0.011, note=note)
            check("Fig 5.4 / Sec 5.2", "%s hw:RMS" % arm, v[2], float(r["hw_rms_mean"]) if r else None, tol=0.011, note=note)
    except FileNotFoundError as e:
        rows.append(("Fig 5.4 / Sec 5.2", "-", "-", "-", "MISSING", str(e)))

    # ---- Section 5.6 numbers (Figure 5.9)
    try:
        t = read_csv("table_5_9_routing.csv")
        REP = {("Product fusion", "random"): (0.02516, 0.01473, 0.718, 1.043), ("Product fusion", "guided"): (0.02358, 0.00963, 0.866, 1.485),
               ("Average fusion", "random"): (None, 0.01390, 0.981, 3.026), ("Average fusion", "guided"): (None, 0.01115, 0.994, 3.716)}
        for (rule, pol), v in REP.items():
            r = next((x for x in t if x["fusion_rule"] == rule and x["routing"] == pol), None)
            for k, rv in zip(("whole_run_mse_mean", "last50_mse_mean", "coverage90_mean", "hw_rms_mean"), v):
                if rv is not None:
                    check("Fig 5.9 / Sec 5.6", "%s %s %s" % (rule, pol, k), rv, float(r[k]) if r else None)
    except FileNotFoundError as e:
        rows.append(("Fig 5.9 / Sec 5.6", "-", "-", "-", "MISSING", str(e)))

    # ---- Figure 5.10 numbers (text file)
    try:
        txt = (TAB / "figure_5_10_numbers.txt").read_text(encoding="utf8")
        pct = re.findall(r"heterogeneous vs homogeneous: whole-run ([+-]\d+)%, last-50 ([+-]\d+)%", txt)
        REP = [("Naive", 350, 490, 25), ("Certainty", 350, 490, 25), ("Product fusion", 102, 30, 3), ("Average fusion", 58, 45, 3)]
        for (arm, w, l, tol), got in zip(REP, pct):
            check("Fig 5.10 / Sec 5.7", "%s whole-run change %%" % arm, w, float(got[0]), tol=tol, note="report says 'around' for Naive/Certainty" if tol > 3 else "")
            check("Fig 5.10 / Sec 5.7", "%s last-50 change %%" % arm, l, float(got[1]), tol=tol, note="report says 'around' for Naive/Certainty" if tol > 3 else "")
        m = re.search(r"Average fusion\s+heterogeneous\s+\S+ \+/- \S+\s+\S+ \+/- \S+\s+\S+ \+/- \S+\s+(\S+) \+/-", txt)
        check("Fig 5.10 / Sec 5.7", "Average fusion heterogeneous hw:RMS", 3.987, float(m.group(1)) if m else None)
    except FileNotFoundError as e:
        rows.append(("Fig 5.10", "-", "-", "-", "MISSING", str(e)))

    # ---- Section 5.1 prose numbers
    try:
        txt = (TAB / "section_5_1_prose_numbers.txt").read_text(encoding="utf8")
        g = lambda pat: float(re.search(pat, txt).group(1))
        check("Sec 5.1", "single instance max error (deg)", 161.96, g(r"maximum error ([\d.]+) deg"))
        check("Sec 5.1", "epistemic at +124", 0.0098, g(r"at \+124 deg: epistemic ([\d.]+)"))
        check("Sec 5.1", "epistemic at +126", 0.0296, g(r"at \+126 deg: epistemic ([\d.]+)"))
        check("Sec 5.1", "epistemic at +129", 0.2683, g(r"at \+129 deg: epistemic ([\d.]+)"))
        check("Sec 5.1", "held-out mean abs error (deg)", 10.51, g(r"mean abs ([\d.]+) \+/-"))
        check("Sec 5.1", "held-out RMS (deg)", 20.97, g(r"RMS ([\d.]+) \+/-"))
        check("Sec 5.1", "epistemic-error Pearson r", 0.7194, g(r"Pearson r ([\d.]+)"))
        check("Sec 5.1", "coverage at nominal 0.50", 0.5004, g(r"nominal 0.50: ([\d.]+)"))
        check("Sec 5.1", "max coverage deviation above median", 0.031, g(r"max deviation above the median \+([\d.]+)"))
        check("Sec 5.1", "population epistemic peak", 0.0562, g(r"epistemic peak ([\d.]+)"))
        check("Sec 5.1", "peak / median bin", 18.3, g(r"\), ([\d.]+)x the median"))
        check("Sec 5.1", "full blue MSE", 0.25378, g(r"full blue filter: MSE ([\d.]+)"))
        check("Sec 5.1", "full red MSE", 0.11893, g(r"full red filter: MSE ([\d.]+)"))
        check("Sec 5.1", "epistemic/aleatoric ratio at w=0", 5.85, g(r"ratio ([\d.]+) at w=0"))
        check("Sec 5.1", "ratio at full blue", 2.80, g(r"blue filter: MSE [\d.]+; epistemic/aleatoric ratio [\d.]+ at w=0 -> ([\d.]+)"))
        check("Sec 5.1", "ratio at full red", 3.47, g(r"red filter: MSE [\d.]+; epistemic/aleatoric ratio [\d.]+ at w=0 -> ([\d.]+)"))
        check("Sec 5.1", "offset sweep MSE rise (x)", 27, g(r"\(([\d.]+)x\)"), tol=0.5)
        check("Sec 5.1", "coverage90 at 0 offset", 0.917, g(r"coverage90 ([\d.]+) ->"))
        check("Sec 5.1", "coverage90 at 40 deg", 0.079, g(r"-> ([\d.]+) at 40 deg"))
        check("Sec 5.1", "coverage90 at max offset", 0.022, g(r"at 40 deg -> ([\d.]+) at"))
        check("Sec 5.1", "mesh instance mean abs error outside excluded ranges (deg)", 3.89, g(r"excluded ranges: mean abs error ([\d.]+)"), tol=0.1,
              note="report 3.89 / p90 8.02; regenerated with all three excluded ranges removed")
    except (FileNotFoundError, AttributeError) as e:
        rows.append(("Sec 5.1", "-", "-", "-", "MISSING", str(e)))

    # ---- figures present
    for name, item in (("fig_3_2_fusion_rules.png", "Fig 3.2"), ("fig_4_2_environments.png", "Fig 4.2"), ("fig_4_3_mesh_geometry.png", "Fig 4.3"),
                       ("fig_5_1_angle_response.png", "Fig 5.1"), ("fig_5_2_calibration.png", "Fig 5.2"), ("fig_5_3a_colour_sweep.png", "Fig 5.3a"),
                       ("fig_5_3b_offset_sweep.png", "Fig 5.3b"), ("fig_5_4_peer_supervision.png", "Fig 5.4"), ("fig_5_5_three_node_geometry.png", "Fig 5.5"),
                       ("fig_5_6_mesh_snapshots.png", "Fig 5.6"), ("fig_5_7_exchange_trajectories.png", "Fig 5.7"), ("fig_5_8_mesh_density.png", "Fig 5.8"),
                       ("fig_5_9_routing.png", "Fig 5.9"), ("fig_5_10_heterogeneity.png", "Fig 5.10"), ("fig_5_11_coverage_maps.png", "Fig 5.11"),
                       ("fig_5_12_node_loss.png", "Fig 5.12"), ("fig_D_1_exchange_trajectories_all.png", "Fig D.1")):
        p = FIG / name
        rows.append((item, "file", name, "%d KB" % (p.stat().st_size // 1024) if p.exists() else "missing", "present" if p.exists() else "MISSING", ""))
    rows.append(("Fig 4.1", "file", "-", "-", "not regenerated", "no generator script exists in the working tree"))

    # ---- write
    n_match = sum(1 for r in rows if r[4] in ("match", "present"))
    n_diff = sum(1 for r in rows if r[4] == "DIFFERS")
    n_miss = sum(1 for r in rows if r[4] == "MISSING")
    md = ["# Verification against the report", "",
          "Regenerated by `results/make_all.py` and compared numerically by `results/verify_against_report.py`.",
          "A quantity matches when it agrees with the report to the report's printed precision.", "",
          "Summary: %d match/present, %d differ, %d missing, out of %d checks." % (n_match, n_diff, n_miss, len(rows)), "",
          "| item | quantity | report | regenerated | status | note |", "|---|---|---|---|---|---|"]
    for item, q, rep, got, st, note in rows:
        g = ("%.6g" % got) if isinstance(got, float) else str(got)
        md.append("| %s | %s | %s | %s | %s | %s |" % (item, q, rep, g, st, note))
    OUT.write_text("\n".join(md) + "\n", encoding="utf8")
    print("\n".join(md[:6]))
    for r in rows:
        if r[4] not in ("match", "present"):
            print("  %-28s %-55s report %-12s got %-12s %s  %s" % (r[0], r[1], r[2], ("%.6g" % r[3]) if isinstance(r[3], float) else r[3], r[4], r[5]))
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
