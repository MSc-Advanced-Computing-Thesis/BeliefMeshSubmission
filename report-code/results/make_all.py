"""Regenerate every figure and table of the report from the stored artefacts.

    python results/make_all.py                 regenerate from artefacts (default, ~15 min)
    python results/make_all.py --only 5.4 D    a subset of sections
    python results/make_all.py --list          show the sections and what each produces

Re-running the experiments themselves is NEVER done by default. It requires
the explicit flag

    python results/make_all.py --rerun-experiments --i-understand-this-takes-days

and roughly forty hours of GPU time across the sections; each section's own
reproduce.py --rerun re-runs just that section.

Outputs go to results/figures/ and results/tables/ (override with --figures /
--tables). A failing section is reported and the run continues; the exit code
is non-zero if any section failed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent

# (key, directory, what it produces, which output flags its reproduce.py accepts)
SECTIONS = [
    ("3-4", "chapter_3_4_design_figures", "Figures 3.2, 4.2, 4.3", "f"),
    ("5.1", "5.1_baseline_model_and_calibration", "Figures 5.1, 5.2, 5.3 and the Section 5.1 prose numbers", "ft"),
    ("5.2", "5.2_peer_supervision", "Figure 5.4 and the Section 5.2 numbers", "ft"),
    ("5.3.1", "5.3_belief_aggregation/5.3.1_divergent_contributors", "Figure 5.5, Table 5.1", "ft"),
    ("5.3.2", "5.3_belief_aggregation/5.3.2_mesh_scale", "Figure 5.6, Table 5.2", "ft"),
    ("5.4", "5.4_comparison_against_parameter_exchange", "Figure 5.7, Table 5.3 (and their as-published variants)", "ft"),
    ("5.5", "5.5_mesh_geometry", "Figure 5.8, Table 5.4", "ft"),
    ("5.6", "5.6_uncertainty_guided_measurement", "Figure 5.9 and the Section 5.6 numbers", "ft"),
    ("5.7", "5.7_heterogeneous_devices", "Figure 5.10, Table 5.5", "ft"),
    ("5.8.1", "5.8_system_robustness/5.8.1_node_failure", "Figures 5.11, 5.12", "f"),
    ("5.8.2", "5.8_system_robustness/5.8.2_sensor_miscalibration", "Tables 5.6, 5.7", "t"),
    ("5.9", "5.9_ablations", "Table 5.8 (and Table F.1)", "t"),
    ("C", "appendix_C_deployment_granularity", "Tables C.1, C.2", "t"),
    ("D", "appendix_D_parameter_exchange_trajectories", "Figure D.1 (and its as-published variant)", "f"),
    ("E", "appendix_E_heterogeneous_capacity", "Table E.1", "t"),
    ("F", "appendix_F_regularisation_weight_sweep", "Table F.1", "t"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="section keys to run (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--figures", default=None, help="figure output directory (default results/figures)")
    ap.add_argument("--tables", default=None, help="table output directory (default results/tables)")
    ap.add_argument("--rerun-experiments", action="store_true",
                    help="re-run the underlying experiments before regenerating (GPU, ~40 h in total)")
    ap.add_argument("--i-understand-this-takes-days", action="store_true",
                    help="required together with --rerun-experiments")
    a = ap.parse_args()

    if a.list:
        for key, d, what, _ in SECTIONS:
            print("%-6s %-58s %s" % (key, d, what))
        return 0
    if a.rerun_experiments and not a.i_understand_this_takes_days:
        ap.error("--rerun-experiments also needs --i-understand-this-takes-days; "
                 "re-running every experiment is roughly forty hours of GPU time")

    todo = [s for s in SECTIONS if not a.only or s[0] in a.only]
    if a.only and len(todo) != len(a.only):
        unknown = sorted(set(a.only) - {s[0] for s in SECTIONS})
        ap.error("unknown section key(s): %s" % unknown)

    py = sys.executable
    failures, t_all = [], time.time()
    for key, d, what, flags in todo:
        script = RESULTS / d / "reproduce.py"
        cmd = [py, "-u", str(script)]
        if a.figures and "f" in flags:
            cmd += ["--figures", a.figures]
        if a.tables and "t" in flags:
            cmd += ["--tables", a.tables]
        if a.rerun_experiments and key != "3-4":
            cmd.append("--rerun")
        print("\n" + "=" * 100 + "\n[%s] %s  ->  %s\n" % (key, d, what) + "=" * 100, flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(script.parent))
        status = "ok" if r.returncode == 0 else "FAILED (exit %d)" % r.returncode
        print("[%s] %s in %.0f s" % (key, status, time.time() - t0), flush=True)
        if r.returncode != 0:
            failures.append((key, d, r.returncode))

    print("\n" + "=" * 100)
    print("%d of %d sections regenerated in %.1f min" % (len(todo) - len(failures), len(todo), (time.time() - t_all) / 60))
    for key, d, rc in failures:
        print("  FAILED  %-6s %s (exit %d)" % (key, d, rc))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
