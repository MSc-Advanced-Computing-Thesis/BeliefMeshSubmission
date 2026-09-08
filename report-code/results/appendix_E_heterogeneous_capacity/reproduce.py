"""Appendix E -- Heterogeneous Capacity Across Environments.

Regenerates Table E.1 (table_E_1_heterogeneous_capacity.csv + .txt) from
artefacts/appendix_E_heterogeneous_capacity/mixed_capacity_five_environments/
{naive,certainty,nig_product,avgfusion}/het/: the four aggregation rules on the
mixed-capacity mesh, one run per offset environment with each environment
paired with its own seed. Each run is scored against its own environment.

--rerun re-runs the 20 experiments (GPU, ~4 h) through
experiment_mixed_capacity_five_environments.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, TABLES  # noqa: E402
from _shared.artefacts import run_dirs  # noqa: E402
from _shared.averaged_readout import arm_summary  # noqa: E402
from _shared.arm_names import name as arm_name  # noqa: E402
from _shared.style import fmt  # noqa: E402

SECTION_ART = ARTEFACTS / "appendix_E_heterogeneous_capacity" / "mixed_capacity_five_environments"
ARMS = ["naive", "certainty", "nig_product", "avgfusion"]


def table_E_1(art: Path, tab_dir: Path):
    rows = [(a, arm_summary(run_dirs(art / a / "het" / "*"))) for a in ARMS]
    lines = ["TABLE E.1 -- AGGREGATION RULES UNDER HETEROGENEOUS CAPACITY ACROSS FIVE ENVIRONMENTS (one run per environment;"
             " spread is across environment and seed jointly)",
             "%-16s %3s %-21s %-21s %-15s %-15s" % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS")]
    for a, r in rows:
        if r is None:
            lines.append("%-16s  -- no runs --" % arm_name(a)); continue
        lines.append("%-16s %3d %-21s %-21s %-15s %-15s"
                     % (arm_name(a), r["n"], fmt(r["whole"]), fmt(r["last50"]), fmt(r["cov"], 3), fmt(r["ratio"], 3)))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_E_1_heterogeneous_capacity.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    with open(tab_dir / "table_E_1_heterogeneous_capacity.csv", "w", encoding="utf8") as f:
        f.write("arm,n_runs,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,last50_mse_sd,coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd\n")
        for a, r in rows:
            if r:
                f.write("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (arm_name(a), r["n"], *r["whole"], *r["last50"], *r["cov"], *r["ratio"]))
    print("\n".join(lines))
    print("  wrote %s" % (tab_dir / "table_E_1_heterogeneous_capacity.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--rerun", action="store_true", help="re-run the 20 experiments first (GPU, ~4 h)")
    a = ap.parse_args()
    if a.rerun:
        subprocess.run([sys.executable, "-u", str(HERE / "experiment_mixed_capacity_five_environments.py"),
                        "--root", a.artefacts], check=True, cwd=str(HERE))
    table_E_1(Path(a.artefacts), Path(a.tables))
