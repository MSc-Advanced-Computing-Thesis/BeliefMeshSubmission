"""Appendix F -- Regularisation Weight Sweep.

Regenerates Table F.1 (table_F_1_lambda_sweep.csv + .txt) from
artefacts/5.9_ablations/lambda_{1p0,2p5,5p0,10p0,25p0}/ (Product fusion on the
reference mesh at five values of the regularisation weight, 5 seeds each). The
computation is Section 5.9's; this script exists so the appendix has its own
entry point.

--rerun re-runs the 25 lambda experiments (GPU, ~5 h) through the Section 5.9
experiment_mesh_aggregation.py script.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
S59 = RESULTS / "5.9_ablations"
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(S59)]

from _shared.paths import ARTEFACTS, TABLES, SEEDS  # noqa: E402

SECTION_ART = ARTEFACTS / "5.9_ablations"


def load_section_5_9():
    spec = importlib.util.spec_from_file_location("section_5_9_reproduce", S59 / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rerun", action="store_true", help="re-run the 25 lambda experiments first (GPU, ~5 h)")
    a = ap.parse_args()
    S59R = load_section_5_9()
    if a.rerun:
        for v in S59R.LAMBDAS:
            subprocess.run([sys.executable, "-u", str(S59 / "experiment_mesh_aggregation.py"), "--mode", "nig_product",
                            "--lam", v, "--seeds", a.seeds, "--root", str(Path(a.artefacts) / ("lambda_" + v.replace(".", "p")))],
                           check=True, cwd=str(S59))
    S59R.table_F_1(S59R.families(Path(a.artefacts)), Path(a.tables))
