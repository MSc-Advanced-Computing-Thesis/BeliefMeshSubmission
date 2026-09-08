"""Appendix D -- Parameter Exchange Trajectories by Environment.

Regenerates Figure D.1 (fig_D_1_exchange_trajectories_all.png): cell-space MSE
per timestep for the four exchange mechanisms in all five offset environments,
from the Section 5.4 artefacts. The readout is Section 5.4's
(5.4_comparison_against_parameter_exchange/exchange_readout.py); as there,
the figure is written twice, once scoring each run against its own field and
once (_as_published) scoring every run against field0, as the report did.

--rerun re-runs the 100 Section 5.4 experiments (GPU, ~16 h).
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
S54 = RESULTS / "5.4_comparison_against_parameter_exchange"
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(S54)]

from _shared.paths import ARTEFACTS, FIGURES  # noqa: E402

SECTION_ART = ARTEFACTS / "5.4_comparison_against_parameter_exchange" / "five_environments"


def load_section_5_4():
    spec = importlib.util.spec_from_file_location("section_5_4_reproduce", S54 / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--rerun", action="store_true", help="re-run the 100 Section 5.4 experiments first (GPU, ~16 h)")
    a = ap.parse_args()
    if a.rerun:
        subprocess.run([sys.executable, "-u", str(S54 / "experiment_parameter_exchange.py"), "--root", a.artefacts],
                       check=True, cwd=str(S54))
    S54R = load_section_5_4()
    from exchange_readout import collect
    for truth, suffix in (("own", ""), ("field0", "_as_published")):
        print("\n=== readout: truth=%s ===" % truth)
        S54R.figure_D_1(collect(Path(a.artefacts), truth), Path(a.figures), suffix)
