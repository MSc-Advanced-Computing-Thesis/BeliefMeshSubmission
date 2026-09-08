"""Section 5.3.1 -- Aggregation Under Divergent Contributor Conditions.

Regenerates, from artefacts/5.3_belief_aggregation/5.3.1_divergent_contributors/
three_node_colour_world/ (five arms x five seeds of the three-node colour
world):

    Figure 5.5  fig_5_5_three_node_geometry.png   (pure geometry: no artefacts)
    Table 5.1   table_5_1_three_node_aggregation.csv (+ .txt with the full
                                                      printed analysis)

--rerun re-runs the 25 three-node experiments (GPU, ~2 h) via
experiment_three_node_colour_world.py before regenerating.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1]
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, FIGURES, TABLES, SEEDS  # noqa: E402

SECTION_ART = ARTEFACTS / "5.3_belief_aggregation" / "5.3.1_divergent_contributors" / "three_node_colour_world"


def regenerate(fig_dir: Path, tab_dir: Path):
    import figure_three_node_geometry as FG
    import table_three_node_results as TT
    FG.FIG_DIR = fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    FG.main()
    tab_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        TT.main(csv_path=tab_dir / "table_5_1_three_node_aggregation.csv")
    text = buf.getvalue()
    (tab_dir / "table_5_1_three_node_aggregation.txt").write_text(text, encoding="utf8")
    print(text)


def rerun(seeds: str, root: Path):
    subprocess.run([sys.executable, "-u", str(HERE / "experiment_three_node_colour_world.py"),
                    "--seeds", seeds, "--root", str(root)], check=True, cwd=str(HERE))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART), help="run root read (and, with --rerun, written)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rerun", action="store_true", help="re-run the 25 experiments first (GPU, ~2 h)")
    a = ap.parse_args()
    if a.rerun:
        rerun(a.seeds, Path(a.artefacts))
    import table_three_node_results as TT
    TT.ROOT = str(Path(a.artefacts)).replace("\\", "/")
    regenerate(Path(a.figures), Path(a.tables))
