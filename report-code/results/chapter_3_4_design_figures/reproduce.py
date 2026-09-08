"""Chapters 3 and 4 -- design figures (no stored artefacts; pure computation).

    Figure 3.2  fig_3_2_fusion_rules.png      Product and Average fusion on three
                                              contributor configurations
    Figure 4.2  fig_4_2_environments.png      colour world and offset world
                                              rendered from the environment code
    Figure 4.3  fig_4_3_mesh_geometry.png     node placement and cell coverage of
                                              the reference 36-node mesh
                (fig_4_3_mesh_geometry_annotated.png carries per-cell counts)

Figure 4.1 (example rotations of the digit seven) is not produced by any
script in the original tree and is not regenerated here.

There is no --rerun: nothing here depends on an experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import FIGURES  # noqa: E402


def regenerate(fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    import figure_3_2_fusion_rules as F32
    import figure_4_2_environments as F42
    import figure_4_3_mesh_geometry as F43
    F32.main(out=fig_dir / "fig_3_2_fusion_rules.png")
    F42.OUT = fig_dir / "fig_4_2_environments.png"
    F42.main()
    F43.OUT_DIR = fig_dir
    F43.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    a = ap.parse_args()
    regenerate(Path(a.figures))
