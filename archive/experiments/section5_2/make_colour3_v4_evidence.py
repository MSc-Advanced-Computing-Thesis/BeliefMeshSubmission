# V4 evidence-against-error figure. Style, dimensions, palette and arm set are
# the v2 module's, reused rather than reimplemented; only the geometry, the run
# root and the output filename differ, so the v2 and v3 figures are untouched.
#
# V4 moves B to (11,9) and C to (9,12), so the module's cell sets and the
# SHARED array must be rebuilt -- they are computed at import time from the v2
# centres and would otherwise silently index the wrong cells.
#
# Arm set matches the v3 figure (five arms) so the two are directly comparable.
#
# Run: python -u experiments/section5_2/make_colour3_v4_evidence.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_colour3_v2_figures as M
import run_colour_three_node_v4 as V4
from beliefmesh.node.mesh import fov_cells

M.A, M.B, M.C = V4.A_CENTRE, V4.B_CENTRE, V4.C_CENTRE
M.a = set(fov_cells(*M.A, M.FOV, M.G))
M.b = set(fov_cells(*M.B, M.FOV, M.G))
M.c = set(fov_cells(*M.C, M.FOV, M.G))
M.SHARED = np.array(sorted(M.c & (M.a | M.b)))
print("v4 geometry: A%s B%s C%s, supervised %d cells"
      % (M.A, M.B, M.C, len(M.SHARED)))

FIVE = ["frozen", "naive", "certainty", "nig_product", "nig_product_avgtrain"]
M.ROOT = "runs/chapter5_v2/s5_3_colour_three_node_v4"
D = M.collect()
M.fig_evidence(D, outname="ch5_5_3_colour3_v4_evidence_vs_error", arms=FIVE)
