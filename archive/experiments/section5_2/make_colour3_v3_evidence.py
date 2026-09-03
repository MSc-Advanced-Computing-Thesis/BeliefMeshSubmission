# V3 evidence-against-error figure, FOUR arms.
#
# nig_product_avgtrain is excluded here -- averaged fusion in TRAINING is
# reported in Section 5.4.3, not in 5.3. Style, dimensions and palette are the
# v2 module's, reused rather than reimplemented; only the run root, the arm
# subset and the output filename differ, so the v2 figure is untouched.
#
# Geometry is identical between v2 and v3 (A (8,9), B (13,9), C (10,12)), so
# the module's cell sets and anchor-slot mapping apply unchanged.
#
# Run: python -u experiments/section5_2/make_colour3_v3_evidence.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_colour3_v2_figures as M

# five arms: the naming brief lists Frozen, Naive, Certainty, Product
# fusion and Average fusion for this figure, which reverses the earlier
# four-arm instruction. Flagged in the report.
FIVE = ["frozen", "naive", "certainty", "nig_product", "nig_product_avgtrain"]
M.ROOT = "runs/chapter5_v2/s5_3_colour_three_node_v3"
D = M.collect()
M.fig_evidence(D, outname="ch5_5_3_colour3_v3_evidence_vs_error", arms=FIVE)
