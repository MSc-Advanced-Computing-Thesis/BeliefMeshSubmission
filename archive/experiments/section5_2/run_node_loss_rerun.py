# Section 5.7 node-loss rerun: all 55 runs, with the averaged-readout arrays.
#
# WHY. run_node_failure.py writes its own save block rather than calling
# runner.run_mesh_experiment, so cell_beliefs_steps / cell_ncov_steps /
# cell_nig_steps were never written for this section. Section 5.7 was the only
# cell-space result in Chapter 5 still on the argmax estimator, and it could
# not be recomputed from the stored artefacts. The save block has now been
# extended (purely additively -- see the note in run_node_failure.py); this
# reruns the section so the arrays exist.
#
# ORDERING MATTERS. The 0% control for a seed must complete BEFORE any other
# condition at that seed, because each condition loads the control's saved
# arrays for the paired comparison. Seeds are therefore the outer loop and the
# control is always first within a seed.
#
# NEW OUTPUT ROOT. Nothing under runs/chapter5_v2/s5_7_node_loss is touched.
#
# Run: python -u experiments/section5_2/run_node_loss_rerun.py [--root DIR]

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR.parent / "src"))

import stage6_spatial_mesh.run_node_failure as NF

SEEDS = [42, 1042, 2042, 3042, 4042]
# control FIRST -- every other condition at this seed reads its arrays
CONDITIONS = [("random", 0.0),
              ("random", 0.10), ("random", 0.25), ("random", 0.40),
              ("random", 0.50), ("random", 0.55), ("random", 0.60),
              ("random", 0.65),
              ("clustered", 0.10), ("clustered", 0.25), ("clustered", 0.40)]
DEFAULT_ROOT = "runs/chapter5_v2/s5_7_node_loss_avg"


def main(root):
    NF.ROOT = Path(root)
    NF.ROOT.mkdir(parents=True, exist_ok=True)
    total = len(SEEDS) * len(CONDITIONS)
    print("NODE-LOSS RERUN: %d runs (%d seeds x %d conditions) -> %s"
          % (total, len(SEEDS), len(CONDITIONS), root), flush=True)
    t0 = time.time()
    done = failed = 0
    for seed in SEEDS:
        for mode, frac in CONDITIONS:
            n = done + failed + 1
            tag = "%s_%dpct_seed%d" % (mode, round(frac * 100), seed)
            print("\n>>> [%d/%d] %s  (elapsed %.1f min)"
                  % (n, total, tag, (time.time() - t0) / 60.0), flush=True)
            try:
                NF.run_condition(mode, frac, seed=seed, mesh_mode="nig_product")
                done += 1
            except Exception:
                # one bad run must not cost the other 54
                failed += 1
                print("!!! FAILED %s" % tag, flush=True)
                traceback.print_exc()
    print("\n=== NODE-LOSS RERUN COMPLETE: %d ok, %d failed, %.1f min ==="
          % (done, failed, (time.time() - t0) / 60.0), flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT)
    a = ap.parse_args()
    main(a.root)
