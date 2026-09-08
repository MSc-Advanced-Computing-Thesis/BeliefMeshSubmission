# Deployment granularity at FULL RUN LENGTH: 36 nodes, 21x21-cell FOVs, 390 steps.
#
# PURPOSE. The 36-step demonstration established that the system OPERATES at
# deployment cell counts. This runs the same configuration to the chapter's
# full 390 steps so its converged behaviour can be set beside the reference
# configuration's. Single seed (42) against the reference's five, so it shows
# broad consistency, not equivalence.
#
# GEOMETRY. 36 nodes on a 6x6 lattice at stride 9, fov_size=20 on a 66x66 grid.
#   - fov_cells uses half = fov//2 with an INCLUSIVE range, so an even
#     fov_size yields fov+1 cells per axis: 20 -> 21x21 = 441 cells.
#   - 66 rather than 65 so no node's FOV is clipped by the grid edge; at 65 the
#     outermost nodes get 400 cells and interior ones 441. 66 gives every node
#     441 and reproduces the reference mesh's mean coverage depth 3.64 and
#     overlap degree 15.0 exactly.
#   - 27 wearables holds the reference density of 3/484 (4356*3/484 = 27).
#
# WEARABLE SEEDING. The reference configuration uses seed 200+i for i in 0..2
# at env seed 42. That convention is extended directly to i in 0..26, seeds
# 200..226. build_random_wander_path draws its own uniform start per seed, so
# 27 independent seeds scatter the starts over the interior with no extra
# placement logic.
#
# ENVIRONMENT. field0_original rescaled to 66x66 -- a field0-FAMILY realisation
# at deployment granularity, NOT the 22-grid field enlarged: the wedge geometry
# rescales proportionally but the background control points are redrawn at the
# larger grid (n_per_axis grows with grid_size), so the large-scale drift is an
# independent draw. Measured spatial correlation of the two time-mean fields:
# 0.288.
#
# Run: python -u experiments/section5_2/run_deployment_scale_demo.py

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import random
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

from beliefmesh.simulation.runner import run_mesh_experiment
from beliefmesh.simulation.field_variants import build_field
from beliefmesh.simulation.offset_fields import build_random_wander_path

from beliefmesh.config import load_config
from beliefmesh.node.mesh import fov_cells

CKPT = CHECKPOINTS["baseline"]
EXCLUDED = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
LR, LAM, SEED = 3e-5, 5.0, 42
G, FOV, STRIDE, N_SIDE, T = 66, 20, 9, 6, 390
N_WEARABLES = 27
OUT = Path(ART + "/appendix_C_deployment_granularity/deployment_66x66")


def centres():
    h = FOV // 2
    return np.array([[h + i * STRIDE, h + j * STRIDE]
                     for i in range(N_SIDE) for j in range(N_SIDE)], dtype=int)


def geometry_report(c):
    fovs = [set(fov_cells(int(a), int(b), FOV, G)) for a, b in c]
    cov = np.zeros((G, G), int)
    for f in fovs:
        for (r, q) in f:
            cov[r, q] += 1
    deg = [sum(1 for j in range(len(fovs)) if j != i and fovs[i] & fovs[j])
           for i in range(len(fovs))]
    print("GEOMETRY")
    print("  grid %dx%d = %d cells; %d nodes at stride %d" % (G, G, G * G, len(c), STRIDE))
    print("  FOV cells per node: min %d max %d" % (min(map(len, fovs)), max(map(len, fovs))))
    print("  coverage %d/%d (%.0f%%), mean depth over covered %.2f"
          % ((cov > 0).sum(), G * G, 100 * (cov > 0).mean(), cov[cov > 0].mean()))
    print("  overlap degree mean %.1f (reference mesh: 15.0)"
          % (sum(deg) / len(deg)))
    return int(cov.max())


def main():
    cfg = load_config()
    cfg.model.lr = LR
    OUT.mkdir(parents=True, exist_ok=True)
    c = centres()
    max_cov = geometry_report(c)

    field = np.asarray(build_field("field0_original", G, T))
    print("\nENVIRONMENT  field0_original rescaled to %dx%d" % (G, G))
    print("  mean |offset| %.3f deg, max %.3f deg" % (np.abs(field).mean(),
                                                      np.abs(field).max()))

    paths = [build_random_wander_path(T, G, seed=200 + i)
             for i in range(N_WEARABLES)]
    starts = np.array([p[0] for p in paths])
    print("\nWEARABLES  %d, seeds 200..%d" % (N_WEARABLES, 200 + N_WEARABLES - 1))
    print("  start positions: row %.1f..%.1f, col %.1f..%.1f; mean pairwise "
          "separation %.1f cells"
          % (starts[:, 0].min(), starts[:, 0].max(),
             starts[:, 1].min(), starts[:, 1].max(),
             float(np.mean([np.linalg.norm(starts[i] - starts[j])
                            for i in range(len(starts))
                            for j in range(i + 1, len(starts))]))))

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    tracemalloc.start()
    t0 = time.time()
    res = run_mesh_experiment(
        cfg, condition="deployment_scale_seed%d" % SEED,
        run_dir=OUT / ("deployment_scale_seed%d" % SEED),
        all_grids=np.full((T, G, G), 0.5), wearable_paths=paths,
        node_centres=c, fov_size=FOV, mode="nig_product",
        baseline_checkpoint=CKPT, n_wearable_samples=1, n_train_repeats=1,
        title="Deployment scale 390 (66x66, FOV 20, 27 wearables, seed %d)" % SEED,
        offset_field=field, env_seed=SEED,
        excluded_rotation_ranges=EXCLUDED,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        sample_target=True, draws_per_target=1,
        track_disagreement=True,
    )
    wall = time.time() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("\nWALL TIME  %.1f s (%.2f min), %.3f s/step over %d steps"
          % (wall, wall / 60.0, wall / T, T))
    print("PEAK PYTHON HEAP (tracemalloc)  %.1f MB" % (peak / 1e6))
    print("  note: tracemalloc counts Python allocations only -- torch tensors")
    print("  and CUDA memory are not included.")
    d = OUT / ("deployment_scale_seed%d" % SEED)
    tot = sum(f.stat().st_size for f in d.glob("*.npy"))
    print("STORED ARRAYS  %.1f MB across %d .npy files"
          % (tot / 1e6, len(list(d.glob("*.npy")))))
    for f in sorted(d.glob("*.npy"), key=lambda x: -x.stat().st_size)[:4]:
        print("    %-24s %7.1f MB" % (f.name, f.stat().st_size / 1e6))
    print("\nrun_mesh_experiment last-50-equivalent: %s" % res.get("mean_mse_last_50"))
    print("=== DEPLOYMENT 390 DONE ===")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description=__doc__ or "Deployment-granularity run: 66x66 cells, 36 nodes, 27 wearables (GPU, ~70 min)")
    _p.parse_args()
    main()
