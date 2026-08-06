# Re-analyze already-completed node-failure runs with the revised metrics
# (dead-zone size, threshold framing, paired-vs-control, distance-from-
# boundary) WITHOUT re-running the simulation -- pure post-hoc computation
# from each run's saved cell_mse_steps.npy / cell_cert_steps.npy, per
# Christian's 2026-08 correction. Overwrites manifest.yaml and figure.png
# in place for each run directory found under ROOT for the given seed.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.evaluation import git_commit
from stage6_spatial_mesh.run_node_failure import (ROOT, compute_stats, make_figure,
                                                   control_dir_for)

from beliefmesh.node.mesh import fov_cells

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")


def coverage_count_for(failure_ids: list[int], G: int) -> np.ndarray:
    centres = np.load(ENV / "node_centres.npy")
    coverage = np.zeros((G, G), dtype=int)
    for i, (cx, cy) in enumerate(centres):
        if i in failure_ids:
            continue
        for (r, c) in fov_cells(int(cx), int(cy), 7, G):
            coverage[r, c] += 1
    return coverage


def reanalyze(run_dir: Path):
    old_manifest = yaml.safe_load(open(run_dir / "manifest.yaml"))
    tag = old_manifest["condition"]
    mesh_mode = old_manifest["mesh_mode"]; seed = old_manifest["seed"]
    failure_mode = old_manifest["failure_mode"]; fraction = old_manifest["fraction"]
    failure_ids = old_manifest["failure_ids"]

    cell_mse_steps = np.load(run_dir / "cell_mse_steps.npy")
    cell_cert_steps = np.load(run_dir / "cell_cert_steps.npy")
    G = cell_mse_steps.shape[1]
    coverage_count = coverage_count_for(failure_ids, G)
    np.save(run_dir / "coverage_count.npy", coverage_count)

    control_mse = control_cert = None
    if fraction > 0:
        cdir = control_dir_for(mesh_mode, seed)
        if (cdir / "cell_mse_steps.npy").exists():
            control_mse = np.load(cdir / "cell_mse_steps.npy")
            control_cert = np.load(cdir / "cell_cert_steps.npy")

    stats = compute_stats(cell_mse_steps, cell_cert_steps, coverage_count, G, control_mse, control_cert)

    manifest = dict(old_manifest)
    manifest["git_commit"] = git_commit()
    manifest["results"] = dict(
        dead_zone_cell_count=stats["dead_zone_cell_count"],
        dead_zone_fraction=stats["dead_zone_fraction"],
        threshold_summary=stats["threshold_summary"],
        pre_window_cert_mse_r=float(stats["pre_r"]), pre_window_cert_mse_p=float(stats["pre_p"]),
        post_window_cert_mse_r=float(stats["post_r"]), post_window_cert_mse_p=float(stats["post_p"]),
        coverage_count_breakdown=stats["coverage_breakdown"],
        distance_from_dead_zone_breakdown=stats["distance_breakdown"],
        paired_vs_control=stats["paired_vs_control"],
    )
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    make_figure(run_dir, failure_mode, fraction, mesh_mode, seed, stats, coverage_count)

    print(f"\n### {tag} (re-analyzed) ###")
    print(f"    dead_zone: {stats['dead_zone_cell_count']} cells ({100*stats['dead_zone_fraction']:.1f}% of grid)")
    print(f"    threshold: covered_mse={stats['threshold_summary']['covered_mean_mse']} "
          f"covered_cert={stats['threshold_summary']['covered_mean_cert']}")
    if stats["paired_vs_control"] and "mean_diff" in stats["paired_vs_control"]:
        pv = stats["paired_vs_control"]
        print(f"    paired_vs_control: n={pv['n_cells_compared']} mean_diff={pv['mean_diff']:+.5f} "
              f"t={pv['t']:.3f} p={pv['p']:.4f}")
    print(f"    cert-mse r: pre={stats['pre_r']:.4f} post={stats['post_r']:.4f}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    # control MUST be re-analyzed first so its arrays exist for the others'
    # paired comparison -- but the control has no paired comparison itself,
    # so process order doesn't actually matter for correctness here, only
    # by convention.
    dirs = sorted(ROOT.glob(f"*_seed{seed}"))
    for d in dirs:
        if (d / "manifest.yaml").exists():
            reanalyze(d)
    print("\n=== ALL RE-ANALYZED ===")
