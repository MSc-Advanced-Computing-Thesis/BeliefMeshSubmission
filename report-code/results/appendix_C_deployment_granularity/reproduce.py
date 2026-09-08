"""Appendix C -- Operation at Deployment Granularity.

Regenerates, from artefacts/appendix_C_deployment_granularity/deployment_66x66/
deployment_scale_seed42/ (36 nodes at stride 9, 21x21 fields of view, 66x66
cells, 27 wearables, 390 steps, one seed):

    Table C.1   table_C_1_deployment_configuration.csv (+ .txt)
                geometry and cost of the deployment-granularity run against the
                reference 22x22 configuration (wall time from the stored
                manifests; array sizes from the stored arrays)
    Table C.2   table_C_2_deployment_performance.csv (+ .txt)
                accuracy and calibration of the run against the reference
                configuration's five-seed mean and range (the Section 5.3.2
                Product fusion arm)

--rerun re-runs the deployment-granularity experiment (GPU, ~70 min) through
experiment_deployment_66x66.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, TABLES  # noqa: E402
from _shared.artefacts import load_array, manifest, run_dirs  # noqa: E402
from _shared.averaged_readout import run_metrics  # noqa: E402
from _shared.style import fmt  # noqa: E402
from beliefmesh.node.mesh import fov_cells  # noqa: E402
from beliefmesh.simulation.field_variants import build_field  # noqa: E402

SECTION_ART = ARTEFACTS / "appendix_C_deployment_granularity" / "deployment_66x66"
REFERENCE_ART = ARTEFACTS / "5.3_belief_aggregation" / "5.3.2_mesh_scale" / "aggregation_arms"

# The stored .npz arrays are compressed; the sizes the report quotes are of the
# uncompressed .npy arrays the runner writes, so they are recomputed from the
# array shapes and dtypes rather than from file sizes on disk.
ARRAYS = ["cell_beliefs_steps", "cell_ncov_steps", "cell_nig_steps", "cell_mse_steps", "cell_hop_steps",
          "cell_cert_steps", "cell_fused_steps", "realised_wearable_path", "realised_wearable_paths",
          "comm_bytes_steps", "fusion_time_steps", "hop_mse_history", "avg_mse_map", "avg_cert_map"]


def geometry(centres, fov, G):
    fovs = [set(fov_cells(int(a), int(b), fov, G)) for a, b in centres]
    cov = np.zeros((G, G), int)
    for f in fovs:
        for (r, q) in f:
            cov[r, q] += 1
    deg = [sum(1 for j in range(len(fovs)) if j != i and fovs[i] & fovs[j]) for i in range(len(fovs))]
    return dict(cells=G * G, cells_per_node=len(fovs[0]), mean_depth=float(cov[cov > 0].mean()),
                mean_degree=float(np.mean(deg)), covered=float((cov > 0).mean()))


def stored_bytes(d: Path) -> float:
    total = 0
    for name in ARRAYS:
        try:
            a = load_array(d, name)
        except (FileNotFoundError, KeyError):
            continue
        total += a.nbytes if hasattr(a, "nbytes") else 0
    return total


def wall_time(m: dict):
    r = m.get("results", {}) or {}
    for k in ("step_wall_time_total_sec", "wall_time_total_sec", "wall_time_sec"):
        if k in r:
            return float(r[k])
    return float("nan")


def table_C_1(dep: Path, ref_runs, tab_dir: Path):
    md = manifest(dep)
    G_d, fov_d = int(md.get("grid_size", 66)), int(md.get("fov_size", 21))
    centres_d = np.array([[fov_d // 2 + i * 9, fov_d // 2 + j * 9] for i in range(6) for j in range(6)])
    gd = geometry(centres_d, fov_d, G_d)
    from beliefmesh.simulation.assets import ENVIRONMENT_DIR
    centres_r = np.load(ENVIRONMENT_DIR / "node_centres.npy")
    gr = geometry(centres_r, 7, 22)
    wt_d = wall_time(md)
    wt_r = float(np.mean([wall_time(manifest(d)) for d in ref_runs]))
    sb_d, sb_r = stored_bytes(dep), float(np.mean([stored_bytes(d) for d in ref_runs]))
    T = int(md.get("total_steps", 390))
    rows = [("Deployment area (cells)", "%d" % gr["cells"], "%d" % gd["cells"]),
            ("Cells per node", "%d" % gr["cells_per_node"], "%d" % gd["cells_per_node"]),
            ("Node stride", "3", "9"),
            ("Wearables", "3", "%d" % md.get("n_wearables", 27)),
            ("Mean coverage depth", "%.2f" % gr["mean_depth"], "%.2f" % gd["mean_depth"]),
            ("Mean overlap degree", "%.1f" % gr["mean_degree"], "%.1f" % gd["mean_degree"]),
            ("Wall time per run (process, from run log)", "9.2 min", "67.3 min"),
            ("Wall time per timestep (process, from run log)", "1.42 s", "10.36 s"),
            ("Step timing total (manifest)", "%.1f min" % (wt_r / 60), "%.1f min" % (wt_d / 60)),
            ("Step timing per timestep (manifest)", "%.2f s" % (wt_r / T), "%.2f s" % (wt_d / T)),
            ("Stored arrays per run (uncompressed, without disagreement log)", "%.1f MB" % (sb_r / 1e6), "%.1f MB" % (sb_d / 1e6))]
    lines = ["TABLE C.1 -- REFERENCE CONFIGURATION AGAINST THE DEPLOYMENT-GRANULARITY RUN (390 timesteps)",
             "%-62s %-20s %-20s" % ("", "reference (22x22)", "deployment (66x66)")]
    lines += ["%-62s %-20s %-20s" % r for r in rows]
    lines.append("  process wall times are the values recorded in the original run logs (not stored with the artefacts);")
    lines.append("  step timing totals are the runner's own per-step timers from each manifest (reference: 5-seed mean).")
    lines.append("  The report's stored-array sizes (39.6 MB, 376.1 MB) include the disagreement log, which is not carried here.")
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_C_1_deployment_configuration.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    with open(tab_dir / "table_C_1_deployment_configuration.csv", "w", encoding="utf8") as f:
        f.write("measure,reference_22x22,deployment_66x66\n")
        for r in rows:
            f.write("%s,%s,%s\n" % r)
    print("\n".join(lines))


def table_C_2(dep: Path, ref_runs, tab_dir: Path):
    d = run_metrics(dep)
    R = [run_metrics(r) for r in ref_runs]
    lines = ["TABLE C.2 -- DEPLOYMENT-GRANULARITY PERFORMANCE (one seed) AGAINST THE REFERENCE CONFIGURATION (5 seeds)",
             "%-14s %-14s %-22s %-20s" % ("measure", "deployment", "reference mean", "reference range")]
    csv = ["measure,deployment_66x66,reference_mean,reference_sd,reference_min,reference_max"]
    for key, lab, p in (("whole", "Whole-run MSE", 5), ("last50", "Last-50 MSE", 5), ("cov", "Coverage", 3), ("ratio", "hw:RMS", 3)):
        v = [r[key] for r in R]
        mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        lines.append("%-14s %-14s %-22s %-20s" % (lab, "%.*f" % (p, d[key]), fmt((mu, sd), p), "%.*f-%.*f" % (p, min(v), p, max(v))))
        csv.append("%s,%.6f,%.6f,%.6f,%.6f,%.6f" % (lab, d[key], mu, sd, min(v), max(v)))
    # environment comparison quoted in the appendix
    f22, f66 = np.asarray(build_field("field0_original", 22, 390)), np.asarray(build_field("field0_original", 66, 390))
    from scipy.ndimage import zoom
    up = np.stack([zoom(f, 3.0, order=1) for f in f22])
    corr = float(np.corrcoef(up.ravel(), f66.ravel())[0, 1])
    lines.append("  mean |offset|: reference field %.1f deg, deployment field %.1f deg; spatial correlation between them (reference upsampled 3x) %.3f"
                 % (np.abs(f22).mean(), np.abs(f66).mean(), corr))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_C_2_deployment_performance.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    (tab_dir / "table_C_2_deployment_performance.csv").write_text("\n".join(csv) + "\n", encoding="utf8")
    print("\n".join(lines))
    print("  wrote %s" % (tab_dir / "table_C_2_deployment_performance.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--reference", default=str(REFERENCE_ART), help="Section 5.3.2 Product fusion runs")
    ap.add_argument("--rerun", action="store_true", help="re-run the deployment-granularity experiment first (GPU, ~70 min)")
    a = ap.parse_args()
    if a.rerun:
        subprocess.run([sys.executable, "-u", str(HERE / "experiment_deployment_66x66.py")], check=True, cwd=str(HERE))
    dep = Path(a.artefacts) / "deployment_scale_seed42"
    ref = run_dirs(Path(a.reference) / "nig_product_sampled_seed*")
    table_C_1(dep, ref, Path(a.tables))
    table_C_2(dep, ref, Path(a.tables))
