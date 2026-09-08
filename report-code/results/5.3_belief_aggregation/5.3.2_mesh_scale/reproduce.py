"""Section 5.3.2 -- Aggregation at Mesh Scale.

Regenerates, from artefacts/5.3_belief_aggregation/5.3.2_mesh_scale/:

    Figure 5.6  fig_5_6_mesh_snapshots.png     three snapshots of one Product
                                               fusion run (snapshot_run/)
    Table 5.2   table_5_2_aggregation_mesh_scale.csv (+ .txt)
                Frozen, Naive, Certainty, Product fusion (aggregation_arms/,
                frozen/) and Average fusion (average_fusion/), 5 seeds each,
                averaged-NIG readout over the last-50 window.

--rerun re-runs the 25 mesh experiments behind the table and the snapshot run
(GPU, ~4 h): experiment_mesh_aggregation.py for the three sampled-target arms,
experiment_frozen_reference.py for Frozen, experiment_average_fusion.py for
Average fusion, experiment_snapshot_run.py for the figure's run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1]
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, FIGURES, TABLES, SEEDS  # noqa: E402
from _shared.artefacts import run_dirs  # noqa: E402
from _shared.averaged_readout import arm_summary  # noqa: E402
from _shared.arm_names import name as arm_name  # noqa: E402
from _shared.style import fmt  # noqa: E402

SECTION_ART = ARTEFACTS / "5.3_belief_aggregation" / "5.3.2_mesh_scale"

# (display key, glob under the section's artefacts)
ARMS = [("frozen", "frozen/agg_cmp_frozen*"),
        ("naive", "aggregation_arms/naive_sampled_seed*"),
        ("certainty", "aggregation_arms/certainty_sampled_seed*"),
        ("nig_product", "aggregation_arms/nig_product_sampled_seed*"),
        ("avgfusion", "average_fusion/avg_training_seed*")]


def table_5_2(art: Path, tab_dir: Path):
    rows = []
    for key, pat in ARMS:
        r = arm_summary(run_dirs(art / pat))
        if r is None:
            print("  no runs for %s (%s)" % (key, pat))
            continue
        rows.append((key, r))
    lines = ["TABLE 5.2 -- AGGREGATION AT MESH SCALE (cell space, averaged-NIG readout, last-50 window, 5 seeds, mean +/- sd)",
             "%-16s %3s %-21s %-21s %-15s %-15s" % ("arm", "n", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS")]
    for key, r in rows:
        lines.append("%-16s %3d %-21s %-21s %-15s %-15s"
                     % (arm_name(key), r["n"], fmt(r["whole"]), fmt(r["last50"]), fmt(r["cov"], 3), fmt(r["ratio"], 3)))
    ref = dict(rows)["nig_product"]["whole"][0]
    for key, r in rows:
        if key not in ("nig_product", "frozen"):
            lines.append("  %s vs Product fusion, whole-run: %+.2f%%" % (arm_name(key), 100 * (r["whole"][0] - ref) / ref))
    tab_dir.mkdir(parents=True, exist_ok=True)
    txt = tab_dir / "table_5_2_aggregation_mesh_scale.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf8")
    csv = tab_dir / "table_5_2_aggregation_mesh_scale.csv"
    with open(csv, "w", encoding="utf8") as f:
        f.write("arm,n_seeds,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,last50_mse_sd,"
                "coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd\n")
        for key, r in rows:
            f.write("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n"
                    % (arm_name(key), r["n"], *r["whole"], *r["last50"], *r["cov"], *r["ratio"]))
    print("\n".join(lines))
    print("  wrote %s\n  wrote %s" % (csv, txt))


def figure_5_6(art: Path, fig_dir: Path):
    import figure_mesh_snapshots as FS
    FS.RUN = art / "snapshot_run" / "fusion_random_seed42"
    FS.FIGS = fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    FS.main()


def rerun(seeds: str, art: Path):
    py = sys.executable
    run = lambda *cmd: subprocess.run([py, "-u", *cmd], check=True, cwd=str(HERE))
    for mode in ("nig_product", "naive", "certainty"):
        run(str(HERE / "experiment_mesh_aggregation.py"), "--mode", mode, "--seeds", seeds,
            "--root", str(art / "aggregation_arms"))
    run(str(HERE / "experiment_frozen_reference.py"), "--mode", "frozen", "--seeds", seeds,
        "--root", str(art / "frozen"))
    run(str(HERE / "experiment_average_fusion.py"), "--seeds", seeds, "--root", str(art / "average_fusion"))
    run(str(HERE / "experiment_snapshot_run.py"), "--mode", "fusion", "--policy", "random",
        "--seeds", "42", "--root", str(art / "snapshot_run"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rerun", action="store_true", help="re-run the experiments first (GPU, ~4 h)")
    a = ap.parse_args()
    if a.rerun:
        rerun(a.seeds, Path(a.artefacts))
    table_5_2(Path(a.artefacts), Path(a.tables))
    figure_5_6(Path(a.artefacts), Path(a.figures))
