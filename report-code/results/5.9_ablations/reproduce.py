"""Section 5.9 -- Ablations (and Appendix F's regularisation weight sweep).

Regenerates, from artefacts/5.9_ablations/:

    Table 5.8   table_5_8_ablation_summary.csv (+ .txt)
                five ablation families, each recorded as null where the
                cross-variant spread of whole-run MSE is below twice the
                within-variant seed sd:
                  lambda_{1p0,2p5,5p0,10p0,25p0}/   regularisation weight
                  consensus_rho/                    consensus weighting rho
                  weighted_fusion/ (+ 5.3.2 arm)    weighted fusion off / on
                  uncertainty_{epistemic,aleatoric,total}/
                  tempering_{off,on}/               gradient tempering
    Table F.1   table_F_1_lambda_sweep.csv (+ .txt), the lambda family in full
                (also written by appendix_F_regularisation_weight_sweep/reproduce.py)

--rerun re-runs the 105 ablation experiments (GPU, ~20 h) through the
experiment_*.py scripts in this directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
sys.path[:0] = [str(RESULTS), str(RESULTS.parent), str(HERE)]

from _shared.paths import ARTEFACTS, TABLES, SEEDS  # noqa: E402
from _shared.artefacts import run_dirs  # noqa: E402
from _shared.averaged_readout import arm_summary  # noqa: E402
from _shared.style import fmt  # noqa: E402

SECTION_ART = ARTEFACTS / "5.9_ablations"
BASELINE_ARM = ARTEFACTS / "5.3_belief_aggregation" / "5.3.2_mesh_scale" / "aggregation_arms" / "nig_product_sampled_seed*"
LAMBDAS = ["1.0", "2.5", "5.0", "10.0", "25.0"]
RHOS = ["0p01", "0p05", "0p1", "0p2", "0p5", "0p8", "0p99"]


def families(art: Path):
    lam = [(v, arm_summary(run_dirs(art / ("lambda_" + v.replace(".", "p")) / "*"))) for v in LAMBDAS]
    rho = [(r.replace("p", "."), arm_summary(run_dirs(art / "consensus_rho" / ("consensus_rho%s_seed*" % r)))) for r in RHOS]
    wgt = [("off (Product fusion)", arm_summary(run_dirs(BASELINE_ARM))),
           ("on", arm_summary(run_dirs(art / "weighted_fusion" / "*")))]
    unc = [(m, arm_summary(run_dirs(art / ("uncertainty_" + m) / "*"))) for m in ("epistemic", "aleatoric", "total")]
    tmp = [("off", arm_summary(run_dirs(art / "tempering_off" / "**"))),
           ("on", arm_summary(run_dirs(art / "tempering_on" / "**")))]
    return [("Regularisation weight lambda", lam, "1.0 - 25.0"),
            ("Consensus rho", rho, "0.01 - 0.99"),
            ("Weighted fusion", wgt, "off / on"),
            ("Uncertainty measure", unc, "epistemic / aleatoric / total"),
            ("Gradient tempering", tmp, "off / on")]


def spread(rows, key="whole"):
    import statistics as st
    v = [r[key][0] for _, r in rows if r]
    sds = [r[key][1] for _, r in rows if r]
    return max(v) - min(v), min(v), max(v), st.mean(sds)


def table_5_8(fams, tab_dir: Path):
    lines = ["TABLE 5.8 -- ABLATION SUMMARY (cell space, averaged-NIG readout, 5 seeds per variant)",
             "%-30s %8s %-30s %-23s %-10s %s" % ("family", "variants", "range swept", "whole-run MSE range", "seed sd", "outcome")]
    csv = ["family,n_variants,range_swept,whole_run_mse_min,whole_run_mse_max,mean_seed_sd,spread,outcome"]
    for name, F, rng in fams:
        F = [(l, r) for l, r in F if r]
        sp, lo, hi, sd = spread(F)
        outcome = "Null" if sp < 2 * sd else "Effect"
        lines.append("%-30s %8d %-30s %-23s %-10s %s (spread %.5f = %.1f x seed sd)"
                     % (name, len(F), rng, "%.5f - %.5f" % (lo, hi), "%.5f" % sd, outcome, sp, sp / sd))
        csv.append("%s,%d,%s,%.6f,%.6f,%.6f,%.6f,%s" % (name, len(F), rng.replace(",", ";"), lo, hi, sd, sp, outcome))
    lines.append("  Null: the whole-run spread across the family's variants is below twice the within-variant seed sd.")
    lines.append("  Gradient tempering runs use one offset environment per seed; the report's row (0.02066-0.02067, seed sd 0.01077)")
    lines.append("  scored all five against the field0 field. Each run is scored against its own field here. Outcome unchanged: null.")
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_5_8_ablation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    (tab_dir / "table_5_8_ablation_summary.csv").write_text("\n".join(csv) + "\n", encoding="utf8")
    print("\n".join(lines))
    print("  wrote %s" % (tab_dir / "table_5_8_ablation_summary.csv"))
    # every variant, for reference
    lines = ["ALL ABLATION VARIANTS (averaged-NIG readout, last-50 window)",
             "%-30s %-22s %3s %-21s %-21s %-15s %-15s" % ("family", "variant", "n", "whole-run MSE", "last-50 MSE", "90% coverage", "hw : RMS")]
    for name, F, rng in fams:
        for lab, r in F:
            if r:
                lines.append("%-30s %-22s %3d %-21s %-21s %-15s %-15s"
                             % (name, lab, r["n"], fmt(r["whole"]), fmt(r["last50"]), fmt(r["cov"], 3), fmt(r["ratio"], 3)))
    (tab_dir / "table_5_8_ablation_variants.txt").write_text("\n".join(lines) + "\n", encoding="utf8")


def table_F_1(fams, tab_dir: Path):
    lam = dict((n, F) for n, F, _ in fams)["Regularisation weight lambda"]
    lines = ["TABLE F.1 -- REGULARISATION WEIGHT SWEEP (5 seeds)",
             "%-8s %-21s %-21s %-9s %-9s" % ("lambda", "whole-run MSE", "last-50 MSE", "coverage", "hw:RMS")]
    csv = ["lambda,n_seeds,whole_run_mse_mean,whole_run_mse_sd,last50_mse_mean,last50_mse_sd,coverage90_mean,coverage90_sd,hw_rms_mean,hw_rms_sd"]
    for lab, r in lam:
        lines.append("%-8s %-21s %-21s %-9.3f %-9.3f" % (lab, fmt(r["whole"]), fmt(r["last50"]), r["cov"][0], r["ratio"][0]))
        csv.append("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f" % (lab, r["n"], *r["whole"], *r["last50"], *r["cov"], *r["ratio"]))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_F_1_lambda_sweep.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    (tab_dir / "table_F_1_lambda_sweep.csv").write_text("\n".join(csv) + "\n", encoding="utf8")
    print("\n".join(lines))
    print("  wrote %s" % (tab_dir / "table_F_1_lambda_sweep.csv"))


def rerun(seeds: str, art: Path):
    py = sys.executable
    run = lambda *cmd: subprocess.run([py, "-u", *cmd], check=True, cwd=str(HERE))
    for v in LAMBDAS:
        run(str(HERE / "experiment_mesh_aggregation.py"), "--mode", "nig_product", "--lam", v, "--seeds", seeds,
            "--root", str(art / ("lambda_" + v.replace(".", "p"))))
    run(str(HERE / "experiment_consensus_rho.py"), "--part", "rho", "--seeds", seeds, "--root", str(art / "consensus_rho"))
    run(str(HERE / "experiment_weighted_fusion.py"), "--seeds", seeds, "--root", str(art / "weighted_fusion"))
    for m in ("epistemic", "aleatoric", "total"):
        run(str(HERE / "experiment_uncertainty_measure.py"), "--measure", m, "--seeds", seeds, "--root", str(art / ("uncertainty_" + m)))
    run(str(HERE / "experiment_tempering.py"), "--arm", "het_temper", "--root", str(art / "tempering_on"))
    run(str(HERE / "experiment_tempering.py"), "--arm", "het_notemper", "--root", str(art / "tempering_off"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--artefacts", default=str(SECTION_ART))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rerun", action="store_true", help="re-run the ablation experiments first (GPU, ~20 h)")
    a = ap.parse_args()
    if a.rerun:
        rerun(a.seeds, Path(a.artefacts))
    fams = families(Path(a.artefacts))
    table_5_8(fams, Path(a.tables))
    table_F_1(fams, Path(a.tables))
