"""Did the burn-in equalise the anchors? Each anchor's error on its own
exclusive cells (argmax readout = the sole covering node) at the end of the
burn-in (last 50 of 300 steps), against the same cells at the start of the v4
run (first 10 steps, and step 0), per seed and averaged.

Run: python analysis_v5_burnin.py [--v4 ROOT] [--v5 ROOT]
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
import argparse
from pathlib import Path
import numpy as np
from _shared.paths import ARTEFACTS, TABLES, SEEDS
from _shared.artefacts import load_array
import experiment_three_node_colour_world as V4

B = ARTEFACTS / "5.3_belief_aggregation" / "5.3.1_divergent_contributors"


def mask(cells, G=V4.G):
    m = np.zeros((G, G), bool)
    for r, c in cells:
        m[r, c] = True
    return m


def main(v4: Path, v5: Path, tab_dir: Path, arm: str = "nig_product"):
    a, b, c = V4.regions()
    ma, mb = mask(a - b - c), mask(b - a - c)
    rows = []
    for s in SEEDS:
        if not (v5 / ("%s_seed%d" % (arm, s)) / "manifest.yaml").exists():
            continue
        m4 = load_array(v4 / ("%s_seed%d" % (arm, s)), "cell_mse_steps")
        mB = load_array(v5 / "burnin" / ("seed%d" % s), "cell_mse_steps")
        m5 = load_array(v5 / ("%s_seed%d" % (arm, s)), "cell_mse_steps")
        deg = lambda x: float(np.sqrt(np.nanmean(x))) * 180
        rows.append(dict(seed=s,
                         v4_A0=deg(m4[0][ma]), v4_B0=deg(m4[0][mb]),
                         v4_A10=deg(m4[:10][:, ma]), v4_B10=deg(m4[:10][:, mb]),
                         burn_A=deg(mB[-50:][:, ma]), burn_B=deg(mB[-50:][:, mb]),
                         v5_A10=deg(m5[:10][:, ma]), v5_B10=deg(m5[:10][:, mb])))
    lines = ["ANCHOR ADAPTATION: RMS error (deg) on each anchor's own exclusive cells, %s arm" % arm,
             "%-6s | v4 start: step0 A/B, steps0-9 A/B | end of burn-in (last 50): A/B | v5 start steps0-9: A/B" % "seed"]
    for r in rows:
        lines.append("%-6d | %6.2f %6.2f  %6.2f %6.2f | %6.2f %6.2f | %6.2f %6.2f"
                     % (r["seed"], r["v4_A0"], r["v4_B0"], r["v4_A10"], r["v4_B10"], r["burn_A"], r["burn_B"], r["v5_A10"], r["v5_B10"]))
    mean = lambda k: np.mean([r[k] for r in rows])
    lines.append("mean   | %6.2f %6.2f  %6.2f %6.2f | %6.2f %6.2f | %6.2f %6.2f"
                 % tuple(mean(k) for k in ("v4_A0", "v4_B0", "v4_A10", "v4_B10", "burn_A", "burn_B", "v5_A10", "v5_B10")))
    lines.append("A - B difference (deg): v4 start step0 %+.2f, steps0-9 %+.2f; end of burn-in %+.2f; v5 start steps0-9 %+.2f"
                 % (mean("v4_A0") - mean("v4_B0"), mean("v4_A10") - mean("v4_B10"), mean("burn_A") - mean("burn_B"), mean("v5_A10") - mean("v5_B10")))
    lines.append("A / B ratio: v4 start steps0-9 %.2f; end of burn-in %.2f" % (mean("v4_A10") / mean("v4_B10"), mean("burn_A") / mean("burn_B")))
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "table_v5_anchor_burnin.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v4", default=str(B / "three_node_colour_world"))
    ap.add_argument("--v5", default=str(B / "three_node_colour_world_v5"))
    ap.add_argument("--tables", default=str(TABLES))
    a = ap.parse_args()
    main(Path(a.v4), Path(a.v5), Path(a.tables))
