# Section 5.1 -- interval coverage on a fine nominal grid.
#
# Pure recomputation from the existing per-sample dump; no model, no dataset,
# no evaluation. Uses analyse.interval_coverage unchanged, so the construction
# is bit-for-bit the same one that produced the four-level table: location
# gamma, df 2*alpha, scale sqrt(beta*(1+nu)/(nu*alpha)), containment tested on
# the WRAPPED circular error, alpha-floor-bound samples excluded and counted.
#
# Run: python -u experiments/section5_1/run_coverage_grid.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import interval_coverage, read_dump, write_rows_csv

SRC = Path("runs/section5_1/r1a/per_sample_baseline.csv")
OUT = Path("runs/section5_1/r1a")
LEVELS = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95
# Above this share of intervals spanning the whole circle, the level's coverage
# is not a calibration measurement -- containment is satisfied by construction.
VACUOUS_THRESHOLD = 0.01


def main():
    d = read_dump(SRC)
    rows = interval_coverage(d, levels=LEVELS)
    for r in rows:
        r["vacuous"] = bool(r["frac_interval_covers_circle"] > VACUOUS_THRESHOLD)
        r["signed_deviation"] = r["empirical"] - r["nominal"]
    write_rows_csv(rows, OUT / "interval_coverage_fine.csv")

    vac = [r for r in rows if r["vacuous"]]
    dev = np.array([r["signed_deviation"] for r in rows])
    summary = {
        "source": str(SRC),
        "n_samples": int(len(d["circular_error_norm"])),
        "levels": LEVELS,
        "construction": "location gamma, df 2*alpha, scale "
                        "sqrt(beta*(1+nu)/(nu*alpha)); containment tested on "
                        "the wrapped circular error (same as the four-level "
                        "table)",
        "alpha_floor_excluded": rows[0]["n_excluded_alpha_floor"],
        "vacuous_threshold_frac_circle": VACUOUS_THRESHOLD,
        "vacuous_levels": [r["nominal"] for r in vac],
        "max_frac_interval_covers_circle": float(
            max(r["frac_interval_covers_circle"] for r in rows)),
        "signed_deviation": {
            "max_over": float(dev.max()),
            "max_over_at_nominal": float(rows[int(dev.argmax())]["nominal"]),
            "max_under": float(dev.min()),
            "max_under_at_nominal": float(rows[int(dev.argmin())]["nominal"]),
            "mean_abs": float(np.abs(dev).mean()),
        },
        "rows": rows,
    }
    with open(OUT / "interval_coverage_fine.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)

    print("%8s %10s %9s %12s %10s %8s"
          % ("nominal", "empirical", "dev", "halfwidth", "fullcircle", "vacuous"))
    for r in rows:
        print("%8.2f %10.4f %+9.4f %10.1f deg %9.4f %8s"
              % (r["nominal"], r["empirical"], r["signed_deviation"],
                 r["mean_half_width_deg"], r["frac_interval_covers_circle"],
                 "YES" if r["vacuous"] else ""))
    print("\nalpha-floor excluded: %d" % rows[0]["n_excluded_alpha_floor"])
    print("max full-circle fraction: %.4f  -> vacuous levels: %s"
          % (summary["max_frac_interval_covers_circle"],
             summary["vacuous_levels"] or "none"))
    print("wrote %s" % (OUT / "interval_coverage_fine.csv"))


if __name__ == "__main__":
    main()
