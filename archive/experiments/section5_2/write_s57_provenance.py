# Section 5.7 provenance backfill.
#
# run_node_failure.py builds its own Mesh and writes its own save block; it does
# NOT call runner.run_mesh_experiment. Consequence: the regeneration's provenance
# fields (sample_target, excluded_rotation_ranges, track_compute_cost) and its
# extra arrays (cell_nig_steps, cell_beliefs_steps, cell_hop_steps,
# cell_fused_steps) never reach these runs. The runs are CORRECT -- the Mesh
# constructor picks up sample_target=True by default, and EXCLUDED_RANGES is
# applied inside the script -- but nothing on disk records either fact.
#
# Undocumented provenance is what caused several of this session's problems, so
# this writes a note file per run directory rather than leaving it implicit. It
# does not modify any manifest or array.
#
# Run: python -u experiments/section5_2/write_s57_provenance.py

from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path("runs/chapter5_v2/s5_7_node_loss")
SCRIPT = Path("experiments/stage6_spatial_mesh/run_node_failure.py")


def script_constants():
    """Read EXCLUDED_RANGES / LR / LAM straight out of the script source, so the
    note reflects what actually ran rather than what this file assumes."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("EXCLUDED_RANGES", "LR", "LAM", "N_WEARABLES", "SEED"):
                try:
                    out[name] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


def main():
    consts = script_constants()
    dirs = sorted(d for d in ROOT.iterdir() if (d / "manifest.yaml").exists())
    for d in dirs:
        m = yaml.safe_load(open(d / "manifest.yaml"))
        seed = int(d.name.rsplit("seed", 1)[1])
        note = {
            "provenance_note": "Section 5.7 regeneration, 2026-08-28",
            "why_this_file_exists":
                "run_node_failure.py does not call runner.run_mesh_experiment, so "
                "the regeneration's manifest provenance fields and extra per-cell "
                "arrays are absent from this run directory. The run is correct; "
                "it is simply undocumented by its own manifest.",
            "sample_target": True,
            "sample_target_source":
                "Mesh.__init__ default (adopted 2026-08-27); run_node_failure.py "
                "constructs Mesh directly and passes no override",
            "draws_per_target": 1,
            "excluded_rotation_ranges": consts.get("EXCLUDED_RANGES"),
            "excluded_rotation_ranges_applied": bool(consts.get("EXCLUDED_RANGES")),
            "excluded_rotation_ranges_source": "EXCLUDED_RANGES in run_node_failure.py",
            "seed": seed,
            "seed_set": [42, 1042, 2042, 3042, 4042],
            "lr": consts.get("LR"), "lam": consts.get("LAM"),
            "n_wearables": consts.get("N_WEARABLES"),
            "mesh_mode": m.get("mesh_mode", "nig_product"),
            "save_block_divergence": {
                "calls_run_mesh_experiment": False,
                "arrays_saved": ["cell_mse_steps", "cell_cert_steps", "coverage_count"],
                "arrays_absent": ["cell_nig_steps", "cell_beliefs_steps",
                                  "cell_hop_steps", "cell_fused_steps"],
                "consequence":
                    "90% interval coverage and the half-width:RMS ratio are NOT "
                    "computable for this section: the Student-t interval needs "
                    "df=2*alpha and scale=sqrt(beta*(1+nu)/(nu*alpha)) as separate "
                    "quantities, and the stored certainty collapses them into one "
                    "non-invertible scalar.",
                "future_repo_note":
                    "Any future change to runner.py's save block will NOT reach "
                    "run_node_failure.py. The two save paths must be kept in sync "
                    "manually, or the script migrated onto run_mesh_experiment.",
            },
            "reporting_instrument":
                "Mean MSE is not the appropriate outcome under node loss: it is a "
                "nanmean that silently excludes dead cells, so it measures the "
                "surviving region while appearing to be a system-wide number. This "
                "section reports dead-zone size and the paired vs-control "
                "comparison over cells retaining coverage in both conditions.",
        }
        (d / "PROVENANCE.yaml").write_text(
            yaml.safe_dump(note, sort_keys=False, default_flow_style=False),
            encoding="utf8")
    print(f"wrote PROVENANCE.yaml into {len(dirs)} run directories under {ROOT}")
    print(f"  exclusions recorded: {consts.get('EXCLUDED_RANGES')}")
    print(f"  lr={consts.get('LR')} lam={consts.get('LAM')} "
          f"n_wearables={consts.get('N_WEARABLES')}")


if __name__ == "__main__":
    main()
