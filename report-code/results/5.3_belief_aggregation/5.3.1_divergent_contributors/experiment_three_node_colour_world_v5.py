# Three-node colour world, version 5: v4 with an anchor burn-in.
#
# WHY. In v4 the two anchors start from the same pretrained checkpoint, which
# responds asymmetrically to the red and blue ends of the filter range
# (Section 5.1), so the anchors begin unequally adapted and the student's
# supervision is confounded with that asymmetry. Here each anchor first adapts
# to its own condition before its beliefs are used.
#
# CHANGE FROM v4. Per seed, a burn-in of BURN_IN timesteps in which only the two
# anchors exist and train on their own confined wearables (the student is
# absent: it neither trains nor receives beliefs). Their model states are saved
# and then loaded into the anchors of the run proper, which is v4 exactly: the
# same geometry, colour field, wearable paths, five arms, sampled targets,
# draws 1, lam 5.0, 390 timesteps. The student starts from the pretrained
# checkpoint as in v4. The frozen arm never trains, so it keeps the pretrained
# checkpoint for all three nodes, as in v4.
#
# Output layout: <root>/burnin/seed<seed>/ (2-node burn-in run with
# checkpoints/node_{0,1}.pth) and <root>/<arm>_seed<seed>/ (as v4).
#
# Run: python -u experiment_three_node_colour_world_v5.py --seeds 42,1042,2042,3042,4042

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR

import argparse
import random
from pathlib import Path

import numpy as np
import torch

import beliefmesh.fusion.nig_product as nigmod
import beliefmesh.simulation.runner as rn
from beliefmesh.config import load_config
from beliefmesh.simulation.runner import run_mesh_experiment
import experiment_three_node_colour_world as V4

BURN_IN = 300
OUT = ART_DIR / "5.3_belief_aggregation" / "5.3.1_divergent_contributors" / "three_node_colour_world_v5"
ARMS = V4.ARMS


def burnin(seed: int, root: Path) -> list[Path]:
    """Anchors A and B alone for BURN_IN steps, each on its own wearable
    confined to its exclusive region. Returns the two checkpoint paths."""
    a, b, c = V4.regions()
    cfg = load_config()
    cfg.model.lr = V4.LR
    paths = [p[:BURN_IN] for p in
             V4.wearable_paths(a - b - c, b - a - c, 300 if seed == V4.BASE_SEED else seed + 300)]
    run_dir = root / "burnin" / ("seed%d" % seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    run_mesh_experiment(
        cfg, condition="burnin_seed%d" % seed, run_dir=run_dir,
        all_grids=V4.colour_field()[:BURN_IN], wearable_paths=paths,
        node_centres=np.array([V4.A_CENTRE, V4.B_CENTRE]),
        fov_size=V4.FOV, mode="nig_product", baseline_checkpoint=V4.CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title="Colour three-node v5 burn-in (anchors only, seed=%d)" % seed,
        offset_field=None, env_seed=seed, excluded_rotation_ranges=V4.EXCLUDED,
        lam=V4.LAM, wearable_policy=None, policy_step_size=0.4,
        sample_target=True, draws_per_target=1, track_disagreement=False,
        apply_colour_filter=True, save_node_checkpoints=True)
    return [run_dir / "checkpoints" / "node_0.pth", run_dir / "checkpoints" / "node_1.pth"]


def main(seeds, root: Path):
    for seed in seeds:
        ck = burnin(seed, root)
        for arm in ARMS:
            cfg = load_config()
            cfg.model.lr = V4.LR
            kw = V4.build_kwargs(arm, seed, root)
            if arm != "frozen":
                kw["baseline_checkpoint"] = [ck[0], ck[1], V4.CKPT]   # A, B burned in; C pretrained
            kw["title"] = "Colour three-node v5 (%s, seed=%d, anchors burned in %d steps)" % (arm, seed, BURN_IN)
            kw["extra_manifest"] = {"burn_in_steps": BURN_IN, "anchor_checkpoints": [str(p) for p in ck]}
            patched = arm == "nig_product_avgtrain"
            if patched:
                nigmod.fuse_nig_product = V4.averaged_fuse
                rn.fuse_nig_product = V4.averaged_fuse
            try:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                res = run_mesh_experiment(cfg, **kw)
                print("### %s last50=%.5f" % (kw["condition"], res["mean_mse_last_50"]), flush=True)
            finally:
                nigmod.fuse_nig_product = V4._ORIG_FUSE
                rn.fuse_nig_product = V4._ORIG_FUSE
    print("=== DONE ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Three-node colour world v5: v4 with a %d-step anchor burn-in (GPU)" % BURN_IN)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--root", type=str, default=str(OUT))
    a = ap.parse_args()
    main([int(x) for x in a.seeds.split(",")], Path(a.root))
