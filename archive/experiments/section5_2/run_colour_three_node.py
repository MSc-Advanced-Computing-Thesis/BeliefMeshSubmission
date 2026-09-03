# Section 5.3 -- three-node COLOUR-world aggregation experiment.
#
# NEW script, NEW output root. No existing code path modified.
#
# WHY. The offset world cannot produce evidence asymmetry between contributors:
# target displacement leaves the INPUT unchanged, so the forward pass is
# bit-identical and reported uncertainty cannot respond (Section 5.1). The
# aggregation rules are therefore indistinguishable there by construction. The
# colour world changes the input, so uncertainty does respond, and this is
# where the rules can differ.
#
# GEOMETRY (cx, cy) = (column, row); fov_cells(cx, cy) returns (row, col).
#   A = (8, 10)   anchor, red end
#   C = (12, 10)  anchor, blue end
#   B = (10, 13)  the third node -- overlaps both, receives no ground truth
# Triple-overlap 12 cells (rows 10-13, cols 9-11); exclusive A 20, C 20, B 21.
#
# ENVIRONMENT. Colour varies with COLUMN, red at A's centre column through to
# blue at C's, so the shared region is rendered at intermediate strengths and
# the two anchors report genuinely different evidence on the same cells:
#     v(col) = clip((cC - col) / (cC - cA), 0, 1)      1.0 = red, 0.0 = blue
# apply_filter_from_env_value maps v=1 -> (r 1.0, b 0.2) and v=0 -> (r 0.2,
# b 1.0). Constant in time, so colour is the ONLY source of variation.
# offset_field is None, so labels carry no displacement -- verified against
# _label, which reads only the angle and the offset field, never the colour.
#
# ARMS. frozen, naive, certainty, nig_product, and nig_product with AVERAGED
# fusion in TRAINING (w_i = 1/N, injected by wrapping fuse_nig_product for the
# duration of that arm only and restoring afterwards).
#
# Run: python -u experiments/section5_2/run_colour_three_node.py --verify
#      python -u experiments/section5_2/run_colour_three_node.py --seeds 42,...

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

import beliefmesh.fusion.nig_product as nigmod
from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.node.mesh import fov_cells

_ORIG_FUSE = nigmod.fuse_nig_product

G, FOV, T = 22, 7, 390
A_CENTRE, C_CENTRE, B_CENTRE = (8, 10), (12, 10), (10, 13)
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
EXCLUDED = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
LAM, LR, BASE_SEED = 5.0, 3e-5, 42
OUT = Path("runs/chapter5_v2/s5_3_colour_three_node")
ARMS = ["frozen", "naive", "certainty", "nig_product", "nig_product_avgtrain"]


def averaged_fuse(beliefs, weights=None):
    if weights is None and len(beliefs) > 1:
        weights = [1.0 / len(beliefs)] * len(beliefs)
    return _ORIG_FUSE(beliefs, weights)


def geometry():
    a = set(fov_cells(*A_CENTRE, FOV, G))
    c = set(fov_cells(*C_CENTRE, FOV, G))
    b = set(fov_cells(*B_CENTRE, FOV, G))
    return a, c, b


def colour_field():
    """v(col): 1.0 red at A's column, 0.0 blue at C's, linear between."""
    cA, cC = A_CENTRE[0], C_CENTRE[0]
    cols = np.arange(G, dtype=float)
    v = np.clip((cC - cols) / float(cC - cA), 0.0, 1.0)
    return np.tile(v[None, None, :], (T, G, 1))


def wearable_paths(a_only, c_only, seed):
    """Deterministic cyclic sweeps, each anchor confined to its OWN exclusive
    coloured region -- same pattern as run_peer_supervision.wearable_paths.
    Neither anchor enters the shared region; B receives no wearable at all."""
    rng = np.random.default_rng(seed)
    out = []
    for cells in (sorted(a_only), sorted(c_only)):
        x = list(cells)
        rng.shuffle(x)
        out.append(np.array([x[t % len(x)] for t in range(T)], dtype=int))
    return out


def build_kwargs(arm, seed, root):
    a, c, b = geometry()
    centres = np.array([A_CENTRE, C_CENTRE, B_CENTRE])
    grids = colour_field()
    paths = wearable_paths(a - c - b, c - a - b, 200 if seed == BASE_SEED else seed + 200)
    mode = "nig_product" if arm == "nig_product_avgtrain" else arm
    tag = "%s_seed%d" % (arm, seed)
    return dict(
        condition=tag, run_dir=Path(root) / tag,
        all_grids=grids, wearable_paths=paths, node_centres=centres,
        fov_size=FOV, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title="Colour three-node (%s, seed=%d)" % (arm, seed),
        offset_field=None, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        sample_target=True, draws_per_target=1,
        track_disagreement=(mode == "nig_product"),
        apply_colour_filter=True,
    )


def report_geometry():
    a, c, b = geometry()
    tri = a & c & b
    v = colour_field()[0]
    print("=" * 88)
    print("GEOMETRY  (cx, cy) = (column, row);  grid %dx%d, FOV %d" % (G, G, FOV))
    print("=" * 88)
    print("  A (red anchor)  centre %s   %d cells" % (A_CENTRE, len(a)))
    print("  C (blue anchor) centre %s   %d cells" % (C_CENTRE, len(c)))
    print("  B (third node)  centre %s   %d cells" % (B_CENTRE, len(b)))
    print("  pairwise overlap : A&C %d   A&B %d   C&B %d" % (len(a & c), len(a & b), len(c & b)))
    print("  TRIPLE overlap   : %d cells, rows %d-%d cols %d-%d"
          % (len(tri), min(r for r, _ in tri), max(r for r, _ in tri),
             min(cc for _, cc in tri), max(cc for _, cc in tri)))
    print("  exclusive        : A %d   C %d   B %d"
          % (len(a - c - b), len(c - a - b), len(b - a - c)))
    f = lambda s: (min(v[r, cc] for r, cc in s), max(v[r, cc] for r, cc in s))
    print("\n  colour value (1.0 = red, 0.0 = blue), min..max per region:")
    print("    A exclusive %.2f .. %.2f" % f(a - c - b))
    print("    C exclusive %.2f .. %.2f" % f(c - a - b))
    print("    triple      %.2f .. %.2f" % f(tri))
    print("  offset field: None (labels carry no displacement)")


def verify(seed=BASE_SEED):
    """Capture the kwargs each arm WOULD receive and report the differences."""
    calls = {arm: build_kwargs(arm, seed, "/tmp/x") for arm in ARMS}

    def eq(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.array_equal(np.asarray(x), np.asarray(y))
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y))
        return x == y

    print("\n" + "=" * 88)
    print("VERIFICATION -- kwargs capture, %d arms at seed %d" % (len(ARMS), seed))
    print("=" * 88)
    assert len(calls) == len(ARMS) and len(ARMS) > 0, "NO CALLS CAPTURED"
    ref = calls["nig_product"]
    NAMING = {"condition", "run_dir", "title"}
    ok = True
    for arm in ARMS:
        d = sorted(k for k in set(ref) | set(calls[arm])
                   if k not in NAMING and not eq(ref[k], calls[arm][k]))
        expect = [] if arm in ("nig_product", "nig_product_avgtrain") else ["mode"]
        if arm == "nig_product":
            expect = []
        if arm in ("naive", "certainty", "frozen"):
            expect = sorted(set(["mode", "track_disagreement"]))
        status = "OK" if d == expect else "UNEXPECTED"
        if d != expect:
            ok = False
        print("  %-24s vs nig_product: differing kwargs = %-40s [%s]"
              % (arm, d if d else "NONE", status))
    print("\n  naming-only keys ignored: %s" % sorted(NAMING))
    print("  nig_product_avgtrain has IDENTICAL kwargs to nig_product -- it differs")
    print("  ONLY by the fuse_nig_product wrapper, which is not a kwarg.")
    b = _ORIG_FUSE([(0.1, 2.0, 3.0, .05), (0.12, 4.0, 3.5, .06)])
    w = averaged_fuse([(0.1, 2.0, 3.0, .05), (0.12, 4.0, 3.5, .06)])
    print("  wrapper check: nu* %.4f -> %.4f (mean), gamma* |d| %.2e"
          % (b[1], w[1], abs(b[0] - w[0])))
    print("\nVERIFY %s" % ("PASSED" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--root", type=str, default=str(OUT))
    a = ap.parse_args()

    report_geometry()
    if a.verify:
        sys.exit(0 if verify() else 1)

    from stage6_spatial_mesh.runner import run_mesh_experiment
    from beliefmesh.config import load_config

    for seed in [int(x) for x in a.seeds.split(",")]:
        for arm in ARMS:
            cfg = load_config()
            cfg.model.lr = LR
            kw = build_kwargs(arm, seed, a.root)
            patched = arm == "nig_product_avgtrain"
            if patched:
                nigmod.fuse_nig_product = averaged_fuse
                import stage6_spatial_mesh.runner as rn
                rn.fuse_nig_product = averaged_fuse
            try:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                res = run_mesh_experiment(cfg, **kw)
                print("### %s last50=%.5f" % (kw["condition"], res["mean_mse_last_50"]),
                      flush=True)
            finally:
                if patched:
                    nigmod.fuse_nig_product = _ORIG_FUSE
                    import stage6_spatial_mesh.runner as rn
                    rn.fuse_nig_product = _ORIG_FUSE
    print("=== DONE ===")


if __name__ == "__main__":
    main()
