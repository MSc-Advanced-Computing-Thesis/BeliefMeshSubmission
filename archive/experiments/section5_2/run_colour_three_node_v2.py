# Section 5.3 -- three-node colour-world experiment, VERSION 2.
#
# NEW script, NEW output root. Nothing existing modified.
#
# NAMING DIFFERS FROM V1. Here A = RED anchor, B = BLUE anchor, C = STUDENT.
# Version 1 (runs/chapter5_v2/s5_3_colour_three_node) used A red, C blue,
# B student. The stored v1 artefacts keep the v1 naming.
#
# WHY V2. V1 produced a large evidence asymmetry (nu-CV 0.55, gamma spread
# 20.8 deg against the offset world's 0.101 and 2-5 deg) and the aggregation
# arms still did not separate. The reason, measured after the fact:
#     r(contributor nu, contributor error) = +0.18 to +0.23
# Positive. The contributor claiming MORE evidence is the one that is MORE
# wrong, so nu-weighted fusion is steered toward the worse contributor.
#
# SUCCESS CRITERION for v2: r(nu, error) < 0 on the student's shared cells.
# Not |r| > 0 -- v1 already had a non-zero correlation, of the wrong sign.
# A failure to flip is reportable and is the stronger finding: it would say the
# evidence term does not track adaptation to local conditions even when
# adaptation differs sharply between contributors.
#
# DESIGN. Each anchor trains ONLY on its own exclusive colour region, so each
# specialises. A&B -- the region both anchors cover -- is excluded from both
# their training (it is not in either exclusive set, and the wearables never
# leave those sets). The student C sits below and between, learns only from
# cells it shares with the anchors, and those cells are rendered across the
# FULL filter range so the in-distribution anchor differs by cell.
#
# Run: python -u experiments/section5_2/run_colour_three_node_v2.py --verify
#      python -u experiments/section5_2/run_colour_three_node_v2.py --seeds ...

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
from beliefmesh.node.mesh import fov_cells

_ORIG_FUSE = nigmod.fuse_nig_product

G, FOV, T = 22, 7, 390
A_CENTRE, B_CENTRE, C_CENTRE = (8, 9), (13, 9), (10, 12)   # (cx, cy)
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
EXCLUDED = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
LAM, LR, BASE_SEED = 5.0, 3e-5, 42
OUT = Path("runs/chapter5_v2/s5_3_colour_three_node_v2")
ARMS = ["frozen", "naive", "certainty", "nig_product", "nig_product_avgtrain"]


def averaged_fuse(beliefs, weights=None):
    if weights is None and len(beliefs) > 1:
        weights = [1.0 / len(beliefs)] * len(beliefs)
    return _ORIG_FUSE(beliefs, weights)


def regions():
    a = set(fov_cells(*A_CENTRE, FOV, G))
    b = set(fov_cells(*B_CENTRE, FOV, G))
    c = set(fov_cells(*C_CENTRE, FOV, G))
    return a, b, c


def colour_field():
    """Ramp across the STUDENT's shared column span, with the anchors' own
    exclusive regions forced to the pure extremes so each specialises:
        A exclusive -> 1.0 (red)      B exclusive -> 0.0 (blue)
    The student's shared cells therefore sweep the full range, so the
    in-distribution anchor differs from cell to cell."""
    a, b, c = regions()
    shared = c & (a | b)
    lo = min(cc for _, cc in shared)
    hi = max(cc for _, cc in shared)
    cols = np.arange(G, dtype=float)
    v = np.clip((hi - cols) / float(hi - lo), 0.0, 1.0)
    grid = np.tile(v[None, :], (G, 1))
    for r, cc in (a - b - c):
        grid[r, cc] = 1.0
    for r, cc in (b - a - c):
        grid[r, cc] = 0.0
    return np.tile(grid[None], (T, 1, 1))


def wearable_paths(a_only, b_only, seed):
    """Each anchor confined to its OWN exclusive region -- so A&B is excluded
    from both, and the student receives no ground truth anywhere."""
    rng = np.random.default_rng(seed)
    out = []
    for cells in (sorted(a_only), sorted(b_only)):
        x = list(cells)
        rng.shuffle(x)
        out.append(np.array([x[t % len(x)] for t in range(T)], dtype=int))
    return out


def build_kwargs(arm, seed, root):
    a, b, c = regions()
    centres = np.array([A_CENTRE, B_CENTRE, C_CENTRE])
    paths = wearable_paths(a - b - c, b - a - c,
                           200 if seed == BASE_SEED else seed + 200)
    mode = "nig_product" if arm == "nig_product_avgtrain" else arm
    tag = "%s_seed%d" % (arm, seed)
    return dict(
        condition=tag, run_dir=Path(root) / tag,
        all_grids=colour_field(), wearable_paths=paths, node_centres=centres,
        fov_size=FOV, mode=mode, baseline_checkpoint=CKPT,
        n_wearable_samples=1, n_train_repeats=1,
        title="Colour three-node v2 (%s, seed=%d)" % (arm, seed),
        offset_field=None, env_seed=seed,
        excluded_rotation_ranges=EXCLUDED,
        lam=LAM, wearable_policy=None, policy_step_size=0.4,
        sample_target=True, draws_per_target=1,
        track_disagreement=(mode == "nig_product"),
        apply_colour_filter=True,
    )


def report_geometry():
    a, b, c = regions()
    v = colour_field()[0]
    shared = sorted(c & (a | b))
    print("=" * 92)
    print("V2 GEOMETRY   NAMING: A = RED anchor, B = BLUE anchor, C = STUDENT")
    print("  (v1 used A red, C blue, B student -- different, stored separately)")
    print("=" * 92)
    for lab, ctr, s in (("A (red anchor)", A_CENTRE, a), ("B (blue anchor)", B_CENTRE, b),
                        ("C (student)", C_CENTRE, c)):
        print("  %-16s centre %-9s %d cells" % (lab, str(ctr), len(s)))
    print("  A&B  %2d  <- EXCLUDED from BOTH anchors' training" % len(a & b))
    print("  A&C  %2d   B&C  %2d   triple %2d" % (len(a & c), len(b & c), len(a & b & c)))
    print("  exclusive: A %d   B %d   C %d" % (len(a - b - c), len(b - a - c), len(c - a - b)))
    print("  C cells shared with an anchor: %d" % len(shared))
    f = lambda s_: (min(v[r, cc] for r, cc in s_), max(v[r, cc] for r, cc in s_))
    print("\n  REALISED colour value (1.0 = red, 0.0 = blue):")
    print("    A exclusive        %.3f .. %.3f" % f(a - b - c))
    print("    B exclusive        %.3f .. %.3f" % f(b - a - c))
    print("    A&B (excluded)     %.3f .. %.3f" % f(a & b))
    print("    C shared w/ anchor %.3f .. %.3f   <- the bias sweep" % f(shared))
    per_col = {}
    for r, cc in shared:
        per_col.setdefault(cc, v[r, cc])
    print("    C shared, per column: " +
          "  ".join("c%d=%.2f" % (k, per_col[k]) for k in sorted(per_col)))
    n_red = sum(1 for r, cc in shared if v[r, cc] > 0.5)
    print("    cells favouring A (v>0.5): %d;  favouring B (v<0.5): %d;  v==0.5: %d"
          % (n_red, sum(1 for r, cc in shared if v[r, cc] < 0.5),
             sum(1 for r, cc in shared if abs(v[r, cc] - 0.5) < 1e-9)))
    print("  offset field: None")


def verify(seed=BASE_SEED):
    calls = {arm: build_kwargs(arm, seed, "/tmp/x") for arm in ARMS}

    def eq(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.array_equal(np.asarray(x), np.asarray(y))
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y))
        return x == y

    print("\n" + "=" * 92)
    print("VERIFICATION -- kwargs capture, %d arms at seed %d" % (len(ARMS), seed))
    print("=" * 92)
    assert len(calls) == len(ARMS) and ARMS, "NO CALLS CAPTURED"
    ref = calls["nig_product"]
    NAMING = {"condition", "run_dir", "title"}
    ok = True
    for arm in ARMS:
        d = sorted(k for k in set(ref) | set(calls[arm])
                   if k not in NAMING and not eq(ref[k], calls[arm][k]))
        expect = [] if arm in ("nig_product", "nig_product_avgtrain") \
            else ["mode", "track_disagreement"]
        if d != expect:
            ok = False
        print("  %-24s differing kwargs = %-36s [%s]"
              % (arm, d if d else "NONE", "OK" if d == expect else "UNEXPECTED"))
    print("\n  naming-only keys ignored: %s" % sorted(NAMING))
    print("  nig_product_avgtrain has IDENTICAL kwargs -- it differs ONLY by the")
    print("  fuse_nig_product wrapper, which is not a kwarg.")
    bl = [(0.1, 2.0, 3.0, .05), (0.12, 4.0, 3.5, .06)]
    s_, w_ = _ORIG_FUSE(bl), averaged_fuse(bl)
    print("  wrapper: nu* %.4f -> %.4f (mean of %s); gamma* |d| %.2e"
          % (s_[1], w_[1], [x[1] for x in bl], abs(s_[0] - w_[0])))
    assert nigmod.fuse_nig_product is _ORIG_FUSE, "wrapper leaked into the module"
    print("  module-level fuse_nig_product is UNPATCHED outside its own arm: OK")
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
    import stage6_spatial_mesh.runner as rn

    for seed in [int(x) for x in a.seeds.split(",")]:
        for arm in ARMS:
            cfg = load_config()
            cfg.model.lr = LR
            kw = build_kwargs(arm, seed, a.root)
            patched = arm == "nig_product_avgtrain"
            if patched:
                nigmod.fuse_nig_product = averaged_fuse
                rn.fuse_nig_product = averaged_fuse
            try:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                res = run_mesh_experiment(cfg, **kw)
                print("### %s last50=%.5f" % (kw["condition"], res["mean_mse_last_50"]),
                      flush=True)
            finally:
                nigmod.fuse_nig_product = _ORIG_FUSE
                rn.fuse_nig_product = _ORIG_FUSE
    print("=== DONE ===")


if __name__ == "__main__":
    main()
