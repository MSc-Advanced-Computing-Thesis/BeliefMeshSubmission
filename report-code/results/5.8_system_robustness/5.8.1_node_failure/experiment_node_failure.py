# Node-failure resilience experiment (2026-08). Objective: demonstrate the
# mesh degrades gracefully under permanent node loss -- localised, not
# system-wide, and self-reported via certainty rather than silent.
#
# REVISION (2026-08, Christian's explicit correction after the seed-42
# single-seed result): the original "mean MSE over time" comparison is
# uninterpretable and has been DROPPED. MSE falls throughout every run
# (including the 0% control) purely from continued training, and the
# reported mean silently excludes NaN cells (nanmean), so it was measuring
# "how well is the surviving+covered region doing" while looking like a
# system-wide number. Replaced with:
#   - dead-zone size (cells with ZERO surviving coverage) as the PRIMARY
#     outcome -- monotonic, clean, matches physical intuition that
#     correlated/clustered damage is worse than scattered damage.
#   - a paired comparison against the 0% control, restricted to cells that
#     retain coverage in BOTH conditions, at matched post-failure timesteps
#     -- isolates whether the SURVIVING region is affected at all, with
#     dead zones (which cannot be compared -- there's nothing to compare)
#     accounted for separately via dead-zone size above.
#   - the coverage-count relationship reported as a THRESHOLD (any surviving
#     coverage vs none) rather than a gradient -- the single-seed data
#     showed no proportional response once coverage >= 1, only a cliff at 0.
#   - a distance-from-dead-zone-boundary check (secondary, cheap, computed
#     from the already-saved per-cell arrays -- no extra simulation) to see
#     whether a transition zone exists right at the dead-zone edge that the
#     coverage-count bins are too coarse to resolve.
# Connectivity and unreachable-node counts are kept as before.
#
# Same field/seed/wearable-policy/node-placement as every other seed-42
# comparison this session (7x7 stride 3, random_wander, offset world) so
# results sit on the same axis as the existing nig_product/fusion numbers.
# Failure triggers at step 150/390 (after burn-in) via Mesh.fail_nodes(),
# which permanently strips the failed nodes from the overlap graph -- see
# that method's docstring in src/beliefmesh/node/mesh.py.
#
# Run: python -u experiments/stage6_spatial_mesh/run_node_failure.py \
#          --failure-mode {random,clustered,none} --fraction 0.25 --seed 42
# (run the --failure-mode none --fraction 0.0 control FIRST for a given
#  seed -- every other condition for that seed loads its saved arrays for
#  the paired comparison.)

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RESULTS = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_RESULTS), str(_RESULTS.parent), str(_Path(__file__).resolve().parent)]
from _shared.paths import ARTEFACTS as ART_DIR, FIGURES as FIG_DIR, TABLES as TAB_DIR
from beliefmesh.simulation.assets import CHECKPOINTS, ENVIRONMENT_DIR
ART = str(ART_DIR).replace("\\", "/")

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, ttest_rel

from beliefmesh.simulation.assets import git_commit
from beliefmesh.simulation.offset_fields import (build_dynamic_offset_field,
                                                         build_random_wander_path)

from beliefmesh.config import load_config
from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.fusion.nig_product import fuse_nig_product
from beliefmesh.metrics.circular import circular_diff
from beliefmesh.node.mesh import Mesh

ENV = ENVIRONMENT_DIR
CKPT = CHECKPOINTS["baseline"]
ROOT = Path(ART + "/5.8_system_robustness/5.8.1_node_failure/node_loss")   # overridable via --root
N_WEARABLES = 3
LR = 3e-5
LAM = 5.0
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]
FAILURE_STEP = 150
PRE_WINDOW = (100, 150)     # control-state reference window, before failure
POST_WINDOW = (340, 390)    # stabilised-after-failure window
GRID_DIM = 6                # 36 nodes laid out as a 6x6 grid (node_centres.npy order)


def random_failure_nodes(fraction: float, n_nodes: int = 36, seed: int = 42) -> list[int]:
    target = round(fraction * n_nodes)
    if target == 0:
        return []
    rng = np.random.default_rng(seed + 1000)  # distinct stream from the run seed
    return sorted(rng.choice(n_nodes, size=target, replace=False).tolist())


def clustered_failure_nodes(fraction: float, n_nodes: int = 36, grid_dim: int = GRID_DIM) -> list[int]:
    """Contiguous block grown row-major from node index 0 (node_centres.npy's
    construction order: index i -> row i//grid_dim, col i%grid_dim, so
    consecutive indices ARE spatially adjacent). A simple growing rectangle
    (full rows, then a partial final row) rather than a graph-BFS blob --
    still a single 4-connected region for any target count, fully
    deterministic, and reads as "one corner of the grid went dark", the
    localised-damage scenario the spec asks for."""
    target = round(fraction * n_nodes)
    if target == 0:
        return []
    cols = grid_dim
    full_rows = target // cols
    rem = target % cols
    selected = list(range(full_rows * cols))
    selected += list(range(full_rows * cols, full_rows * cols + rem))
    return selected


def build_env_and_mesh(mesh_mode: str, seed: int):
    cfg = load_config()
    cfg.model.lr = LR
    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = build_dynamic_offset_field(G, T)  # deterministic, same field every condition/seed
    wearable_seed_base = 200 if seed == 42 else seed + 200
    paths = [build_random_wander_path(T, G, seed=wearable_seed_base + i) for i in range(N_WEARABLES)]

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = GridEnvironment(G, uniform, offset_field=field, rotation_seed=seed,
                          excluded_rotation_ranges=EXCLUDED_RANGES)
    mesh = Mesh(centres, fov_size=7, grid_size=G, environment=env,
                pretrained_path=CKPT, lr=LR, fusion_grid=circular_grid(cfg.fusion.grid_size),
                mode=mesh_mode, sample_seed=seed, lam=LAM)
    return env, mesh, paths, T, G


def control_dir_for(mesh_mode: str, seed: int) -> Path:
    """The 0% condition is shared between random/clustered (no nodes fail
    either way) and is always tagged with failure_mode='random' -- see
    run_condition()'s fraction==0 special-case."""
    return ROOT / f"{mesh_mode}_random_0pct_seed{seed}"


def compute_stats(cell_mse_steps: np.ndarray, cell_cert_steps: np.ndarray,
                   coverage_count: np.ndarray, G: int,
                   control_mse: np.ndarray | None, control_cert: np.ndarray | None):
    """All post-hoc analysis, computed purely from the per-cell arrays (and
    optionally a matched control run's arrays) -- no simulation needed, so
    this is reusable both live (run_condition) and for pure re-analysis of
    already-completed runs from their saved .npy files."""

    def avg_map(arr, lo, hi):
        with np.errstate(invalid="ignore"):
            return np.nanmean(arr[lo:hi], axis=0)

    pre_mse_map = avg_map(cell_mse_steps, *PRE_WINDOW)
    pre_cert_map = avg_map(cell_cert_steps, *PRE_WINDOW)
    post_mse_map = avg_map(cell_mse_steps, *POST_WINDOW)
    post_cert_map = avg_map(cell_cert_steps, *POST_WINDOW)

    def window_corr(lo, hi):
        m = cell_mse_steps[lo:hi]; c = cell_cert_steps[lo:hi]
        mask = ~np.isnan(m) & ~np.isnan(c)
        if mask.sum() < 3:
            return float("nan"), float("nan")
        return pearsonr(c[mask], m[mask])

    pre_r, pre_p = window_corr(*PRE_WINDOW)
    post_r, post_p = window_corr(*POST_WINDOW)

    # --- dead-zone size: PRIMARY outcome ------------------------------
    dead_zone_mask = coverage_count == 0
    dead_zone_cell_count = int(dead_zone_mask.sum())
    dead_zone_fraction = dead_zone_cell_count / (G * G)

    # --- threshold framing: any surviving coverage vs none -------------
    covered_mask = coverage_count >= 1
    def pooled(map_, mask):
        vals = map_[mask]; vals = vals[~np.isnan(vals)]
        return float(np.mean(vals)) if len(vals) else None
    threshold_summary = dict(
        covered_mean_mse=pooled(post_mse_map, covered_mask),
        covered_mean_cert=pooled(post_cert_map, covered_mask),
        dead_mean_mse=pooled(post_mse_map, dead_zone_mask),   # expect None -- nothing to average
        dead_mean_cert=pooled(post_cert_map, dead_zone_mask),
        n_covered_cells=int(covered_mask.sum()),
        n_dead_cells=dead_zone_cell_count,
    )

    # --- coverage-count breakdown (kept for transparency, NOT the headline) -
    coverage_breakdown = {}
    for b in sorted(set(coverage_count.flatten().tolist())):
        mask = coverage_count == b
        coverage_breakdown[int(b)] = dict(
            n_cells=int(mask.sum()),
            mean_mse=pooled(post_mse_map, mask),
            mean_cert=pooled(post_cert_map, mask),
        )

    # --- paired comparison vs 0% control, matched timesteps+cells ------
    paired = None
    if control_mse is not None:
        lo, hi = POST_WINDOW
        joint_mask = (~np.isnan(cell_mse_steps[lo:hi])) & (~np.isnan(control_mse[lo:hi]))
        n_pairs = int(joint_mask.sum())
        if n_pairs >= 3:
            # per-cell mean over the window (cell = unit of replication, not
            # cell-timestep, to avoid pseudo-replication from within-window
            # autocorrelation), restricted to cells covered on EVERY step of
            # the window in BOTH conditions (a clean "same cell judged both
            # ways" comparison rather than a shifting cell set per step).
            fail_win = cell_mse_steps[lo:hi]; ctrl_win = control_mse[lo:hi]
            always_covered = (~np.isnan(fail_win)).all(axis=0) & (~np.isnan(ctrl_win)).all(axis=0)
            n_cells_compared = int(always_covered.sum())
            if n_cells_compared >= 3:
                fail_cell_means = np.nanmean(fail_win[:, always_covered], axis=0)
                ctrl_cell_means = np.nanmean(ctrl_win[:, always_covered], axis=0)
                diffs = fail_cell_means - ctrl_cell_means
                t, p = ttest_rel(fail_cell_means, ctrl_cell_means)
                paired = dict(
                    n_cells_compared=n_cells_compared,
                    mean_diff=float(diffs.mean()),  # positive = worse than control
                    std_diff=float(diffs.std()),
                    t=float(t), p=float(p),
                )
            else:
                paired = dict(n_cells_compared=n_cells_compared, note="too few always-covered cells to compare")

    # --- distance-from-dead-zone-boundary (secondary, cheap) -----------
    distance_breakdown = None
    if dead_zone_cell_count > 0:
        dead_rc = np.argwhere(dead_zone_mask)
        rows, cols = np.mgrid[0:G, 0:G]
        all_rc = np.stack([rows.ravel(), cols.ravel()], axis=1).astype(float)
        # distance from every cell to its nearest dead-zone cell
        d = np.linalg.norm(all_rc[:, None, :] - dead_rc[None, :, :].astype(float), axis=2).min(axis=1)
        dist_map = d.reshape(G, G)
        distance_breakdown = {}
        bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 100)]
        for lo_d, hi_d in bins:
            mask = covered_mask & (dist_map >= lo_d) & (dist_map < hi_d)
            if mask.sum() == 0:
                continue
            distance_breakdown[f"{lo_d}-{hi_d}"] = dict(
                n_cells=int(mask.sum()),
                mean_mse=pooled(post_mse_map, mask),
                mean_cert=pooled(post_cert_map, mask),
            )

    return dict(
        pre_mse_map=pre_mse_map, pre_cert_map=pre_cert_map,
        post_mse_map=post_mse_map, post_cert_map=post_cert_map,
        pre_r=pre_r, pre_p=pre_p, post_r=post_r, post_p=post_p,
        dead_zone_cell_count=dead_zone_cell_count, dead_zone_fraction=dead_zone_fraction,
        threshold_summary=threshold_summary,
        coverage_breakdown=coverage_breakdown,
        paired_vs_control=paired,
        distance_breakdown=distance_breakdown,
    )


def make_figure(run_dir, failure_mode, fraction, mesh_mode, seed, stats, coverage_count):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Node failure: {failure_mode} {round(fraction*100)}% "
                f"({mesh_mode}, seed {seed})", fontsize=14, fontweight="bold")
    ax = axes[0, 0]
    cb = stats["coverage_breakdown"]
    if cb:
        xs = sorted(cb.keys())
        mse_ys = [cb[x]["mean_mse"] or 0 for x in xs]
        colors = ["black" if x == 0 else "crimson" for x in xs]
        ax.bar(xs, mse_ys, color=colors, alpha=0.8)
        ax.set_title("Post-failure MSE by surviving coverage count\n(threshold, not gradient -- black=dead zone)")
        ax.set_xlabel("# surviving nodes covering cell"); ax.set_ylabel("Mean MSE")
        ax.grid(alpha=0.3)
    ax = axes[0, 1]
    db = stats["distance_breakdown"]
    if db:
        xs = list(db.keys())
        mse_ys = [db[x]["mean_mse"] or 0 for x in xs]
        ax.bar(xs, mse_ys, color="steelblue", alpha=0.8)
        ax.set_title("Post-failure MSE by distance from dead-zone boundary")
        ax.set_xlabel("Distance (cells)"); ax.set_ylabel("Mean MSE"); ax.grid(alpha=0.3)
    else:
        ax.axis("off"); ax.set_title("No dead zone (control)")
    ax = axes[0, 2]
    pv = stats["paired_vs_control"]
    ax.axis("off")
    if pv and "mean_diff" in pv:
        txt = (f"Paired vs 0% control (post-failure window)\n\n"
               f"n cells compared: {pv['n_cells_compared']}\n"
               f"mean diff (failure - control): {pv['mean_diff']:+.5f}\n"
               f"std diff: {pv['std_diff']:.5f}\n"
               f"paired t={pv['t']:.3f}, p={pv['p']:.4f}\n\n"
               f"dead-zone cells: {stats['dead_zone_cell_count']} "
               f"({100*stats['dead_zone_fraction']:.1f}% of grid)")
    else:
        txt = f"dead-zone cells: {stats['dead_zone_cell_count']} ({100*stats['dead_zone_fraction']:.1f}% of grid)"
    ax.text(0.05, 0.5, txt, fontsize=11, va="center", family="monospace")
    for ax, data, cmap, label in [
        (axes[1, 0], stats["pre_mse_map"], "RdYlGn_r", "Pre-failure MSE"),
        (axes[1, 1], stats["post_mse_map"], "RdYlGn_r", "Post-failure MSE"),
        (axes[1, 2], stats["post_cert_map"], "RdYlGn", "Post-failure Certainty"),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(label); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(run_dir / "figure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_condition(failure_mode: str, fraction: float, seed: int = 42, mesh_mode: str = "nig_product"):
    tag = f"{mesh_mode}_{failure_mode}_{round(fraction*100)}pct_seed{seed}"
    run_dir = ROOT / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    env, mesh, paths, T, G = build_env_and_mesh(mesh_mode, seed)

    if fraction == 0:
        failure_ids = []
    elif failure_mode == "random":
        failure_ids = random_failure_nodes(fraction, n_nodes=len(mesh.nodes), seed=seed)
    elif failure_mode == "clustered":
        failure_ids = clustered_failure_nodes(fraction, n_nodes=len(mesh.nodes))
    else:
        raise ValueError(failure_mode)

    cell_mse_steps = np.full((T, G, G), np.nan)
    cell_cert_steps = np.full((T, G, G), np.nan)

    # SAVE-ONLY ADDITION (2026-09-03). This script writes its own save block
    # rather than calling runner.run_mesh_experiment, so the arrays the
    # AVERAGED readout needs -- every covering node's belief, not just the
    # argmax one -- were never written for this section, and Section 5.7 was
    # the only cell-space result in Chapter 5 still on the argmax estimator.
    #
    # Names, shapes and dtypes mirror runner.py's save block exactly so the
    # same readers work on both. Purely additive: the step_best selection
    # below is unchanged (same iteration order, same strict >), so
    # cell_mse_steps and cell_cert_steps are bit-identical to before -- which
    # --verify-inert asserts against the stored runs.
    #
    # Coverage is taken over ALL nodes, pre-failure, matching runner.py's
    # _max_cov semantics; failure only ever reduces the count.
    _cov_full = np.zeros((G, G), dtype=int)
    for _i in mesh.nodes:
        for (_r, _c) in mesh.nodes[_i].fov_cells:
            _cov_full[_r, _c] += 1
    _max_cov = max(1, int(_cov_full.max()))
    cell_nig_steps = np.full((T, G, G, 3), np.nan)
    cell_hop_steps = np.full((T, G, G), np.nan)
    cell_beliefs_steps = np.full((T, G, G, _max_cov, 4), np.nan, dtype=np.float32)
    cell_ncov_steps = np.zeros((T, G, G), dtype=np.int16)
    cell_fused_steps = np.full((T, G, G, 4), np.nan, dtype=np.float32)

    pre_failure_graph_stats = mesh.overlap_graph_stats()
    post_failure_graph_stats = pre_failure_graph_stats

    for step in range(T):
        positions = [paths[w][step] for w in range(N_WEARABLES)]
        if step == FAILURE_STEP and failure_ids:
            pre_failure_graph_stats = mesh.overlap_graph_stats()
            mesh.fail_nodes(failure_ids)
            post_failure_graph_stats = mesh.overlap_graph_stats()
        mesh.run_timestep(positions, step)

        # step_best is unchanged in behaviour: same iteration order, same
        # strict >, so the same node wins every tie it won before. It now
        # carries (nu, alpha, beta, hop) alongside, which were already in
        # scope, and step_all collects every covering belief.
        step_best = {}
        step_all = defaultdict(list)
        for node in mesh.nodes.values():
            for cell, (g, n, a, b) in node.cell_beliefs.items():
                cert = 1.0 / (1.0 + min(b / (max(n, 1e-6) * max(a - 1, 1e-6)), 10.0))
                step_all[cell].append((g, n, a, b))
                if cell not in step_best or cert > step_best[cell][1]:
                    step_best[cell] = (g, cert, n, a, b, node.hop_distance)
        for cell, blist in step_all.items():
            k = min(len(blist), _max_cov)
            cell_ncov_steps[step, cell[0], cell[1]] = k
            cell_beliefs_steps[step, cell[0], cell[1], :k] = np.asarray(
                blist[:k], dtype=np.float32)
            cell_fused_steps[step, cell[0], cell[1]] = np.asarray(
                fuse_nig_product(blist), dtype=np.float32)
        for cell, (pred, cert, nu_b, al_b, be_b, hop_b) in step_best.items():
            _, truth = env.get_cell_input(cell[0], cell[1], step)
            err = circular_diff(torch.tensor(pred), truth).item() ** 2
            cell_mse_steps[step, cell[0], cell[1]] = err
            cell_cert_steps[step, cell[0], cell[1]] = cert
            cell_nig_steps[step, cell[0], cell[1]] = (nu_b, al_b, be_b)
            if hop_b is not None:
                cell_hop_steps[step, cell[0], cell[1]] = float(hop_b)

        if step % 30 == 0 or step == FAILURE_STEP:
            with np.errstate(invalid="ignore"):
                mm = np.nanmean(cell_mse_steps[step]); mc = np.nanmean(cell_cert_steps[step])
            print(f"[{tag}] step {step:03d}/{T} mean_mse={mm:.4f} mean_cert={mc:.4f}", flush=True)

    surviving = [i for i in mesh.nodes if i not in mesh.failed_nodes]
    coverage_count = np.zeros((G, G), dtype=int)
    for i in surviving:
        for (r, c) in mesh.nodes[i].fov_cells:
            coverage_count[r, c] += 1
    unreachable = sorted(mesh.unreachable_nodes())

    np.save(run_dir / "cell_mse_steps.npy", cell_mse_steps)
    np.save(run_dir / "cell_cert_steps.npy", cell_cert_steps)
    np.save(run_dir / "coverage_count.npy", coverage_count)
    # the averaged-readout arrays; same names runner.py uses
    np.save(run_dir / "cell_nig_steps.npy", cell_nig_steps)
    np.save(run_dir / "cell_hop_steps.npy", cell_hop_steps)
    np.save(run_dir / "cell_beliefs_steps.npy", cell_beliefs_steps)
    np.save(run_dir / "cell_ncov_steps.npy", cell_ncov_steps)
    np.save(run_dir / "cell_fused_steps.npy", cell_fused_steps)

    control_mse = control_cert = None
    if fraction > 0:
        cdir = control_dir_for(mesh_mode, seed)
        if (cdir / "cell_mse_steps.npy").exists():
            control_mse = np.load(cdir / "cell_mse_steps.npy")
            control_cert = np.load(cdir / "cell_cert_steps.npy")
        else:
            print(f"[{tag}] WARNING: control run not found at {cdir} -- run the "
                  f"--failure-mode none --fraction 0.0 condition for seed {seed} first. "
                  f"Paired-vs-control comparison will be skipped.")

    stats = compute_stats(cell_mse_steps, cell_cert_steps, coverage_count, G, control_mse, control_cert)

    manifest = dict(
        stage="stage6_node_failure", condition=tag,
        failure_mode=failure_mode, fraction=fraction, failure_step=FAILURE_STEP,
        failure_ids=failure_ids, mesh_mode=mesh_mode, seed=seed,
        git_commit=git_commit(),
        pre_failure_graph_stats=pre_failure_graph_stats,
        post_failure_graph_stats=post_failure_graph_stats,
        unreachable_node_count=len(unreachable),
        unreachable_node_ids=unreachable,
        results=dict(
            dead_zone_cell_count=stats["dead_zone_cell_count"],
            dead_zone_fraction=stats["dead_zone_fraction"],
            threshold_summary=stats["threshold_summary"],
            pre_window_cert_mse_r=float(stats["pre_r"]), pre_window_cert_mse_p=float(stats["pre_p"]),
            post_window_cert_mse_r=float(stats["post_r"]), post_window_cert_mse_p=float(stats["post_p"]),
            coverage_count_breakdown=stats["coverage_breakdown"],
            distance_from_dead_zone_breakdown=stats["distance_breakdown"],
            paired_vs_control=stats["paired_vs_control"],
        ),
    )
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    make_figure(run_dir, failure_mode, fraction, mesh_mode, seed, stats, coverage_count)

    print(f"\n### {tag} ###")
    print(f"    dead_zone: {stats['dead_zone_cell_count']} cells ({100*stats['dead_zone_fraction']:.1f}% of grid)")
    print(f"    threshold: covered_mse={stats['threshold_summary']['covered_mean_mse']} "
          f"covered_cert={stats['threshold_summary']['covered_mean_cert']}")
    if stats["paired_vs_control"] and "mean_diff" in stats["paired_vs_control"]:
        pv = stats["paired_vs_control"]
        print(f"    paired_vs_control: n={pv['n_cells_compared']} mean_diff={pv['mean_diff']:+.5f} "
              f"t={pv['t']:.3f} p={pv['p']:.4f}")
    print(f"    cert-mse r: pre={stats['pre_r']:.4f} post={stats['post_r']:.4f}")
    print(f"    graph: pre_components={pre_failure_graph_stats['n_connected_components']} "
          f"post_components={post_failure_graph_stats['n_connected_components']} "
          f"pre_degree={pre_failure_graph_stats['mean_degree']:.2f} "
          f"post_degree={post_failure_graph_stats['mean_degree']:.2f}")
    print(f"    unreachable_nodes={len(unreachable)} {unreachable}")
    print("=== DONE ===")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-mode", choices=["random", "clustered", "none"], required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mesh-mode", type=str, default="nig_product")
    parser.add_argument("--root", type=str, default=None,
                        help="output root; default (None) keeps the original path")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seeds; default (None) uses --seed alone, "
                             "reproducing the original single-seed invocation")
    args = parser.parse_args()
    if args.root:
        globals()["ROOT"] = Path(args.root)
    fm = "random" if args.failure_mode == "none" else args.failure_mode
    for _sd in ([int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]):
        run_condition(fm, args.fraction, seed=_sd, mesh_mode=args.mesh_mode)
