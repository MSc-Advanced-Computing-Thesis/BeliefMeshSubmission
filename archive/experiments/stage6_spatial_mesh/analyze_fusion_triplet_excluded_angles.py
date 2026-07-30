# Deep-dive diagnostic: the standard pipeline only saves aggregated per-cell
# maps (cell_mse_steps.npy / cell_cert_steps.npy), never individual nodes'
# raw beliefs or what specifically got fused at a given cell/step. This
# reproduces the exact static_spatial_fusion run (same seed, field, config)
# but drives the mesh loop manually so we can record, for a chosen trio of
# mutually-overlapping nodes:
#   - each node's own predicted belief (gamma, nu, alpha, beta) for a shared
#     cell, every step it has one
#   - its hop distance each step (anchor vs propagated, and how far)
#   - every _aggregate() call at that shared cell: which nodes contributed,
#     their raw beliefs, and what the fused output was
#
# This is a read-only analysis harness -- it calls the SAME MeshNode.train_on
# / predict_cells / Mesh._aggregate methods the real run uses, just with the
# outer loop unrolled so we can hook in between steps. Not a new experiment,
# a magnifying glass on one that already ran.
#
# Run: python -u experiments/stage6_spatial_mesh/analyze_fusion_triplet.py

from __future__ import annotations

import inspect
import random
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_spatial_mesh.run_offset_experiments import build_random_wander_path

from beliefmesh.config import load_config
from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.metrics.circular import circular_diff
from beliefmesh.models.evidential import student_t_marginal
from beliefmesh.node.mesh import Mesh

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
FIELD = Path("runs/stage6/offset_world/dynamic_static_spatial/field.npy")
OUT = Path("runs/stage6/offset_world/dynamic_static_spatial/triplet_analysis_excluded_angles")
N_WEARABLES = 3
LR = 3e-4
# Excursion-frame analysis found the worst errors concentrated at two
# rotation ranges for this specific base digit: 120-150 deg (this "7" tilted
# into a visually ambiguous near-horizontal shape) and near the +-180 deg
# wrap boundary. Excluding both via rejection sampling on every rotation
# draw (training AND evaluation) to test whether removing them cleans up
# the excursion pattern.
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]


def sample_safe_rotation(rng) -> float:
    while True:
        angle = float(rng.uniform(-180, 180))
        if not any(lo <= angle <= hi for lo, hi in EXCLUDED_RANGES):
            return angle
# CORRECTED (previous version's bug): the true target at a cell is
# (this step's randomly-sampled rotation + the offset field value), NOT the
# offset alone -- GridEnvironment.cell_rotations is an independent random
# draw for every (step, row, col), so plotting gamma against a flat
# offset-only line (as the original version of this script did) makes a
# genuinely varying, correctly-tracking prediction look like chaotic drift.
# This version computes the real per-step target and tracks circular ERROR,
# which is what actually matters.


def main():
    cfg = load_config()
    cfg.model.lr = LR
    OUT.mkdir(parents=True, exist_ok=True)

    grids = np.load(ENV / "environment_grids.npy")
    T, G = grids.shape[0], grids.shape[1]
    uniform = np.full((T, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.load(FIELD)

    env = GridEnvironment(G, uniform, offset_field=field, rotation_seed=42)

    # regenerate cell_rotations (evaluation) avoiding the excluded ranges
    rng = np.random.default_rng(42)
    env.cell_rotations = np.array(
        [sample_safe_rotation(rng) for _ in range(env.cell_rotations.size)]
    ).reshape(env.cell_rotations.shape)

    # patch get_multiple_rotations (training) to avoid them too
    import torchvision.transforms.functional as TF
    from beliefmesh.data.grid_environment import apply_filter_from_env_value

    def safe_get_multiple_rotations(cell, step, n, rng):
        row, col = cell
        env_value = env.environment_grids[step, row, col]
        images, targets = [], []
        for _ in range(n):
            angle = sample_safe_rotation(rng)
            image = TF.rotate(env.base_image, angle, fill=1.0)
            images.append(apply_filter_from_env_value(image, env_value))
            targets.append(env._label(angle, row, col, step))
        return images, targets

    env.get_multiple_rotations = safe_get_multiple_rotations

    mesh = Mesh(centres, fov_size=7, grid_size=G, environment=env,
               pretrained_path=CKPT, lr=cfg.model.lr,
               fusion_grid=circular_grid(cfg.fusion.grid_size),
               mode="fusion", sample_seed=42, lam=0.1)

    # pick 3 mutually-reachable nodes near the grid centre
    centre_id = min(mesh.nodes, key=lambda i: np.hypot(*(mesh.nodes[i].centre[0] - G/2,
                                                          mesh.nodes[i].centre[1] - G/2)))
    a = centre_id
    b = mesh.overlap_graph[a][0]
    c_candidates = [n for n in mesh.overlap_graph[b] if n != a]
    c = c_candidates[0] if c_candidates else mesh.overlap_graph[a][1]
    trio = [a, b, c]
    print(f"trio node ids: {trio}, centres: {[mesh.nodes[i].centre for i in trio]}")

    # a cell shared by at least two of the trio
    shared_candidates = (mesh.shared_cells.get((a, b), []) or
                         mesh.shared_cells.get((b, c), []) or
                         mesh.shared_cells.get((a, c), []))
    assert shared_candidates, "no shared cell found between the chosen trio"
    cell = shared_candidates[len(shared_candidates) // 2]
    print(f"tracked cell: {cell}")
    offset_deg = float(env.offset_field[0, cell[0], cell[1]]) if env.offset_field is not None else None
    print(f"offset component at this cell (static): {offset_deg:.1f} deg "
          f"(the TRUE per-step target also includes a random rotation on top of this)")

    paths = [build_random_wander_path(T, G, seed=200 + i) for i in range(N_WEARABLES)]

    # true target for this cell at each step: rotation_this_step + offset,
    # matching exactly what _label() computes (wrapped, normalised -> deg)
    rotation_deg = env.cell_rotations[:, cell[0], cell[1]].copy()
    true_target_deg = np.array([
        (env._label(float(rotation_deg[step]), cell[0], cell[1], step).item()) * 180.0
        for step in range(T)
    ])

    def implied_offset_deg(gamma_deg: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """gamma minus the known random rotation for those steps, wrapped --
        isolates just the offset component the model implies it learned,
        directly comparable to a FLAT true-offset line (Christian's fix:
        cleaner to read than a moving target)."""
        g_norm = torch.tensor(gamma_deg / 180.0)
        rot_norm = torch.tensor(rotation_deg[steps.astype(int)] / 180.0)
        return circular_diff(g_norm, rot_norm).numpy() * 180.0

    # history[node_id] -> list of (step, hop_distance, gamma_deg, nu, alpha, beta) or None
    history = {n: [] for n in trio}
    fusion_events = []  # (step, contributor_ids, contributor_beliefs_deg, fused_deg)

    orig_aggregate = mesh._aggregate

    def traced_aggregate(contributions):
        result = orig_aggregate(contributions)
        # _aggregate doesn't receive the cell being aggregated -- the caller
        # (run_timestep's BFS loop) has it in a local named `cell`, one frame
        # up. Grabbing it via the frame rather than editing mesh.py, since
        # this is diagnostic-only and shouldn't touch the real aggregation
        # path. Filtering to ONLY our tracked cell here (not "any cell any
        # trio node touches anywhere in the 36-node mesh", which is what
        # produced 97,495 events last run and made panel 3 unplottable).
        caller_cell = inspect.currentframe().f_back.f_locals.get("cell")
        if caller_cell == cell:
            beliefs_deg = [(cid, b_[0] * 180.0) for cid, b_ in contributions]
            fusion_events.append((current_step[0], [cid for cid, _ in contributions],
                                  beliefs_deg, result[0] * 180.0))
        return result

    mesh._aggregate = traced_aggregate
    current_step = [0]

    for step in range(T):
        current_step[0] = step
        positions = [paths[w][step] for w in range(N_WEARABLES)]
        mesh.run_timestep(positions, step, n_wearable_samples=1, n_train_repeats=1)
        for n in trio:
            node = mesh.nodes[n]
            if cell in node.cell_beliefs:
                g, nu, alpha, beta = node.cell_beliefs[cell]
                history[n].append((step, node.hop_distance, g * 180.0, nu, alpha, beta))
            else:
                history[n].append((step, node.hop_distance, None, None, None, None))
        if step % 10 == 0:
            print(f"step {step}/{T}")

    mesh._aggregate = orig_aggregate
    print(f"loop done, {len(fusion_events)} fusion events recorded, saving raw data...")

    import pickle
    with open(OUT / "raw_data.pkl", "wb") as f:
        pickle.dump({
            "trio": trio, "cell": cell, "offset_deg": offset_deg,
            "rotation_deg": rotation_deg, "true_target_deg": true_target_deg,
            "history": history, "fusion_events": fusion_events,
        }, f)
    print(f"saved {OUT / 'raw_data.pkl'}")

    # ── plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True)
    colors = {trio[0]: "#e41a1c", trio[1]: "#377eb8", trio[2]: "#4daf4a"}

    ax = axes[0]
    ax.axhline(offset_deg, color="black", linestyle="--", linewidth=1,
              label=f"true offset ({offset_deg:.1f} deg)", zorder=1)
    for n in trio:
        steps_h = np.array([h[0] for h in history[n] if h[2] is not None])
        gammas = np.array([h[2] for h in history[n] if h[2] is not None])
        stds = np.array([np.sqrt(h[5] * (1 + h[3]) / (h[3] * h[4])) * 180.0
                         for h in history[n] if h[2] is not None])
        implied = implied_offset_deg(gammas, steps_h)
        ax.plot(steps_h, implied, color=colors[n], label=f"node {n} implied offset (γ - rotation)",
               linewidth=1.2)
        ax.fill_between(steps_h, implied - stds, implied + stds, color=colors[n], alpha=0.15)
    ax.set_ylabel("implied offset (deg)")
    ax.set_title(f"Node predictions, NORMALISED (γ minus the known random rotation) -- "
                 f"isolates just the offset component, flat truth line")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    print("  panel 1 (predictions, normalised) done")

    ax = axes[1]
    for n in trio:
        steps_h = np.array([h[0] for h in history[n] if h[2] is not None])
        gammas = np.array([h[2] for h in history[n] if h[2] is not None]) / 180.0
        truth_n = true_target_deg[steps_h.astype(int)] / 180.0
        err = circular_diff(torch.tensor(gammas), torch.tensor(truth_n)).numpy() * 180.0
        ax.plot(steps_h, np.abs(err), color=colors[n], label=f"node {n} |error|", linewidth=1.0)
    ax.set_ylabel("|circular error| (deg)")
    ax.set_title("Actual error (prediction vs true rotation+offset target) -- what matters")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    print("  panel 2 (error) done")

    ax = axes[2]
    for n in trio:
        steps_h = [h[0] for h in history[n]]
        hops = [h[1] if h[1] is not None else np.nan for h in history[n]]
        ax.plot(steps_h, hops, color=colors[n], label=f"node {n} hop", linewidth=1.0,
                drawstyle="steps-post")
    ax.set_ylabel("hop distance"); ax.set_title("Hop distance over time (0 = anchor this step)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    print("  panel 3 (hop distance) done")

    ax = axes[3]
    ax.axhline(offset_deg, color="black", linestyle="--", linewidth=1,
              label=f"true offset ({offset_deg:.1f} deg)", zorder=1)
    fe_steps = np.array([e[0] for e in fusion_events])
    fe_fused = implied_offset_deg(np.array([e[3] for e in fusion_events]), fe_steps)
    ax.scatter(fe_steps, fe_fused, color="purple", s=12, label="fused output (implied offset)", zorder=3)
    # vectorised, one scatter() call per contributor colour instead of one
    # call per point -- a Python-loop-of-individual-scatter-calls is what
    # made the previous run (97,495 unfiltered events) effectively hang
    for cid in trio:
        pts = [(e[0], val) for e in fusion_events for c2, val in e[2] if c2 == cid]
        if pts:
            xs_, ys_ = zip(*pts)
            xs_ = np.array(xs_)
            ys_implied = implied_offset_deg(np.array(ys_), xs_)
            ax.scatter(xs_, ys_implied, color=colors[cid], s=6, alpha=0.5, zorder=2)
    ax.set_xlabel("step"); ax.set_ylabel("implied offset (deg)")
    ax.set_title(f"Fusion events at cell {cell}, normalised: contributor beliefs (small dots) -> "
                 f"fused output (purple), n={len(fusion_events)}")
    ax.legend(); ax.grid(True, alpha=0.3)
    print("  panel 3 (fusion events) done, calling tight_layout + savefig...")

    plt.tight_layout()
    print("  tight_layout done, saving...")
    out_path = OUT / "triplet_timeseries.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}, building figure 2 (distribution snapshots)...")

    # ── distribution snapshots at a few interesting steps (normalised: shift
    # each node's Student-t location by -rotation_this_step, so all snapshots
    # are directly comparable against one flat true-offset line, same fix as
    # the timeseries panels above) ───────────────────────────────────────────
    snapshot_steps = [T // 8, T // 2, (7 * T) // 8]
    fig2, axes2 = plt.subplots(1, len(snapshot_steps), figsize=(6 * len(snapshot_steps), 4.5))
    xs = torch.linspace(-1.5, 1.5, 400)
    for ax, s in zip(axes2, snapshot_steps):
        print(f"  building snapshot panel for step {s}...")
        ax.axvline(offset_deg, color="black", linestyle="--", linewidth=1, label="true offset")
        rot_norm = rotation_deg[s] / 180.0
        for n in trio:
            h = history[n][s]
            if h[2] is None:
                continue
            g_norm, nu, alpha, beta = h[2] / 180.0, h[3], h[4], h[5]
            implied_loc = circular_diff(torch.tensor(g_norm), torch.tensor(rot_norm))
            dist = student_t_marginal(implied_loc, torch.tensor(nu),
                                      torch.tensor(alpha), torch.tensor(beta))
            dens = dist.log_prob(xs).exp()
            ax.plot(xs.numpy() * 180.0, dens.numpy(), color=colors[n], label=f"node {n}")
        ax.set_title(f"step {s}")
        ax.set_xlabel("implied offset (deg)")
        ax.legend(fontsize=8)
    print("  all snapshot panels done, saving figure 2...")
    fig2.suptitle(f"Predictive Student-t densities at shared cell {cell}, NORMALISED "
                  f"(rotation subtracted), 3 snapshots")
    plt.tight_layout()
    out_path2 = OUT / "triplet_distributions.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out_path2}")

    print(f"\n{len(fusion_events)} fusion events recorded at this cell across the run")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
