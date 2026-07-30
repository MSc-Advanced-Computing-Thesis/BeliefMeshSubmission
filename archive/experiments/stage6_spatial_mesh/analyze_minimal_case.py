# Back to first principles: today's investigation found fusion performing
# worse than a trivial constant guess on a moderately complex static field,
# and a proposed fix (self-consistency) made things 5x worse despite looking
# better on one hand-picked example. Before touching anything else, isolate
# the absolute simplest possible case and verify the basics actually work:
#
#   - environment: a single UNIFORM +20 deg offset everywhere (no spatial
#     variation at all -- the trivial constant-target task)
#   - a single STATIC wearable (fixed position for all 390 steps, never
#     moves) placed so exactly TWO nodes (2, 3) become anchors, permanently,
#     every step -- every other node relies ENTIRELY on propagation from
#     these same two sources, repeatedly, with a source signal that never
#     changes (removes staleness/tracking confounds entirely, isolates pure
#     propagation fidelity)
#   - node 1 is a third node that shares a cell with BOTH anchors (triple
#     overlap) but is NEVER itself an anchor -- its belief there comes
#     entirely from fusion, nothing else
#
# Two panels:
#   1. Both anchors' (2, 3) own predictive distributions on a cell they
#      directly share with each other -- sanity check on BASIC LEARNING:
#      do two independently-trained models, given an identical trivial
#      constant target, converge to matching, correct, tight beliefs at all?
#   2. Node 1's (never an anchor) predictive distribution on a cell it
#      shares with both anchors -- sanity check on PROPAGATION FIDELITY:
#      does the correct, learned belief survive being fused and passed
#      one hop outward, or does it get corrupted?
#
# If panel 1 fails, the bug is in basic model training (nig_loss/optimiser/
# lr), nothing to do with fusion at all. If panel 1 succeeds but panel 2
# fails, the bug is specifically in aggregation/propagation.
#
# Uses the current (reverted, self-consistency-free) fusion mechanism plus
# the validated nu-clamp bug fix. lr=3e-4, n_train_repeats=1 (today's
# established-correct settings).
#
# Run: python -u experiments/stage6_spatial_mesh/analyze_minimal_case.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beliefmesh.config import load_config
from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.models.evidential import student_t_marginal
from beliefmesh.node.mesh import Mesh

ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
OUT = Path("runs/stage6/offset_world/minimal_case_fixed_rotation")
LR = 3e-4
UNIFORM_OFFSET_DEG = 20.0
FIXED_ROTATION_DEG = 0.0  # hold the digit's own rotation constant too -- see
# note below: get_multiple_rotations draws angle=rng.uniform(-180,180) FRESH
# on every hop-0 training call, so the real regression target is
# (random_rotation + offset), not a constant offset alone. The first version
# of this diagnostic compared gamma against a flat +20 line while the actual
# target was varying across the whole circle every sample -- not a fair test.
# Monkeypatching the rotation to a fixed value below makes the target
# genuinely constant, isolating whether repeated single-sample training can
# converge on ANY fixed target at all.
T_STEPS = 390

ANCHOR_A, ANCHOR_B, DOWNSTREAM = 2, 3, 1
WEARABLE_CELL = (2, 11)   # shared by nodes 2 & 3 only
DOWNSTREAM_CELL = (3, 9)  # shared by nodes 1, 2 & 3 (triple overlap)


def main():
    cfg = load_config()
    cfg.model.lr = LR
    OUT.mkdir(parents=True, exist_ok=True)

    G = 22
    uniform_grid = np.full((T_STEPS, G, G), 0.5)
    centres = np.load(ENV / "node_centres.npy")
    field = np.full((T_STEPS, G, G), UNIFORM_OFFSET_DEG)

    env = GridEnvironment(G, uniform_grid, offset_field=field, rotation_seed=42)

    import torchvision.transforms.functional as TF
    from beliefmesh.data.grid_environment import apply_filter_from_env_value

    def fixed_get_multiple_rotations(cell, step, n, rng):
        row, col = cell
        env_value = env.environment_grids[step, row, col]
        images, targets = [], []
        for _ in range(n):
            image = TF.rotate(env.base_image, FIXED_ROTATION_DEG, fill=1.0)
            images.append(apply_filter_from_env_value(image, env_value))
            targets.append(env._label(FIXED_ROTATION_DEG, row, col, step))
        return images, targets

    env.get_multiple_rotations = fixed_get_multiple_rotations

    mesh = Mesh(centres, fov_size=7, grid_size=G, environment=env,
               pretrained_path=CKPT, lr=cfg.model.lr,
               fusion_grid=circular_grid(cfg.fusion.grid_size),
               mode="fusion", sample_seed=42, lam=0.1)

    assert WEARABLE_CELL in mesh.nodes[ANCHOR_A].fov_set
    assert WEARABLE_CELL in mesh.nodes[ANCHOR_B].fov_set
    assert WEARABLE_CELL not in mesh.nodes[DOWNSTREAM].fov_set
    assert DOWNSTREAM_CELL in mesh.nodes[ANCHOR_A].fov_set
    assert DOWNSTREAM_CELL in mesh.nodes[ANCHOR_B].fov_set
    assert DOWNSTREAM_CELL in mesh.nodes[DOWNSTREAM].fov_set
    print(f"verified: wearable cell {WEARABLE_CELL} covers exactly nodes "
          f"{ANCHOR_A},{ANCHOR_B}; downstream cell {DOWNSTREAM_CELL} covers all three")

    wearable_pos = np.array(WEARABLE_CELL, dtype=float)

    history_anchor = {ANCHOR_A: [], ANCHOR_B: []}
    history_downstream = []

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    for step in range(T_STEPS):
        mesh.run_timestep([wearable_pos], step, n_wearable_samples=1, n_train_repeats=1)
        for n in (ANCHOR_A, ANCHOR_B):
            node = mesh.nodes[n]
            if WEARABLE_CELL in node.cell_beliefs:
                g, nu, alpha, beta = node.cell_beliefs[WEARABLE_CELL]
                history_anchor[n].append((step, g * 180.0, nu, alpha, beta))
        d_node = mesh.nodes[DOWNSTREAM]
        if DOWNSTREAM_CELL in d_node.cell_beliefs:
            g, nu, alpha, beta = d_node.cell_beliefs[DOWNSTREAM_CELL]
            history_downstream.append((step, d_node.hop_distance, g * 180.0, nu, alpha, beta))
        if step % 20 == 0:
            print(f"step {step}/{T_STEPS}")

    print(f"anchor A samples: {len(history_anchor[ANCHOR_A])}, "
          f"anchor B samples: {len(history_anchor[ANCHOR_B])}, "
          f"downstream samples: {len(history_downstream)}")

    # ── panel 1: both anchors on their shared cell ──────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    colors = {ANCHOR_A: "#e41a1c", ANCHOR_B: "#377eb8", DOWNSTREAM: "#4daf4a"}

    ax = axes[0]
    ax.axhline(UNIFORM_OFFSET_DEG, color="black", linestyle="--", linewidth=1, label="ground truth (+20)")
    for n in (ANCHOR_A, ANCHOR_B):
        h = history_anchor[n]
        steps_h = [x[0] for x in h]
        gammas = np.array([x[1] for x in h])
        stds = np.array([np.sqrt(x[4] * (1 + x[2]) / (x[2] * x[3])) * 180.0 for x in h])
        ax.plot(steps_h, gammas, color=colors[n], label=f"node {n} (anchor) γ", linewidth=1.2)
        ax.fill_between(steps_h, gammas - stds, gammas + stds, color=colors[n], alpha=0.15)
    ax.set_ylabel("predicted offset (deg)")
    ax.set_title(f"PANEL 1 -- both anchors' own predictions at their shared cell {WEARABLE_CELL} "
                 f"(trivial constant target, direct ground-truth training)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.axhline(UNIFORM_OFFSET_DEG, color="black", linestyle="--", linewidth=1, label="ground truth (+20)")
    steps_h = [x[0] for x in history_downstream]
    gammas = np.array([x[2] for x in history_downstream])
    stds = np.array([np.sqrt(x[5] * (1 + x[3]) / (x[3] * x[4])) * 180.0 for x in history_downstream])
    ax.plot(steps_h, gammas, color=colors[DOWNSTREAM], label=f"node {DOWNSTREAM} (never anchor) γ", linewidth=1.2)
    ax.fill_between(steps_h, gammas - stds, gammas + stds, color=colors[DOWNSTREAM], alpha=0.15)
    ax.set_xlabel("step"); ax.set_ylabel("predicted offset (deg)")
    ax.set_title(f"PANEL 2 -- downstream (never-anchor) node {DOWNSTREAM}'s prediction at shared cell "
                 f"{DOWNSTREAM_CELL}, via propagation only")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT / "minimal_case_timeseries.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # ── distribution snapshots ──────────────────────────────────────────────
    snapshot_idxs = [len(history_anchor[ANCHOR_A]) // 8, len(history_anchor[ANCHOR_A]) // 2,
                     (7 * len(history_anchor[ANCHOR_A])) // 8]
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 8))
    xs = torch.linspace(-1.0, 1.5, 400)
    for col, idx in enumerate(snapshot_idxs):
        ax = axes2[0, col]
        ax.axvline(UNIFORM_OFFSET_DEG, color="black", linestyle="--", linewidth=1, label="truth")
        for n in (ANCHOR_A, ANCHOR_B):
            h = history_anchor[n]
            if idx >= len(h):
                continue
            _, g, nu, alpha, beta = h[idx]
            dist = student_t_marginal(torch.tensor(g / 180.0), torch.tensor(nu),
                                      torch.tensor(alpha), torch.tensor(beta))
            dens = dist.log_prob(xs).exp()
            ax.plot(xs.numpy() * 180.0, dens.numpy(), color=colors[n], label=f"node {n}")
        ax.set_title(f"Panel 1 snapshot, step {history_anchor[ANCHOR_A][idx][0] if idx < len(history_anchor[ANCHOR_A]) else '?'}")
        ax.legend(fontsize=8)

        ax = axes2[1, col]
        ax.axvline(UNIFORM_OFFSET_DEG, color="black", linestyle="--", linewidth=1, label="truth")
        d_idx = min(idx, len(history_downstream) - 1)
        if history_downstream:
            _, _, g, nu, alpha, beta = history_downstream[d_idx]
            dist = student_t_marginal(torch.tensor(g / 180.0), torch.tensor(nu),
                                      torch.tensor(alpha), torch.tensor(beta))
            dens = dist.log_prob(xs).exp()
            ax.plot(xs.numpy() * 180.0, dens.numpy(), color=colors[DOWNSTREAM], label=f"node {DOWNSTREAM}")
        ax.set_title(f"Panel 2 snapshot, step {history_downstream[d_idx][0] if history_downstream else '?'}")
        ax.set_xlabel("offset (deg)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path2 = OUT / "minimal_case_distributions.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out_path2}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
