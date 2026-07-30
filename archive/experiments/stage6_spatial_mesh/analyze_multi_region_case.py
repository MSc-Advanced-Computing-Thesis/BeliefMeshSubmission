# Step up from the single-offset minimal case (now understood: single-node
# learning works, optimizer is sound at lr=3e-4, remaining error traces to a
# known, now-excluded visual ambiguity at specific rotation angles) to a
# genuine multi-region test: two anchor nodes with DIFFERENT true offsets,
# and a third node that overlaps both but far more strongly with one than
# the other -- "close to A, far from B."
#
# Custom 3-node geometry (not the standard 36-node grid, deliberately
# controlled): A centre=(9,9), C centre=(12,9), B centre=(17,9), FOV=7.
#   A-C overlap: 28 cells (strong)
#   C-B overlap: 14 cells (weak, half of A-C)
#   A-B overlap: 0 cells (no direct connection -- everything must go via C)
# Offset field: hard split at col 13 -- cols 0-12 true=+30 deg (A's side),
# cols 13-21 true=-30 deg (B's side). Two STATIC wearables: one fixed at
# (row=9,col=7) inside A's FOV only, one fixed at (row=9,col=19) inside B's
# FOV only -- so A and B are permanent anchors every step, C is NEVER an
# anchor, purely propagation-driven from two simultaneously-conflicting
# sources.
#
# C has no coordinate awareness (same limitation flagged throughout this
# whole investigation) -- it structurally cannot represent "this cell is
# +30, that cell is -30" as two different outputs; the real question is
# what compromise it settles on, and whether that compromise is at least
# sane (e.g. weighted toward A given the 2:1 stronger connection) rather
# than nonsensical.
#
# Tracks: A's own belief (flat +30 truth), B's own belief (flat -30 truth),
# and C's belief on TWO cells -- one shared only with A (true=+30), one
# shared only with B (true=-30) -- to see whether C's fused/trained belief
# differs at all between them, and how each compares to its true value.
#
# Reuses today's validated fixes: lr=3e-4, excluded hard rotation angles
# (120-150, near +-180), normalised (rotation-subtracted) tracking.
#
# Run: python -u experiments/stage6_spatial_mesh/analyze_multi_region_case.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beliefmesh.config import load_config
from beliefmesh.data.grid_environment import GridEnvironment, apply_filter_from_env_value
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.metrics.circular import circular_diff
from beliefmesh.node.mesh import Mesh

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--lr", type=float, default=3e-4)
_args, _ = _parser.parse_known_args()

CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
OUT = Path(f"runs/stage6/offset_world/multi_region_case_lr{_args.lr:.0e}")
G = 22
T = 390
LR = _args.lr
OFFSET_A_DEG = 30.0
OFFSET_B_DEG = -30.0
COL_SPLIT = 13
NODE_A, NODE_C, NODE_B = (9, 9), (12, 9), (17, 9)  # (cx, cy)
WEARABLE_A_CELL = (9, 7)   # (row, col), inside A's FOV only
WEARABLE_B_CELL = (9, 19)  # inside B's FOV only
CELL_NEAR_A = (9, 10)      # C's cell shared with A only, true=+30
CELL_NEAR_B = (9, 14)      # C's cell shared with B only, true=-30
EXCLUDED_RANGES = [(120.0, 150.0), (165.0, 180.0), (-180.0, -165.0)]


def sample_safe_rotation(rng) -> float:
    while True:
        angle = float(rng.uniform(-180, 180))
        if not any(lo <= angle <= hi for lo, hi in EXCLUDED_RANGES):
            return angle


def main():
    cfg = load_config()
    cfg.model.lr = LR
    OUT.mkdir(parents=True, exist_ok=True)

    uniform = np.full((T, G, G), 0.5)
    field = np.full((T, G, G), OFFSET_B_DEG)
    field[:, :, :COL_SPLIT] = OFFSET_A_DEG
    centres = np.array([NODE_A, NODE_C, NODE_B])

    env = GridEnvironment(G, uniform, offset_field=field, rotation_seed=42)
    rng = np.random.default_rng(42)
    env.cell_rotations = np.array(
        [sample_safe_rotation(rng) for _ in range(env.cell_rotations.size)]
    ).reshape(env.cell_rotations.shape)

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

    node_a_id, node_c_id, node_b_id = 0, 1, 2
    assert WEARABLE_A_CELL in mesh.nodes[node_a_id].fov_set
    assert WEARABLE_A_CELL not in mesh.nodes[node_c_id].fov_set
    assert WEARABLE_B_CELL in mesh.nodes[node_b_id].fov_set
    assert WEARABLE_B_CELL not in mesh.nodes[node_c_id].fov_set
    assert CELL_NEAR_A in mesh.nodes[node_c_id].fov_set
    assert CELL_NEAR_B in mesh.nodes[node_c_id].fov_set
    print(f"verified geometry: A-C overlap={len(mesh.shared_cells.get((0,1),[]))}, "
          f"C-B overlap={len(mesh.shared_cells.get((1,2),[]))}, "
          f"A-B overlap={len(mesh.shared_cells.get((0,2),[]))}")

    wearable_a_pos = np.array(WEARABLE_A_CELL, dtype=float)
    wearable_b_pos = np.array(WEARABLE_B_CELL, dtype=float)

    hist_a, hist_b = [], []
    hist_c_near_a, hist_c_near_b = [], []

    for step in range(T):
        mesh.run_timestep([wearable_a_pos, wearable_b_pos], step,
                          n_wearable_samples=1, n_train_repeats=1)
        node_a, node_c, node_b = mesh.nodes[node_a_id], mesh.nodes[node_c_id], mesh.nodes[node_b_id]
        if WEARABLE_A_CELL in node_a.cell_beliefs:
            g, nu, al, be = node_a.cell_beliefs[WEARABLE_A_CELL]
            hist_a.append((step, g * 180.0, nu, al, be))
        if WEARABLE_B_CELL in node_b.cell_beliefs:
            g, nu, al, be = node_b.cell_beliefs[WEARABLE_B_CELL]
            hist_b.append((step, g * 180.0, nu, al, be))
        if CELL_NEAR_A in node_c.cell_beliefs:
            g, nu, al, be = node_c.cell_beliefs[CELL_NEAR_A]
            hist_c_near_a.append((step, g * 180.0, nu, al, be))
        if CELL_NEAR_B in node_c.cell_beliefs:
            g, nu, al, be = node_c.cell_beliefs[CELL_NEAR_B]
            hist_c_near_b.append((step, g * 180.0, nu, al, be))
        if step % 20 == 0:
            print(f"step {step}/{T}")

    print(f"samples: A={len(hist_a)} B={len(hist_b)} "
          f"C_near_A={len(hist_c_near_a)} C_near_B={len(hist_c_near_b)}")

    import pickle
    with open(OUT / "raw_data.pkl", "wb") as f:
        pickle.dump({"hist_a": hist_a, "hist_b": hist_b,
                    "hist_c_near_a": hist_c_near_a, "hist_c_near_b": hist_c_near_b}, f)

    # ── plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.axhline(OFFSET_A_DEG, color="#e41a1c", linestyle="--", linewidth=1, label="true A offset (+30)")
    ax.axhline(OFFSET_B_DEG, color="#377eb8", linestyle="--", linewidth=1, label="true B offset (-30)")
    for h, color, label in [(hist_a, "#e41a1c", "node A (anchor)"),
                            (hist_b, "#377eb8", "node B (anchor)")]:
        if not h:
            continue
        steps_h = np.array([x[0] for x in h])
        gammas = np.array([x[1] for x in h])
        rot = env.cell_rotations[steps_h, WEARABLE_A_CELL[0] if label.startswith("node A")
                                  else WEARABLE_B_CELL[0],
                                  WEARABLE_A_CELL[1] if label.startswith("node A")
                                  else WEARABLE_B_CELL[1]]
        implied = circular_diff(torch.tensor(gammas / 180.0), torch.tensor(rot / 180.0)).numpy() * 180.0
        ax.plot(steps_h, implied, color=color, label=label, linewidth=1.0)
    ax.set_ylabel("implied offset (deg)")
    ax.set_title("Anchors A and B: own predictions (normalised), should sit at their own flat truth lines")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.axhline(OFFSET_A_DEG, color="#e41a1c", linestyle="--", linewidth=1, label="true A offset (+30)")
    ax.axhline(OFFSET_B_DEG, color="#377eb8", linestyle="--", linewidth=1, label="true B offset (-30)")
    for h, cell, color, label in [(hist_c_near_a, CELL_NEAR_A, "#ff7f00", "C @ cell-near-A"),
                                  (hist_c_near_b, CELL_NEAR_B, "#984ea3", "C @ cell-near-B")]:
        if not h:
            continue
        steps_h = np.array([x[0] for x in h])
        gammas = np.array([x[1] for x in h])
        rot = env.cell_rotations[steps_h, cell[0], cell[1]]
        implied = circular_diff(torch.tensor(gammas / 180.0), torch.tensor(rot / 180.0)).numpy() * 180.0
        ax.plot(steps_h, implied, color=color, label=label, linewidth=1.0)
    ax.set_xlabel("step"); ax.set_ylabel("implied offset (deg)")
    ax.set_title("Node C (never an anchor): does it differentiate its two cells at all, "
                "and where does it compromise?")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT / "multi_region_timeseries.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")

    for h, cell, cell_label, truth in [(hist_a, WEARABLE_A_CELL, "A", OFFSET_A_DEG),
                                        (hist_b, WEARABLE_B_CELL, "B", OFFSET_B_DEG),
                                        (hist_c_near_a, CELL_NEAR_A, "C-near-A", OFFSET_A_DEG),
                                        (hist_c_near_b, CELL_NEAR_B, "C-near-B", OFFSET_B_DEG)]:
        if len(h) < 20:
            continue
        last50 = h[-50:]
        gammas = np.array([x[1] for x in last50]) / 180.0
        steps_h = np.array([x[0] for x in last50])
        rot = env.cell_rotations[steps_h, cell[0], cell[1]] / 180.0
        implied = circular_diff(torch.tensor(gammas), torch.tensor(rot)).numpy() * 180.0
        print(f"{cell_label}: last-50-step implied offset mean={implied.mean():.1f} "
              f"std={implied.std():.1f} (true={truth})")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
