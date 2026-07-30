# The sharpest diagnostic yet: minimal_case_fixed_rotation showed that even
# with a 100% deterministic, bit-identical (image, target) pair repeated for
# 390 steps -- zero randomness anywhere -- a single node's predictions never
# converge, oscillating persistently between roughly -100 and +75 deg around
# a true target of +20. No mesh, no fusion, no propagation involved in that
# test at all -- just repeated single-sample SGD on one fixed example. That
# points straight at the optimizer/learning-rate, not at fusion mechanics.
#
# This isolates it further: ONE MeshNode, no Mesh/overlap graph at all,
# training repeatedly on the exact same fixed (image, target) pair, sweeping
# across several learning rates. If a smaller lr converges cleanly, that
# confirms simple overshoot (step size too large for the local curvature) --
# a straightforward, fixable optimizer bug that's been quietly corrupting
# every result collected today.
#
# Run: python -u experiments/stage6_spatial_mesh/analyze_lr_sweep_single_node.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beliefmesh.data.grid_environment import GridEnvironment, apply_filter_from_env_value
from beliefmesh.node.mesh import MeshNode

CKPT = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
OUT = Path("runs/stage6/offset_world/minimal_case_fixed_rotation")
N_STEPS = 390
TRUE_OFFSET_DEG = 20.0
FIXED_ROTATION_DEG = 0.0
LRS = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the ONE fixed (image, target) pair, reusing GridEnvironment's
    # exact image pipeline for a faithful, apples-to-apples reproduction
    dummy_grids = __import__("numpy").full((1, 22, 22), 0.5)
    env = GridEnvironment(22, dummy_grids, offset_field=None, rotation_seed=42)
    image = TF.rotate(env.base_image, FIXED_ROTATION_DEG, fill=1.0)
    image = apply_filter_from_env_value(image, 0.5)
    target_norm = ((FIXED_ROTATION_DEG / 180.0 + TRUE_OFFSET_DEG / 180.0 + 1.0) % 2.0) - 1.0
    target = torch.tensor(target_norm, dtype=torch.float32)
    print(f"fixed target: {target.item()*180:.2f} deg (should be {TRUE_OFFSET_DEG})")

    results = {}
    for lr in LRS:
        node = MeshNode(0, (11, 11), [(11, 11)], lr=lr, device=device)
        node.load_checkpoint(CKPT)
        gammas, losses = [], []
        for step in range(N_STEPS):
            loss = node.train_on([image], torch.tensor([target.item()]), lam=0.1)
            node.predict_cells([image], [(11, 11)])
            g, nu, alpha, beta = node.cell_beliefs[(11, 11)]
            gammas.append(g * 180.0)
            losses.append(loss)
        results[lr] = (gammas, losses)
        final_20 = gammas[-20:]
        print(f"lr={lr:.0e}: final-20-step gamma mean={sum(final_20)/len(final_20):.2f} "
              f"std={(sum((x-sum(final_20)/len(final_20))**2 for x in final_20)/len(final_20))**0.5:.2f} "
              f"final loss={losses[-1]:.4f}")

    fig, axes = plt.subplots(len(LRS), 2, figsize=(14, 3 * len(LRS)))
    for i, lr in enumerate(LRS):
        gammas, losses = results[lr]
        ax = axes[i, 0]
        ax.axhline(TRUE_OFFSET_DEG, color="black", linestyle="--", linewidth=1)
        ax.plot(gammas, color="#377eb8", linewidth=1)
        ax.set_ylabel(f"lr={lr:.0e}\ngamma (deg)")
        if i == 0:
            ax.set_title("predicted offset over 390 identical-example steps")
        ax = axes[i, 1]
        ax.plot(losses, color="#e41a1c", linewidth=1)
        ax.set_ylabel("nig_loss")
        if i == 0:
            ax.set_title("training loss")
    axes[-1, 0].set_xlabel("step"); axes[-1, 1].set_xlabel("step")
    plt.tight_layout()
    out_path = OUT / "lr_sweep_single_node.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
