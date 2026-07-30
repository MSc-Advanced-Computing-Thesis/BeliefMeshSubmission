# Pin down exactly when/where the confident-wrong excursions happen (seen in
# triplet_analysis_corrected's panel 2), and render the actual input images
# the nodes were queried with at those moments for visual inspection --
# building on the saved raw_data.pkl rather than re-running the mesh.
#
# Run: python -u experiments/stage6_spatial_mesh/analyze_excursion_frames.py

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beliefmesh.data.grid_environment import GridEnvironment, apply_filter_from_env_value
from beliefmesh.metrics.circular import circular_diff

OUT = Path("runs/stage6/offset_world/dynamic_static_spatial/triplet_analysis_corrected")
ENV = Path("experiments/stage6_spatial_mesh/environment_v2")
TOP_N = 8


def main():
    with open(OUT / "raw_data.pkl", "rb") as f:
        data = pickle.load(f)
    trio = data["trio"]; cell = data["cell"]; offset_deg = data["offset_deg"]
    rotation_deg = data["rotation_deg"]; true_target_deg = data["true_target_deg"]
    history = data["history"]

    # compute |error| per node per step
    errors = {}  # node -> (steps array, abs_error array)
    for n in trio:
        steps_h = np.array([h[0] for h in history[n] if h[2] is not None])
        if len(steps_h) == 0:
            continue
        gammas = np.array([h[2] for h in history[n] if h[2] is not None]) / 180.0
        truth_n = true_target_deg[steps_h.astype(int)] / 180.0
        err = circular_diff(torch.tensor(gammas), torch.tensor(truth_n)).numpy() * 180.0
        errors[n] = (steps_h, np.abs(err))

    # pool all (node, step, error) and take the top-N worst across the whole trio
    pooled = []
    for n, (steps_h, abs_err) in errors.items():
        for s, e in zip(steps_h, abs_err):
            pooled.append((e, n, int(s)))
    pooled.sort(reverse=True)
    top = pooled[:TOP_N]
    print("Top worst-error (node, step, |error|) instances:")
    for e, n, s in top:
        rot = rotation_deg[s]
        print(f"  step={s:3d} node={n} |error|={e:6.1f} deg  rotation_this_step={rot:7.1f} deg "
              f"true_target={true_target_deg[s]:7.1f} deg")

    # render the actual query image at each flagged step for visual inspection
    dummy_grids = np.full((1, 22, 22), 0.5)
    env = GridEnvironment(22, dummy_grids, offset_field=None, rotation_seed=42)

    fig, axes = plt.subplots(2, (TOP_N + 1) // 2, figsize=(4 * ((TOP_N + 1) // 2), 8))
    axes = axes.ravel()
    for ax, (e, n, s) in zip(axes, top):
        rot = float(rotation_deg[s])
        image = TF.rotate(env.base_image, rot, fill=1.0)
        image = apply_filter_from_env_value(image, 0.5)
        img_np = image.permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(f"step {s}, node {n}\nrot={rot:.0f}° |err|={e:.0f}°", fontsize=9)
        ax.axis("off")
    for ax in axes[len(top):]:
        ax.axis("off")
    plt.suptitle(f"Actual query images at the {TOP_N} worst-error instances (cell {cell})")
    plt.tight_layout()
    out_path = OUT / "excursion_frames.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")

    # also check: do excursions cluster at specific rotation angles (e.g. near
    # the +-180 wrap boundary, a classic place for circular-diff bugs)?
    all_rot = np.array([rotation_deg[s] for _, _, s in pooled[:50]])
    print(f"\nrotation values for the 50 worst-error instances: "
          f"mean|rot|={np.abs(all_rot).mean():.1f}, near-boundary(|rot|>150) count="
          f"{(np.abs(all_rot) > 150).sum()}/50")
    all_rot_random = rotation_deg[np.random.default_rng(0).integers(0, len(rotation_deg), 50)]
    print(f"(reference, 50 RANDOM steps: mean|rot|={np.abs(all_rot_random).mean():.1f}, "
          f"near-boundary count={(np.abs(all_rot_random) > 150).sum()}/50)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
