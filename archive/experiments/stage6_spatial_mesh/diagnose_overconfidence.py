# Diagnostic: is the fusion-vs-naive tie on dynamic_v2 explained by uniform
# node overconfidence?
#
# Hypothesis (Spec-adjacent; raised during 6D analysis): if all contributors'
# Student-t densities are similarly and extremely sharp, the product's mode
# sits at ~ the precision-weighted mean ~ the plain mean, so fusion's and
# naive's pseudo-labels nearly coincide BY CONSTRUCTION and the arms cannot
# separate. Measured here directly: run the fusion mesh on dynamic_v2 and log,
# per fusion call, (a) contributors' Student-t scales, (b) the max/min scale
# ratio (differentiation), (c) |fused mode - naive mean| in degrees.
#
# Run from the project root: python -u experiments/stage6_spatial_mesh/diagnose_overconfidence.py

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beliefmesh.config import load_config
from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.metrics.circular import circular_diff
from beliefmesh.node.mesh import Mesh

ENV_DIR = Path("experiments/stage6_spatial_mesh/environment_v2")
BASELINE = Path("runs/stage0/baseline/checkpoints/pretrained_digit7.pth")
N_STEPS = 150
OUT = Path("runs/stage6/diagnostics")


class InstrumentedMesh(Mesh):
    """Fusion mesh that records contributor sharpness and label divergence."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log: list[dict] = []
        self._step = 0

    def _aggregate(self, contributions):
        label, agreement, inherited = super()._aggregate(contributions)
        if len(contributions) >= 2:
            beliefs = [b for _, b in contributions]
            gammas = np.array([b[0] for b in beliefs])
            scales = np.array([
                float(np.sqrt(b[3] * (1 + b[1]) / max(b[1] * b[2], 1e-6)))
                for b in beliefs])
            naive_mean = float(gammas.mean())
            div_deg = abs(circular_diff(torch.tensor(label),
                                        torch.tensor(naive_mean)).item()) * 180.0
            self.log.append({
                "step": self._step, "n": len(beliefs),
                "mean_scale_deg": float(scales.mean() * 180.0),
                "scale_ratio": float(scales.max() / max(scales.min(), 1e-9)),
                "fused_vs_naive_deg": div_deg,
            })
        return label, agreement, inherited


def main():
    cfg = load_config()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grids = np.load(ENV_DIR / "environment_grids.npy")
    path = np.load(ENV_DIR / "wearable_path.npy")
    mesh = InstrumentedMesh(
        np.load(ENV_DIR / "node_centres.npy"), fov_size=7, grid_size=grids.shape[1],
        environment=GridEnvironment(grids.shape[1], grids),
        pretrained_path=BASELINE, lr=cfg.model.lr,
        fusion_grid=circular_grid(cfg.fusion.grid_size), mode="fusion", device=device)

    for step in range(N_STEPS):
        mesh._step = step
        mesh.run_timestep([path[step]], step, n_wearable_samples=1, n_train_repeats=10)
        if step % 25 == 0:
            recent = [e for e in mesh.log if e["step"] >= step - 25]
            if recent:
                print(f"step {step:03d} | mean t-scale "
                      f"{np.mean([e['mean_scale_deg'] for e in recent]):6.2f} deg | "
                      f"scale ratio {np.mean([e['scale_ratio'] for e in recent]):5.2f} | "
                      f"|fused-naive| {np.mean([e['fused_vs_naive_deg'] for e in recent]):5.2f} deg")

    OUT.mkdir(parents=True, exist_ok=True)
    early = [e for e in mesh.log if e["step"] < 30]
    late = [e for e in mesh.log if e["step"] >= N_STEPS - 30]
    summary = {
        "n_steps": N_STEPS,
        "early(<30)": {
            "mean_t_scale_deg": float(np.mean([e["mean_scale_deg"] for e in early])),
            "mean_scale_ratio": float(np.mean([e["scale_ratio"] for e in early])),
            "mean_fused_vs_naive_deg": float(np.mean([e["fused_vs_naive_deg"] for e in early])),
        },
        "late(last30)": {
            "mean_t_scale_deg": float(np.mean([e["mean_scale_deg"] for e in late])),
            "mean_scale_ratio": float(np.mean([e["scale_ratio"] for e in late])),
            "mean_fused_vs_naive_deg": float(np.mean([e["fused_vs_naive_deg"] for e in late])),
            "p90_fused_vs_naive_deg": float(np.percentile(
                [e["fused_vs_naive_deg"] for e in late], 90)),
        },
    }
    with open(OUT / "overconfidence_diagnostic.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
