# Thesis figure: three qualitative fusion behaviours (agreement, asymmetric
# confidence, disagreement) for nig_product's closed-form product-of-NIG
# fusion, generated entirely from the real code path -- fuse_nig_product()
# for the fused parameters, student_t_marginal() for every plotted density.
# No curve is hand-constructed. Run:
#   python -u experiments/stage6_spatial_mesh/make_fusion_illustration.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beliefmesh.fusion.nig_product import fuse_nig_product
from beliefmesh.models.evidential import student_t_marginal

OUT = Path("figures/nig_product_fusion_illustration.pdf")

CONTRIB_COLOR = "#7F77DD"
FUSED_COLOR = "#0F6E56"
MODE_COLOR = "#888780"

# (gamma, nu, alpha, beta) per contributor, chosen so each panel's qualitative
# claim is unambiguous at a glance -- see printed output below for the exact
# values and the resulting fused parameters.
PANELS = {
    "Agreement": [
        (-0.035, 10.0, 8.0, 0.05),
        (0.035, 10.0, 8.0, 0.052),
    ],
    "Asymmetric confidence": [
        (-0.40, 1.0, 1.5, 0.30),   # low nu: diffuse, low-confidence contributor
        (0.35, 10.0, 8.0, 0.05),   # high nu: sharp, high-confidence contributor
    ],
    "Disagreement": [
        (-0.40, 10.0, 8.0, 0.05),
        (0.40, 10.0, 8.0, 0.05),
    ],
}

X = torch.linspace(-1.0, 1.0, 2000)


def density(gamma, nu, alpha, beta, x=X):
    dist = student_t_marginal(torch.tensor(gamma), torch.tensor(nu),
                              torch.tensor(alpha), torch.tensor(beta))
    return dist.log_prob(x).exp()


def main():
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharex=True)
    y_maxes = []

    contrib_line = fused_line = mode_line = None
    for ax, (title, contributors) in zip(axes, PANELS.items()):
        fused = fuse_nig_product(list(contributors))
        gamma_star, nu_star, alpha_star, beta_star = fused

        print(f"\n=== {title} ===")
        for i, (g, n, a, b) in enumerate(contributors, 1):
            print(f"  contributor {i}: gamma={g:.4f} nu={n:.4f} alpha={a:.4f} beta={b:.4f}")
        print(f"  fused:         gamma*={gamma_star:.4f} nu*={nu_star:.4f} "
              f"alpha*={alpha_star:.4f} beta*={beta_star:.4f}")

        panel_y_max = 0.0
        for g, n, a, b in contributors:
            y = density(g, n, a, b)
            panel_y_max = max(panel_y_max, float(y.max()))
            line, = ax.plot(X.numpy(), y.numpy(), color=CONTRIB_COLOR,
                            linewidth=1.1, alpha=0.75)
            if contrib_line is None:
                contrib_line = line

        y_fused = density(gamma_star, nu_star, alpha_star, beta_star)
        panel_y_max = max(panel_y_max, float(y_fused.max()))
        line, = ax.plot(X.numpy(), y_fused.numpy(), color=FUSED_COLOR, linewidth=2.2)
        if fused_line is None:
            fused_line = line

        vline = ax.axvline(gamma_star, color=MODE_COLOR, linestyle="--", linewidth=0.9)
        if mode_line is None:
            mode_line = vline

        ax.set_title(title, fontsize=9)
        ax.set_xlim(-1.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("none")
        ax.tick_params(labelsize=7)
        y_maxes.append(panel_y_max)

    # Shared y-axis check: agreement's contributors/fused are far narrower
    # (much taller peak density) than disagreement's inflated, wide fused
    # curve -- forcing one shared y-limit would flatten one panel or clip the
    # other, so each panel keeps its own y-limit (10% headroom).
    spread = max(y_maxes) / min(y_maxes)
    if spread > 3.0:
        print(f"\nNOTE: panel peak densities span a {spread:.1f}x range "
              f"({min(y_maxes):.2f} to {max(y_maxes):.2f}) -- a single shared "
              "y-axis would make at least one panel illegible, so each panel "
              "uses its own y-limit (x-axis is shared).")
        for ax, y_max in zip(axes, y_maxes):
            ax.set_ylim(0, y_max * 1.1)
    else:
        shared_max = max(y_maxes) * 1.1
        for ax in axes:
            ax.set_ylim(0, shared_max)

    axes[0].set_ylabel("Density", fontsize=8)
    for ax in axes:
        ax.set_xlabel(r"$y$", fontsize=8)

    fig.legend([contrib_line, fused_line, mode_line],
              ["Contributor beliefs", "Fused belief", "Fused mode"],
              loc="lower center", ncol=3, frameon=False, fontsize=7.5,
              bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
