# Thesis figure: three qualitative fusion behaviours (agreement, asymmetric
# confidence, disagreement) under BOTH fusion rules, generated entirely from
# the real code path -- fuse_nig_product() for the fused parameters,
# student_t_marginal() for every plotted density. No curve is hand-built.
#
# Product fusion  : fuse_nig_product(beliefs)            nu* = sum(nu_i)
# Average fusion  : fuse_nig_product(beliefs, w=1/N)     nu* = mean(nu_i)
#
# The point: gamma* is IDENTICAL under both -- a common scaling of the
# weights cancels between numerator and denominator -- while the fused
# interval is wider under averaging, because the evidence is the mean
# rather than the sum. The identity is asserted below, not assumed.
# Run:
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

# PNG only, per the chapter standard.
OUT = Path("figures/fusion_rules_illustration.png")

CONTRIB_COLOR = "#7F77DD"
FUSED_COLOR = "#0F6E56"     # product fusion
AVG_COLOR = "#E6821B"       # average fusion -- distinct hue, not a
                            # dashed variant: the two curves nearly
                            # coincide, so line style alone did not
                            # separate them at a glance
MODE_COLOR = "#888780"

from scipy.stats import t as _student_t


def half_width(nu, alpha, beta, q=0.95):
    """90% central half-width of the fused Student-t predictive."""
    scale = (beta * (1.0 + nu) / (nu * alpha)) ** 0.5
    return float(_student_t.ppf(q, df=2.0 * alpha) * scale)

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


def main(contrib_color=None, show_inset=True, out=None):
    global CONTRIB_COLOR, OUT
    if contrib_color:
        CONTRIB_COLOR = contrib_color
    if out:
        OUT = Path(out)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharex=True)
    y_maxes = []

    contrib_line = fused_line = avg_line = mode_line = None
    max_gamma_gap = 0.0
    hw_rows = []
    for panel_i, (ax, (title, contributors)) in enumerate(
            zip(axes, PANELS.items())):
        bl = list(contributors)
        fused = fuse_nig_product(bl)
        gamma_star, nu_star, alpha_star, beta_star = fused
        w = [1.0 / len(bl)] * len(bl)
        gamma_avg, nu_avg, alpha_avg, beta_avg = fuse_nig_product(bl, weights=w)
        max_gamma_gap = max(max_gamma_gap, abs(gamma_star - gamma_avg))
        hw_p = half_width(nu_star, alpha_star, beta_star)
        hw_a = half_width(nu_avg, alpha_avg, beta_avg)
        hw_rows.append((title, gamma_star, gamma_avg, hw_p, hw_a))

        print(f"\n=== {title} ===")
        for i, (g, n, a, b) in enumerate(contributors, 1):
            print(f"  contributor {i}: gamma={g:.4f} nu={n:.4f} alpha={a:.4f} beta={b:.4f}")
        print(f"  product fusion: gamma*={gamma_star:.6f} nu*={nu_star:.4f} "
              f"alpha*={alpha_star:.4f} beta*={beta_star:.6f}")
        print(f"  average fusion: gamma*={gamma_avg:.6f} nu*={nu_avg:.4f} "
              f"alpha*={alpha_avg:.4f} beta*={beta_avg:.6f}")

        # Agreement: the three curves nearly coincide, so the product curve
        # is put BEHIND the contributors there and the average in front,
        # otherwise the thick product band hides the contributors entirely.
        z_product = 1 if panel_i == 0 else 3
        panel_y_max = 0.0
        for g, n, a, b in contributors:
            y = density(g, n, a, b)
            panel_y_max = max(panel_y_max, float(y.max()))
            line, = ax.plot(X.numpy(), y.numpy(), color=CONTRIB_COLOR,
                            linewidth=1.1, alpha=0.75, zorder=2)
            if contrib_line is None:
                contrib_line = line

        y_fused = density(gamma_star, nu_star, alpha_star, beta_star)
        panel_y_max = max(panel_y_max, float(y_fused.max()))
        line, = ax.plot(X.numpy(), y_fused.numpy(), color=FUSED_COLOR,
                        linewidth=2.8, zorder=z_product)
        if fused_line is None:
            fused_line = line

        y_avg = density(gamma_avg, nu_avg, alpha_avg, beta_avg)
        panel_y_max = max(panel_y_max, float(y_avg.max()))
        line, = ax.plot(X.numpy(), y_avg.numpy(), color=AVG_COLOR,
                        linewidth=1.5, zorder=4)
        if avg_line is None:
            avg_line = line

        vline = ax.axvline(gamma_star, color=MODE_COLOR, linestyle="--", linewidth=0.9)
        if mode_line is None:
            mode_line = vline

        # Agreement panel: the three curves are separated by ~0.3 in peak
        # density on a 0-5 axis, so the whole-panel view cannot show which is
        # which. An inset on the peak carries the actual comparison.
        if panel_i == 0 and show_inset:
            # upper-right corner, clear of the curves: above half-height the
            # Agreement peak occupies only x in [-0.09, 0.09], i.e. axes
            # fraction 0.455-0.545, so an inset starting at 0.62 cannot
            # overlap it
            axins = ax.inset_axes([0.62, 0.52, 0.36, 0.44])
            for g, n, a, b in contributors:
                axins.plot(X.numpy(), density(g, n, a, b).numpy(),
                           color=CONTRIB_COLOR, linewidth=1.1, alpha=0.75, zorder=2)
            axins.plot(X.numpy(), y_fused.numpy(), color=FUSED_COLOR,
                       linewidth=2.8, zorder=1)
            axins.plot(X.numpy(), y_avg.numpy(), color=AVG_COLOR,
                       linewidth=1.5, zorder=4)
            axins.axvline(gamma_star, color=MODE_COLOR, linestyle="--",
                          linewidth=0.9, zorder=0)
            # bound the inset by ALL three peaks, so the average curve is not
            # clipped at the frame -- it is the lowest of the three
            peaks = [float(y_fused.max()), float(y_avg.max())] + [
                float(density(g, n, a, b).max()) for g, n, a, b in contributors]
            axins.set_xlim(-0.16, 0.16)
            axins.set_ylim(min(peaks) * 0.965, max(peaks) * 1.02)
            axins.set_xticks([-0.1, 0.0, 0.1])
            axins.yaxis.tick_right()   # keep labels off the main curve
            axins.tick_params(labelsize=5.5, length=2, pad=1)
            for sp in axins.spines.values():
                sp.set_linewidth(0.6)
                sp.set_color("#999999")
            ax.indicate_inset_zoom(axins, edgecolor="#999999", linewidth=0.7,
                                   alpha=0.9)

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

    fig.legend([contrib_line, fused_line, avg_line, mode_line],
              ["Contributor beliefs", "Product fusion", "Average fusion",
               "Fused mode"],
              loc="lower center", ncol=4, frameon=False, fontsize=7.0,
              bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}")

    print("")
    print("GAMMA* IDENTITY CHECK across all panels: max |product - average| "
          f"= {max_gamma_gap:.3e}")
    assert max_gamma_gap < 1e-12, "gamma* is NOT identical under the two rules"
    print("  PASSED -- the common 1/N scaling cancels in gamma*")
    print("")
    print("FUSED 90% HALF-WIDTHS (normalised units; x180 for degrees)")
    print("%-24s %12s %12s %10s" % ("panel", "product", "average", "ratio"))
    for title, gp, ga, hp, ha in hw_rows:
        print("%-24s %12.5f %12.5f %9.2fx" % (title, hp, ha, ha / hp))
    for title, gp, ga, hp, ha in hw_rows:
        print("  %-22s product %.2f deg, average %.2f deg"
              % (title, hp * 180.0, ha * 180.0))


if __name__ == "__main__":
    main()
