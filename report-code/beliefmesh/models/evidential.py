"""NIG loss and distributional utilities. Spec Sec 3.

These operate on (gamma, nu, alpha, beta) tuples regardless of which network
produced them -- EvidentialCNN's fc2 emits the raw four values directly, so
there is no separate learnable head here, just the fixed math: the loss used
to train against a scalar target, and the Student-t marginal predictive that
fusion (Spec Sec 4) consumes.

Epistemic uncertainty is 1/nu, aleatoric is beta/(alpha-1). The marginal
predictive is Student-t with location gamma, df 2*alpha, scale
sqrt(beta*(1+nu)/(nu*alpha)).
"""

from __future__ import annotations

import torch

from beliefmesh.metrics.circular import circular_diff


def nig_loss(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
    lam: float = 0.1,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evidential regression loss: Student-t NLL plus evidence regularisation.

    target is the scalar angle (normalised units, [-1, 1)) - ground truth in
    Stage 0, a fused-mode pseudo-label everywhere else (Spec Sec 5). lam scales
    the evidence regulariser, which penalises confidence (large nu, alpha) in
    proportion to the wrapped error magnitude.

    weights, if given (consensus-tempered gradient): per-sample scale on the
    loss before reduction, e.g. agreement*inherited-trust per fused cell --
    a low-trust fused label produces a smaller effective gradient step
    instead of being trained on at full strength like a fully-trusted one.
    None (every other mode, and anchor ground-truth training) reduces to the
    ordinary unweighted mean -- no behaviour change.
    """
    nu = nu.clamp(min=1e-6)
    omega = 2 * beta * (1 + nu)

    diff = circular_diff(target, gamma)

    loss_nll = 0.5 * torch.log(torch.tensor(torch.pi) / nu) - alpha * torch.log(omega) + \
               (alpha + 0.5) * torch.log(torch.square(diff) * nu + omega) + \
               torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    loss_reg = torch.abs(diff) * (2 * nu + alpha)

    per_sample = loss_nll + lam * loss_reg
    if weights is None:
        return per_sample.mean()
    weights = weights.clamp(min=1e-3)
    return (per_sample * weights).sum() / weights.sum()


def predictive_uncertainty(nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Scalar uncertainty summary Var[mu] = beta / (nu * (alpha - 1)).

    DISPLAY/EVALUATION ONLY (calibration correlation, plots, logging). Never
    feed this into fusion -- fusion consumes the full belief via
    student_t_marginal; collapsing to (prediction, uncertainty) before fusion
    is Defect 1 from the prior implementation. Formula matches the prior
    repo's predict() helper exactly, so calibration numbers stay comparable.

    Unaffected by training_uncertainty()'s `measure` ablation below -- this
    function always reports the epistemic component, regardless of which
    measure a given run used to WEIGHT training, so that cert-MSE
    correlation is measured with the same ruler across every arm of that
    ablation (see training_uncertainty's docstring)."""
    return beta / (nu.clamp(min=1e-6) * (alpha - 1).clamp(min=1e-6))


def training_uncertainty(nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                         measure: str = "epistemic") -> torch.Tensor:
    """The uncertainty measure that FEEDS the fused/contributor certainty used
    to temper the training gradient (2026-08, uncertainty-measure ablation --
    Mesh(uncertainty_measure=...), consumed only inside node.mesh._aggregate's
    nig_product family). Three measures, all derived from the same NIG
    parameters, same clamps as predictive_uncertainty:

      epistemic -- beta / (nu * (alpha - 1))            [default; identical
                   expression to predictive_uncertainty above -- verified
                   bit-for-bit equivalent in tests/test_evidential.py]
      aleatoric -- beta / (alpha - 1)                     [no nu term at all]
      total     -- beta * (1 + nu) / (nu * (alpha - 1))   [epistemic + aleatoric,
                   also the Student-t predictive VARIANCE's numerator/(nu*(a-1))
                   form -- NOT the predictive scale itself, which additionally
                   involves a sqrt (see student_t_marginal)]

    The design chapter argues epistemic is the correct choice for gating a
    training update (evidence, not total predictive spread, should decide how
    hard to trust a fused label); this function exists to make that choice an
    explicit, swappable ablation rather than a load-bearing assumption baked
    into _aggregate() itself. Caller (mesh.py) is responsible for the
    downstream .clamp(max=10.0) and 1/(1+u) mapping -- unchanged and identical
    regardless of which measure is selected here."""
    nu_c = nu.clamp(min=1e-6)
    alpha_m1 = (alpha - 1).clamp(min=1e-6)
    if measure == "epistemic":
        return beta / (nu_c * alpha_m1)
    if measure == "aleatoric":
        return beta / alpha_m1
    if measure == "total":
        return beta * (1 + nu_c) / (nu_c * alpha_m1)
    raise ValueError(f"unknown uncertainty measure {measure!r}, expected "
                      "'epistemic', 'aleatoric', or 'total'")


def student_t_marginal(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.distributions.StudentT:
    """Marginal predictive over the target: Student-t with location gamma,
    df 2*alpha, scale sqrt(beta*(1+nu)/(nu*alpha)) (Spec Sec 3).

    Note: StudentT.log_prob does NOT wrap on the circle - it uses a flat
    (y - loc). Fusion (Spec Sec 4.3) must evaluate log-densities at wrapped
    offsets, i.e. at gamma + circular_diff(y, gamma), so that beliefs at +179
    and -179 degrees produce one shared mode rather than two disjoint ones.
    """
    scale = torch.sqrt(beta * (1 + nu) / (nu * alpha))
    return torch.distributions.StudentT(df=2 * alpha, loc=gamma, scale=scale)
