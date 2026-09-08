"""Closed-form product-of-NIG fusion (objective: nig_product mode, 2026-08).

Rather than evaluating each contributor's Student-t marginal on a discretised
circular grid and taking the argmax of the summed log-density
(see product_of_experts.fuse), multiply the NIG densities directly. The
product of N NIG(gamma_i, nu_i, alpha_i, beta_i) densities, each raised to an
exponent-style tempering weight w_i (a power-likelihood / generalised-Bayes
weighting, the same mechanism product_of_experts.fuse already uses for its
consensus weights -- see that module's docstring), is itself an NIG with
closed-form parameters:

    n_eff      = sum_i w_i
    nu_star    = sum_i (w_i * nu_i)
    gamma_star = sum_i (w_i * nu_i * gamma_i) / nu_star
    alpha_star = sum_i (w_i * alpha_i) + 1.5 * (n_eff - 1)
    beta_star  = sum_i (w_i * beta_i)
                 + 0.5 * (sum_i (w_i * nu_i * gamma_i^2) - nu_star * gamma_star^2)

Derivation sketch: p_i(mu, sigma^2) = N(mu | gamma_i, sigma^2/nu_i) *
InvGamma(sigma^2 | alpha_i, beta_i), so
p_i(mu,sigma^2) ~ (sigma^2)^-(alpha_i+3/2) * exp(-1/(2 sigma^2) [nu_i(mu-gamma_i)^2 + 2 beta_i]).
Raising to power w_i and taking prod_i, the exponent of (sigma^2) sums to
-(sum_i w_i alpha_i + 3/2 sum_i w_i), matching alpha_star + 3/2 with n_eff =
sum_i w_i. Completing the square in mu inside the exp(...) term gives
nu_star, gamma_star and the beta_star correction above. Unweighted (all
w_i=1) is n_eff=N -- the plain, unweighted closed-form product.

Circular wraparound: gamma lives on [-1, 1) with period 2 (Spec Sec 2:
beliefmesh.metrics.circular). The algebra above assumes a flat, non-circular
mu, so each gamma_i is temporarily re-expressed as an offset from an
arbitrary reference (the first contributor's own gamma) via circular_diff
before combining, and gamma_star is wrapped back into [-1, 1) at the end.
This is exact, not an approximation: every (gamma_i - gamma_star) term in the
derivation is shift-invariant (it only ever appears as a pairwise
difference), so re-referencing both operands by the same constant changes
nothing about the result except where the final wrap needs to happen.
"""

from __future__ import annotations

NIGBelief = tuple[float, float, float, float]

_NU_FLOOR = 1e-6      # guards nu_star division-by-~0 when evidence is near-absent
_ALPHA_FLOOR = 1.0 + 1e-3   # predictive_uncertainty's beta/(nu*(alpha-1)) is undefined at alpha<=1


def _wrap(x: float) -> float:
    """Wrap a flat angle value/offset into [-1, 1) -- same convention as
    beliefmesh.metrics.circular.circular_diff(x, 0), reimplemented here in
    plain Python since this module works on the plain floats MeshNode.
    cell_beliefs already stores (no torch dependency needed for scalar
    closed-form arithmetic)."""
    return x - 2.0 * round(x / 2.0)


def fuse_nig_product(
    beliefs: list[NIGBelief],
    weights: list[float] | None = None,
) -> NIGBelief:
    """Closed-form product of N>=1 NIG beliefs.

    beliefs: list of (gamma, nu, alpha, beta) plain-float tuples.
    weights: optional per-contributor tempering exponents w_i, same length as
    beliefs. None means unweighted (all w_i = 1, n_eff = N) -- the plain
    nig_product mode. A specific per-contributor weighting (e.g.
    consensus-trust * predictive-certainty) gives nig_product_weighted.

    Returns (gamma_star, nu_star, alpha_star, beta_star). For a single
    contributor this returns that contributor's parameters unchanged; for N
    identical contributors gamma_star equals their common gamma (see
    tests/test_nig_product.py for both as explicit gates).
    """
    if not beliefs:
        raise ValueError("fuse_nig_product() needs at least 1 contributor")
    w = weights if weights is not None else [1.0] * len(beliefs)
    if len(w) != len(beliefs):
        raise ValueError(f"weights length {len(w)} != beliefs length {len(beliefs)}")

    n_eff = sum(w)
    ref = beliefs[0][0]  # arbitrary circular reference -- see module docstring

    nu_star_raw = sum(wi * b[1] for wi, b in zip(w, beliefs))
    nu_star = max(nu_star_raw, _NU_FLOOR)

    offsets = [_wrap(b[0] - ref) for b in beliefs]  # x_i = circular_diff(gamma_i, ref)
    x_star = sum(wi * b[1] * xi for wi, b, xi in zip(w, beliefs, offsets)) / nu_star
    gamma_star = _wrap(ref + x_star)

    alpha_star_raw = sum(wi * b[2] for wi, b in zip(w, beliefs)) + 1.5 * (n_eff - 1.0)
    alpha_star = max(alpha_star_raw, _ALPHA_FLOOR)

    sq_term = sum(wi * b[1] * xi * xi for wi, b, xi in zip(w, beliefs, offsets))
    beta_star = sum(wi * b[3] for wi, b in zip(w, beliefs)) + 0.5 * (sq_term - nu_star * x_star * x_star)
    beta_star = max(beta_star, 1e-6)  # InvGamma rate must stay positive

    return gamma_star, nu_star, alpha_star, beta_star
