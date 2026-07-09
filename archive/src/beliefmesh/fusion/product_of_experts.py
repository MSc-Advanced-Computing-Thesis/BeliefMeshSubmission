"""Product-of-experts belief fusion. Spec Sec 4.1-4.3.

Single function for all N >= 2 contributors -- there is no separate pairwise
code path. N=2 unweighted must reproduce the old bayesian_fusion_grid exactly
(Spec Sec 9 step 2, hard gate) before anything built on top of this is trusted.

total(y) = sum_i w_i * log p_i(y), evaluated on circular_grid(G), fused_mode = argmax.
Weights w_i (consensus, Spec Sec 6.2) are applied POST-log -- pre-log scaling
is a no-op on the argmax and must not be used.
"""

from __future__ import annotations

import torch


def fuse(beliefs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], grid_size: int, weights: list[float] | None = None) -> tuple[float, torch.Tensor]:
    """beliefs: list of per-contributor (gamma, nu, alpha, beta). weights: per-contributor tempering exponent, defaults to all-ones (Spec Sec 9 step 3, unweighted).

    Returns (fused_mode, log_density_sum_over_grid) -- the log-density array is
    reused by consensus.js_divergence so it is returned rather than recomputed.
    """
    raise NotImplementedError
