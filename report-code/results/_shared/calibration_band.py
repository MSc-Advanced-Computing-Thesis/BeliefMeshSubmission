# The empirical 90% coverage reference for hw:RMS.
#
# NOT 1.0. A ratio of 1.0 means the half-width equals the RMS error, which for
# a Gaussian gives ~68% coverage, not 90%. The Gaussian 90% point is 1.645, and
# the value measured across the stored runs is 1.65-1.87 depending on the
# arm's error-distribution shape:
#
#     nig_product  1.847 +/- 0.124      5.2 cross-run   1.706
#     naive        1.864 +/- 0.102      frozen          1.667 +/- 0.007
#     certainty    1.865 +/- 0.085      Gaussian ideal  1.645
#
# Measured by scaling each run's own half-widths until coverage hits exactly
# 0.90 (run_cert_mse_conventions-style readout over stored beliefs).
#
# Because it varies by arm, figures draw a BAND, never a single line: a single
# value would be wrong for some arm on every figure.

BAND = (1.65, 1.87)
BAND_LABEL = "empirical 90% reference"


def draw(ax, colour="#c1121f", on_top=False):
    """Shade the empirical 90%-coverage band on an hw:RMS axis.

    on_top: bar charts occlude a band drawn behind them, so where bars cross
    the band it is drawn OVER them at low alpha instead."""
    z = 3 if on_top else 0
    a = 0.11 if on_top else 0.13
    ax.axhspan(BAND[0], BAND[1], color=colour, alpha=a, lw=0, zorder=z)
    ax.axhline(BAND[0], color=colour, ls=":", lw=0.7, alpha=0.6, zorder=z)
    ax.axhline(BAND[1], color=colour, ls=":", lw=0.7, alpha=0.6, zorder=z)


def ensure_visible(ax, pad=1.06):
    """Keep the band in view without letting it dominate a low-valued panel."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0.0), max(hi, BAND[1] * pad))
