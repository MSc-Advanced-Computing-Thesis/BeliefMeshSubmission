# Thesis-facing display names for the aggregation arms.
#
# The code names (nig_product, nig_product_avgtrain, gossip_uniform, ...) must
# not appear in any figure axis label, legend entry, annotation or table row.
# Every generator imports from here so the convention cannot drift.

DISPLAY = {
    "frozen": "Frozen",
    "naive": "Naive",
    "certainty": "Certainty",
    "nig_product": "Product fusion",
    "nig_product_avgtrain": "Average fusion",
    "nig_product_weighted": "Weighted product fusion",
    # parameter-exchange comparators (Section 5.5)
    "fusion": "Product fusion",
    "gossip_uniform": "Gossip uniform",
    "gossip_weighted": "Gossip weighted",
    "fedavg_global": "FedAvg",
}

# two-line variants, for tick labels where horizontal room is tight
WRAPPED = {
    "nig_product": "Product\nfusion",
    "nig_product_avgtrain": "Average\nfusion",
    "gossip_uniform": "Gossip\nuniform",
    "gossip_weighted": "Gossip\nweighted",
    "fusion": "Product\nfusion",
}


def name(code: str, wrapped: bool = False) -> str:
    if wrapped and code in WRAPPED:
        return WRAPPED[code]
    return DISPLAY.get(code, code)
