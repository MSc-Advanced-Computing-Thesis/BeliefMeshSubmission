"""Repository locations used by every results script.

    REPO       repository root
    ARTEFACTS  stored experiment outputs, one directory per report section
    FIGURES    where regenerated figures are written (results/figures)
    TABLES     where regenerated tables are written (results/tables)
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
ARTEFACTS = REPO / "artefacts"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

SEEDS = [42, 1042, 2042, 3042, 4042]


def artefact(*parts: str) -> Path:
    """artefact("5.9_ablations", "lambda_5p0") -> ARTEFACTS/5.9_ablations/lambda_5p0"""
    return ARTEFACTS.joinpath(*parts)
