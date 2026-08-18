"""Thin shim: the canonical implementation lives in the winning package
itself (winning.factor.core). The research scripts import raceutil by
long habit; every function here is the deployed package's.

Requires the winning package importable (pip install -e <repo root>).
"""

from winning.factor.core import (  # noqa: F401
    _lattice,
    abilities_from_probabilities,
    abilities_from_probabilities_factor,
    abilities_from_win_probabilities,
    factor_model,
    factor_model_contrast,
    factor_model_projected,
    hermite_nodes,
    jacobian_vector_product,
    qmc_nodes,
    win_probabilities,
    win_probabilities_factor,
)
