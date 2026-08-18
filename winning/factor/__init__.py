"""Factor-correlated races: the kinetics/Scalable-Probit-Calibration core,
come home. Forward shares, calibration, Jacobian-vector products, and
factor fitting for U = mu + V f + sqrt(D) eps, max-wins or min-wins.

Self-contained NumPy; the optional `fastrace` extension accelerates the
heavy passes transparently when importable.
"""

from .races import (  # noqa: F401
    abilities_from_race,
    calibrate_abilities,
    race_probabilities,
    removal_shares,
    tie_densities,
)
from .core import (  # noqa: F401
    abilities_from_win_probabilities,
    factor_model_contrast,
    factor_model_projected,
    hermite_nodes,
    jacobian_vector_product,
    qmc_nodes,
    win_probabilities_factor,
)
