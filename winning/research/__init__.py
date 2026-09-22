"""The retired research engine (formerly ``winning.thurstone``).

Kept for the pipelines built on it -- the lookup-curve fitters, the
Kalman and dynamic trackers, the Laplacian tools -- none of which has a
one-line replacement yet. For the inverse problem itself, use the front
door: ``winning.calibrate_abilities(p, V=, D=)`` is materially faster
and more accurate and dispatches to the compiled kernels, which nothing
here does. See CHANGELOG for the measured gap.
"""
import warnings as _warnings

_warnings.warn(
    "winning.research is the retired research engine; for calibration "
    "under the normal race (or a built-in base= such as gumbel) use "
    "winning.calibrate_abilities(p, V=, D=, base=), which is materially "
    "faster and more accurate and uses the compiled kernels. A custom "
    "Density has no front-door equivalent and calibrates a different "
    "model: this package stays for it and for the pipelines that still "
    "need it.",
    DeprecationWarning, stacklevel=2)

from .conventions import (
    ALT_A,
    ALT_L,
    ALT_SCALE,
    ALT_UNIT,
    NAN_DIVIDEND,
    STD_A,
    STD_L,
    STD_SCALE,
    STD_UNIT,
)

# Diffeomorphism modules
from .correlated import (
    FactorRace,
    factor_model,
    gaussian_factor_race,
    gaussian_nodes,
    hermite_nodes,
    solve_abilities,
)
from .cube_to_simplex import CubeToSimplexMapping, SigmoidParams
from .density import Density
from .global_fit import GlobalAbilityCalibrator
from .global_ls import GlobalLSCalibrator
from .inference import AbilityCalibrator
from .kalman_tracker import KalmanAbilityTracker
from .laplacian import (
    InversionResult,
    LaplacianOperator,
    invert_outright_probabilities,
    laplacian_dense,
    laplacian_matvec,
    laplacian_weights,
    outright_win_probabilities,
)
from .lattice import UniformLattice
from .multiray import ConditionSpec, MultiRayGlobalCalibrator
from .optimization import OptimizationResult, ParameterBounds, optimize_diffeomorphism
from .pricing import Race, StatePricer
from .quality_assessment import QualityMetrics, comprehensive_quality_assessment

__all__ = [
    "UniformLattice",
    "Density",
    "Race",
    "StatePricer",
    "AbilityCalibrator",
    "GlobalAbilityCalibrator",
    "GlobalLSCalibrator",
    "KalmanAbilityTracker",
    "ConditionSpec",
    "MultiRayGlobalCalibrator",
    # Correlated races (latent Gaussian factors; softmax races via gumbel_min)
    "FactorRace",
    "factor_model",
    "gaussian_factor_race",
    "gaussian_nodes",
    "hermite_nodes",
    "solve_abilities",
    # Laplacian Jacobian structure and Newton-CG inversion
    "laplacian_weights",
    "laplacian_dense",
    "laplacian_matvec",
    "LaplacianOperator",
    "outright_win_probabilities",
    "invert_outright_probabilities",
    "InversionResult",
    "NAN_DIVIDEND",
    "STD_L",
    "STD_UNIT",
    "STD_SCALE",
    "STD_A",
    "ALT_L",
    "ALT_UNIT",
    "ALT_SCALE",
    "ALT_A",
    # Diffeomorphism functionality
    "CubeToSimplexMapping",
    "SigmoidParams",
    "QualityMetrics",
    "comprehensive_quality_assessment",
    "optimize_diffeomorphism",
    "ParameterBounds",
    "OptimizationResult",
]
