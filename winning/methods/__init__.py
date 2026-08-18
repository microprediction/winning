"""The arena: choice/winning-probability methods behind one interface.

Every method computes the same object -- the N-vector of winning
probabilities P(alternative i has the extreme utility) for the factor model
U = mu + V f + sqrt(D) eps (max-wins convention here throughout) -- and is
registered with a name, so the benchmark harness (winning.bench) can run
them interchangeably and record accuracy-time results.

Contestants implemented natively (self-contained NumPy):
    lattice        the shared-survival-field transform (this package's
                   original algorithm, factor-generalized)
    direct_mc      draw utilities, argmax, average
    sobol_direct   the same with scrambled-Sobol points
    factor_rqmc    per-alternative (k+1)-dimensional conditioned RQMC
    ghk            per-alternative sequential importance sampling
    qmc_ghk        GHK with scrambled-Sobol uniforms
    tilting        Botev-style minimax exponential tilting

External contestants are wrapped when importable (e.g. thurstone's
FactorRace; the fastrace compiled kernels accelerate `lattice` when
present). The arena imports contestants, never the other way around.
"""

from .registry import METHODS, get_method, register  # noqa: F401
from . import native  # noqa: F401  (registers the built-in contestants)
from . import orthant_extra  # noqa: F401  (second wave: GB, ME, EP, SMC)
