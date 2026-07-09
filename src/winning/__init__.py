"""winning: rating systems and benchmarks built on the thurstone ability transform.

The core algorithms (lattice order statistics, the fast ability transform)
live in the `thurstone` package; this package applies them to rating systems
and benchmarks them against TrueSkill, OpenSkill, Glicko-2 and Elo.

Versions 1.x of `winning` contained the original SIAM-paper implementation,
now re-homed in `thurstone`.
"""

from .elo import EloRating
from .exact import gaussian_win_probabilities
from .glicko2 import Glicko2Rating
from .ratingsystem import Rating, RatingSystem
from .thurstonerating import ThurstoneRating

__version__ = "2.0.0.dev0"

__all__ = [
    "Rating",
    "RatingSystem",
    "EloRating",
    "Glicko2Rating",
    "ThurstoneRating",
    "gaussian_win_probabilities",
    "__version__",
]
