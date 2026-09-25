"""Glicko-2, frozen. The pre-renovation `src/winning` package, pruned.

This was the 2.x rating-systems package before the live one moved to
`winning/`. Everything that had a maintained equivalent has been deleted;
what remains is the Glicko-2 implementation and its dependency closure,
because several chess experiments compare against it and nothing in the
live package does Glicko-2.

Nothing here is packaged, collected by pytest, or maintained. See
../README.md.
"""

from .exact import gaussian_win_probabilities
from .glicko2 import Glicko2Rating
from .ratingsystem import Rating, RatingSystem

__all__ = ["Glicko2Rating", "Rating", "RatingSystem",
           "gaussian_win_probabilities"]
