"""Top-k/rank lattice sizes go through shapes.as_points (#444).

int() read points=-1, 2.5 and "257" as budgets and the adaptive grid then
replaced them, while the general race refused the same values."""
import pytest

from winning.factor.topk import (abilities_from_topk, bottom_k_probabilities,
                                 rank_probabilities, top_k_jacobians,
                                 top_k_probabilities)

MU = [-0.5, 0.0, 0.5]
DOORS = {
    "top_k": lambda p: top_k_probabilities(MU, 1, points=p),
    "top_k_factor": lambda p: top_k_probabilities(MU, 1, V=[[1], [0], [-1]], points=p),
    "bottom_k": lambda p: bottom_k_probabilities(MU, 1, points=p),
    "rank": lambda p: rank_probabilities(MU, points=p),
    "jacobians": lambda p: top_k_jacobians(MU, 1, points=p),
    "inverse": lambda p: abilities_from_topk([0.5, 0.3, 0.2], 1, points=p),
}


@pytest.mark.parametrize("door", sorted(DOORS))
@pytest.mark.parametrize("points", [-1, 0, 1, 2.5, "257", True, None, float("nan")])
def test_bad_points_are_refused(door, points):
    with pytest.raises((ValueError, TypeError)):
        DOORS[door](points)


@pytest.mark.parametrize("door", sorted(DOORS))
def test_whole_points_still_price(door):
    DOORS[door](257)
