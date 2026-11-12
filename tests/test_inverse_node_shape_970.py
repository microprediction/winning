"""The race inverse checks a caller's node rule as the forward does (#970).

as_factor_law accepts flat nodes, so the inverse solved and certified a
rank-1 rule given as a 1-D F at residual 0, while race_probabilities
refused the same rule. The shared check runs before the inverse's moments,
pair closed form or certificate.
"""
import numpy as np
import pytest

from winning.factor.races import abilities_from_race, race_probabilities

OPTS = dict(V=[[0.5], [-0.5]], D=[0.5, 0.5], W=[0.5, 0.5])


@pytest.mark.parametrize("target", [[0.6, 0.4], [0.5, 0.3, 0.2]])
def test_a_flat_node_vector_is_refused_by_both_doors(target):
    n = len(target)
    opts = dict(V=np.linspace(-0.5, 0.5, n)[:, None], D=np.full(n, 0.5),
                F=[-1.0, 1.0], W=[0.5, 0.5])
    with pytest.raises(ValueError, match="F must be a 2-D"):
        race_probabilities(np.zeros(n), **opts)
    with pytest.raises(ValueError, match="F must be a 2-D"):
        abilities_from_race(target, **opts)


@pytest.mark.parametrize("F, W, match", [
    ([[-1.0, 0.0], [1.0, 0.0]], [0.5, 0.5], "factor columns"),
    ([[-1.0], [1.0]], [0.5, 0.3, 0.2], "nodes but W has"),
    ([[-1.0], [1.0]], [[0.5, 0.5]], "W must be a 1-D"),
])
def test_rank_and_count_mismatches_are_refused_by_the_inverse(F, W, match):
    with pytest.raises(ValueError, match=match):
        abilities_from_race([0.6, 0.4], V=[[0.5], [-0.5]], D=[0.5, 0.5], F=F, W=W)


def test_the_shaped_rule_still_inverts_and_reprices():
    opts = dict(OPTS, F=[[-1.0], [1.0]])
    mu, info = abilities_from_race([0.6, 0.4], **opts, return_info=True)
    assert info["converged"]
    np.testing.assert_allclose(race_probabilities(mu, **opts), [0.6, 0.4], atol=1e-12)
