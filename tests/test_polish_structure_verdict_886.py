"""polish_race(p0=, structure=) carries the inverse's verdict (#886).

The declarative Factor(V, D) path discarded abilities_from_race's
convergence, so the same failed inversion that the explicit V, D call
flags came back info['converged'] == True with no warning.
"""
import warnings

import numpy as np
import pytest

from winning.factor.polish import polish_race
from winning.factor.structures import Blocks, Factor, Nested

P0 = np.array([2.5561714747719375e-8, 0.003954132588313348, 0.3838102877687196, 0.6122355540812523])
V = np.array([[-2.4482602171824595], [-3.909470185595318], [-2.465409300506771], [3.865886382696166]])
D = np.array([0.5440542468074961, 1.1197260935882694, 0.43645493804312074, 0.698143690730512])


@pytest.mark.parametrize("kw", [dict(V=V, D=D), dict(structure=Factor(V=V, D=D))], ids=["explicit", "structure"])
def test_a_failed_p0_inversion_is_not_certified(kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p, mu, info = polish_race(p0=P0, points=33, **kw)
    assert not info["converged"]
    assert any("did not converge on p0" in str(w.message) for w in caught)


@pytest.mark.parametrize("structure", [
    Factor(V=[[0.3], [-0.2], [0.1]], D=[1.0, 1.2, 0.8]),
    Blocks(cluster=[0, 0, 1], loading=[0.5, 0.5, 0.5], D=[1.0, 1.0, 1.0]),
    Nested(cluster=[0, 0, 1], loading=[0.5, 0.5, 0.5], D=[1.0, 1.0, 1.0], coupling=[0.3, 0.3, 0.3]),
], ids=["factor", "blocks", "nested"])
def test_a_reachable_p0_still_converges(structure):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p, mu, info = polish_race(p0=[0.5, 0.3, 0.2], structure=structure)
    assert info["converged"]
    np.testing.assert_allclose(p, [0.5, 0.3, 0.2], atol=1e-6)
