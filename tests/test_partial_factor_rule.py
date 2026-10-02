"""A factor rule is nodes AND weights (#290), and the pair inverse is
certified by the forward for a caller's rule (#374)."""
import numpy as np
import pytest

from winning.factor.blocks import roots_hermitenorm
from winning.factor.races import abilities_from_race, race_probabilities


def test_a_lone_f_or_w_is_refused_forward_and_inverse():
    mu = np.zeros(3)
    V = np.array([[-1.0], [0.0], [1.0]])
    D = np.full(3, 0.25)
    with pytest.raises(ValueError, match="both F"):
        race_probabilities(mu, V=V, D=D, F=np.array([[2.0]]), points=257)
    with pytest.raises(ValueError, match="both F"):
        race_probabilities(mu, V=V, D=D, W=[1.0], points=257)
    with pytest.raises(ValueError, match="both F"):
        abilities_from_race([0.8, 0.15, 0.05], V=V, D=D, F=np.array([[2.0]]))


def _hermite(q):
    x, w = roots_hermitenorm(q)
    return x[:, None], w / w.sum()


@pytest.mark.parametrize("label,V,D,F,W,truth", [
    ("translated Hermite", np.array([[0.0], [1.0]]), np.array([1.0, 1.0]),
     _hermite(31)[0] + 1.0, _hermite(31)[1], np.array([-0.5, 0.5])),
    ("two-point law", np.array([[0.0], [1.0]]), np.array([0.05, 0.05]),
     np.array([[-1.0], [1.0]]), np.array([0.5, 0.5]), np.array([-0.4, 0.4])),
])
def test_the_pair_inverse_reprices_a_caller_rule(label, V, D, F, W, truth):
    # the closed form dropped the rule's mean (21.8 points) and treats any
    # law as Gaussian (a centred two-point law, 12 points); it is now a
    # start certified by the forward
    t = race_probabilities(truth, V=V, D=D, F=F, W=W, points=1025)
    mu, info = abilities_from_race(t, V=V, D=D, F=F, W=W, points=1025,
                                   return_info=True)
    back = race_probabilities(mu, V=V, D=D, F=F, W=W, points=1025)
    assert info["converged"], (label, info)
    assert np.abs(back - t).max() < 1e-7, label
