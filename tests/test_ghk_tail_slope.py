"""GHK keeps its tail slope (#270).

The truncated draw Phi^-1(u Phi(b)) was clipped at 1e-300, so behind a
bound deeper than about -37 every draw sat at Phi^-1(1e-300) and the
later conditional factors were evaluated at the wrong point: at gap -80
the log probability came back -2378.5 against -2001.25. References are
a 1-d conditional-normal quadrature in log space (J = 3, rank one)."""
import numpy as np
import pytest

from winning.methods.native import _contrast, _ghk_prob

V = np.array([[0.3], [-0.2], [0.1]])
D = np.ones(3)
REFS = [(-10.0, -37.51799189156518), (-50.0, -789.4427401866274),
        (-60.0, -1131.5948698859718), (-80.0, -2001.2502944154676)]


@pytest.mark.parametrize("g,ref", REFS)
def test_three_way_tail_tracks_quadrature(g, ref):
    u = np.random.default_rng(1).random((4000, 2))
    mu = np.array([g, 0.0, 0.5])
    lp = _ghk_prob(mu, _contrast(0, V=V, D=D), 0, 4000, u)
    assert abs(lp - ref) < 0.05


def test_binary_tail_is_exact_far_below_1e300():
    from scipy.special import log_ndtr
    u = np.random.default_rng(1).random((16, 1))
    for g in (53.0, 200.0):
        lp = _ghk_prob(np.array([0.0, g]), _contrast(0, V=np.zeros((2, 1)),
                                                     D=np.ones(2)), 0, 16, u)
        assert lp == pytest.approx(log_ndtr(-g / np.sqrt(2)), rel=1e-12)
