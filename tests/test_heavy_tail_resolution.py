"""Base-aware lattice resolution for heavy-tailed Student bases (#385, #386).

References are direct improper quadrature of the two-runner integral
P(X1 < X0) = int f0(x) F1(x) dx."""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t

from winning import rustconfig
from winning.factor.races import race_probabilities, student_base


@pytest.fixture(autouse=True)
def _pure():
    was = rustconfig.rust_active()
    rustconfig.use_rust(False)
    yield
    rustconfig.use_rust(was)


def _student_ref(nu, gap):
    s = np.sqrt(nu / (nu - 2.0))
    return quad(lambda x: t.pdf(s * x, nu) * s * t.cdf(s * (x - gap), nu),
                -np.inf, np.inf, epsabs=1e-13, epsrel=1e-12, limit=2000)[0]


@pytest.mark.parametrize("nu", [2.1, 2.5, 3.0])
def test_more_points_never_make_the_student_race_worse(nu):
    ref = _student_ref(nu, 0.5)
    errs = []
    for points in (2001, 4001, 8001, 16001, 32769):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = race_probabilities([0.0, 0.5], D=[1.0, 1.0], base=student_base(nu),
                                   points=points, window="bulk")
        errs.append(abs(p[1] - ref))
    # was 9.4e-6 at 2001 then 3.8e-3 at 4001 and 3.9e-3 at 32769
    assert max(errs) < 2e-6, errs
    for a, b in zip(errs, errs[1:]):
        assert b <= 1.5 * a + 1e-10, errs


def test_an_underresolved_student_window_says_so():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        race_probabilities([0.0, 0.5], D=[1.0, 1.0], base=student_base(2.1),
                           points=1001, window="bulk")
    assert any("lattice spacing" in str(w.message) for w in seen)


def test_student_declares_its_central_scale():
    assert student_base(2.1).resolution == pytest.approx(np.sqrt(0.1 / 2.1))
    assert student_base(1e6).resolution == pytest.approx(1.0, abs=1e-5)


# --- #386 ----------------------------------------------------------------

@pytest.mark.parametrize("nu", [2.01, 2.1, 2.5, 3.0])
@pytest.mark.parametrize("points", [None, 4001, 16001, 32769])
def test_top_k_student_race_prices_instead_of_raising(nu, points):
    from winning.factor.topk import top_k_probabilities
    kw = {} if points is None else dict(points=points)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = top_k_probabilities([0.0, 0.5], 1, D=[1.0, 1.0],
                                base=student_base(nu), **kw)
    assert abs(q.sum() - 1) < 1e-9
    assert q[1] == pytest.approx(_student_ref(nu, 0.5), abs=1e-6)


def test_top_k_relaxation_is_reported():
    from winning.factor.topk import top_k_probabilities
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        top_k_probabilities([0.0, 0.5], 1, D=[1.0, 1.0], base=student_base(2.1))
    assert any("relaxed delta" in str(w.message) for w in seen)
