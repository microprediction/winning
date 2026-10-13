"""Regressions found by the 2026-10-13 benchmark rerun (#716, #717, #718)."""
import numpy as np
import pytest
from scipy.special import log_ndtr

import winning.likelihood as L
from winning.likelihood import _mills_gap, _tilt_mode

# the contrasts of the julia Fishing fit's first BFGS trial step (#716)
_C0 = np.array([-174823.91047091805, -174848.3798915476,
                -174803.45321830967])[:, None]
_T = np.array([1.6888473976942278, 1.694115546292384,
               2.9793037849297694])[:, None]


def test_tilt_curvature_stays_above_one_at_far_tail_contrasts():
    """#716: lam (a + lam) cancelled and the tilt curvature went to -196
    (julia DomainError) / -682 (python NaN)."""
    m, h, *_ = _tilt_mode(_C0, _T, 1.0)
    assert np.all(np.isfinite(m)) and np.all(h >= 1.0)
    rng = np.random.default_rng(1)
    for scale in (1e3, 1e5, 1e6, 1e8):
        c0 = -np.abs(rng.normal(scale, scale / 10, (3, 500)))
        t = rng.uniform(1, 3, (3, 1))
        m, h, dm_c, dh_c, dm_t, dh_t = _tilt_mode(c0, t, 1.0)
        assert np.all(h >= 1.0)
        for arr in (m, dm_c, dh_c, dm_t, dh_t):
            assert np.all(np.isfinite(arr))


def test_mills_gap_far_tail_matches_the_asymptotic_series():
    for a in (-1e3, -5.7e4, -1e6, -1e9):
        y = -a
        lam, gap = _mills_gap(np.array([a]))
        series = 1 / y - 2 / y ** 3 + 10 / y ** 5 - 74 / y ** 7
        assert gap[0] == pytest.approx(series, rel=1e-13)
        assert lam[0] == pytest.approx(y + series, rel=1e-15)
        assert 0.0 < lam[0] * gap[0] <= 1.0      # 1 - 1/y^2 rounds
    # nearer in, the old expression's cancellation is still small
    for a in (-25.0, -50.0):
        lam, gap = _mills_gap(np.array([a]))
        old = np.exp(-0.5 * a * a - 0.5 * np.log(2 * np.pi) - log_ndtr(a))
        assert gap[0] == pytest.approx(a + old, rel=1e-9)


def test_mills_gap_is_the_old_expression_above_the_tail_and_continuous():
    a = np.linspace(L._MILLS_TAIL, 8.0, 2001)
    lam, gap = _mills_gap(a)
    old = np.exp(-0.5 * a * a - 0.5 * np.log(2 * np.pi) - log_ndtr(a))
    assert np.array_equal(lam, old) and np.array_equal(gap, a + old)
    lo = _mills_gap(np.array([np.nextafter(L._MILLS_TAIL, -np.inf)]))
    hi = _mills_gap(np.array([L._MILLS_TAIL]))
    assert lo[0][0] == pytest.approx(hi[0][0], rel=1e-14)
    assert lo[1][0] == pytest.approx(hi[1][0], rel=1e-10)
