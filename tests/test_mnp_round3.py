"""Multinomial-probit likelihood and estimator regressions, round 3:
the own-noise tilt (#128, #270, #352), the smooth blend statistic
(#420), the symmetrised sharp cloud (#605), both normalisations (#376),
one-alternative panels (#581) and separation (#642)."""

import warnings

import numpy as np
import pytest
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal as mvn

from winning.likelihood import (_smooth_sharpness, choice_loglik_and_score,
                                symmetric_sobol_normal)
from winning.mnprobit import MNProbit, MNProbitClassifier, _prob_of


def _ternary_oracle(mu, k=0):
    """log P(k) for independent unit-variance utilities, by a dense 1-D
    integral over the chosen alternative's noise."""
    z = np.linspace(-80, 80, 400001)
    a = -z * z / 2 - 0.5 * np.log(2 * np.pi)
    for j in range(len(mu)):
        if j != k:
            a = a + log_ndtr(mu[k] - mu[j] + z)
    M = a.max()
    return M + np.log(np.trapezoid(np.exp(a - M), z))


@pytest.mark.parametrize("g", [-1.0, -3.0, -5.0, -8.0, -20.0])
def test_rare_choices_are_not_lost_in_the_tail(g):
    """#128: mu = [g, 0, 1], no factor. g = -8 gave -35.3 against the
    exact -28.9, and -295.8 against -146.6 at -20."""
    mu = np.array([g, 0.0, 1.0])
    ll = choice_loglik_and_score(mu[None, :], np.zeros((3, 1)), [0])[0]
    assert ll == pytest.approx(_ternary_oracle(mu), abs=1e-5)


def test_factor_tail_is_close():
    """#270 fixture: V = [0, .5, .2]; -245.7 against -129.8 at gap -20.
    The own-noise tilt takes it to within 0.3%; the factor tail is the
    remainder."""
    V = np.array([[0.0], [0.5], [0.2]])
    ll = choice_loglik_and_score(np.array([[-20.0, 0, 0]]), V, [0])[0]
    assert abs(ll - (-129.77342)) < 0.5
    ll5 = choice_loglik_and_score(np.array([[-5.0, 0, 0]]), V, [0])[0]
    assert ll5 == pytest.approx(-11.317079397, abs=1e-6)


@pytest.mark.parametrize("eps", [0.5, 0.1])
def test_less_noisy_rival_is_resolved(eps):
    """#352: D = [0.1, 1, 1], observing alternative 1, gave 0.3556
    against 0.3678."""
    D = np.array([eps, 1.0, 1.0])
    A = np.array([[-1, 1, 0], [0, 1, -1.0]])
    ex = mvn(mean=np.zeros(2), cov=A @ np.diag(D) @ A.T).cdf(np.zeros(2))
    ll = choice_loglik_and_score(np.zeros((1, 3)), np.zeros((3, 1)), [1],
                                 D=D)[0]
    assert np.exp(ll) == pytest.approx(ex, rel=2e-4)


def test_tilted_score_is_the_gradient_on_all_branches():
    rng = np.random.default_rng(0)
    T, J, r = 6, 4, 2
    mu = rng.normal(size=(T, J)) * 2
    ch = rng.integers(0, J, T)
    D = rng.uniform(0.4, 1.6, J)
    base = rng.normal(size=(J, r))
    for target in (1.0, 2.95, 3.0, 3.05, 5.0):     # GH, blend, Sobol
        V = base * target / _smooth_sharpness(base, D)[0]
        ll, dmu, dV = choice_loglik_and_score(mu, V, ch, D=D)
        h = 1e-6
        for (t, j) in [(0, 1), (3, 3)]:
            e = np.zeros_like(mu)
            e[t, j] = h
            fd = (choice_loglik_and_score(mu + e, V, ch, D=D)[0]
                  - choice_loglik_and_score(mu - e, V, ch, D=D)[0]) / (2 * h)
            assert dmu[t, j] == pytest.approx(fd, abs=2e-6)
        for (j, c) in [(1, 0), (3, 1)]:
            e = np.zeros_like(V)
            e[j, c] = h
            fd = (choice_loglik_and_score(mu, V + e, ch, D=D)[0]
                  - choice_loglik_and_score(mu, V - e, ch, D=D)[0]) / (2 * h)
            assert dV[j, c] == pytest.approx(fd, abs=2e-6)


def test_blend_has_no_cusp_at_tied_rows():
    """#420 comment: tied centred row norms put a cusp in the blend
    weight and made the score label-order dependent."""
    a = 3 / np.sqrt(2)
    V = np.array([[-a], [0.0], [a]])
    mu = np.array([[-2.0, 0.4, 1.1], [0.2, -1.0, 0.7]])
    ch = np.array([0, 1])
    ll, _, dV = choice_loglik_and_score(mu, V, ch)
    h = 1e-7
    for i in range(3):
        vp, vm = V.copy(), V.copy()
        vp[i, 0] += h
        vm[i, 0] -= h
        right = (choice_loglik_and_score(mu, vp, ch)[0] - ll) / h
        left = (ll - choice_loglik_and_score(mu, vm, ch)[0]) / h
        assert abs(right - left) < 1e-5
        assert dV[i, 0] == pytest.approx(0.5 * (left + right), abs=1e-5)
    perm = [2, 1, 0]
    ll2, _, dV2 = choice_loglik_and_score(mu[:, perm], V[perm],
                                          np.array([2, 1]))
    assert ll2 == pytest.approx(ll, abs=1e-12)
    assert np.allclose(dV2[np.argsort(perm)], dV, atol=1e-12)


def test_binary_switch_boundary_is_continuous():
    a0 = 3 * np.sqrt(2.0)
    out = []
    for a in (a0 * (1 - 1e-10), a0 * (1 + 1e-10)):
        V = np.array([[0.0], [a]])
        ll, _, dV = choice_loglik_and_score(np.array([[-3.0, 0.0]]), V, [0])
        assert ll == pytest.approx(log_ndtr(-3.0 / np.sqrt(2.0 + a * a)),
                                   abs=1e-12)
        out.append(dV[1, 0])
    assert abs(out[0] - out[1]) < 1e-8


_MU605 = np.array([[0.14067943910283115, -0.1688419276279345,
                    0.8188357505677056, -0.7906732620426022]])
_V605 = np.array([[-4.44796743024364, 2.0653322851949887],
                  [1.5404628475279805, -3.7535504592745266],
                  [6.505591205368869, 0.2331481770537056],
                  [-3.5980866226532098, 1.4550699970258316]])


def _predict605(A):
    p = np.array([_prob_of(_MU605, A, k)[0] for k in range(4)])
    return p / p.sum()


@pytest.mark.parametrize("transform", [
    lambda V: V * [-1.0, 1.0], lambda V: V * [1.0, -1.0],
    lambda V: V[:, ::-1], lambda V: -V[:, ::-1]])
def test_sharp_cloud_ignores_factor_signs_and_order(transform):
    """#605: a sign flip moved the sharp-branch probability by 0.0047."""
    base = _predict605(_V605)
    assert np.abs(_predict605(transform(_V605)) - base).max() < 1e-12
    ref = np.array([0.3610138, 0.1938048, 0.4039584, 0.0412231])
    assert np.abs(base - ref).max() < 3e-3


@pytest.mark.parametrize("r", [1, 2, 3, 4, 5])
def test_symmetric_cloud_is_closed_under_its_group(r):
    F, W = symmetric_sobol_normal(r, 1, 1024, seed=0)
    assert abs(W.sum() - 1.0) < 1e-12 and len(F) <= 1200
    # #694: the group (order <= 8) never collapses the own-noise rule
    assert len(np.unique(F[:, -1])) >= 1024 // 8
    key = {tuple(np.round(row, 12)) for row in F}
    if r <= 3:          # sign flips up to r = 3
        flip = F.copy()
        flip[:, 0] = -flip[:, 0]
        assert {tuple(np.round(row, 12)) for row in flip} == key
    if r == 2:          # and permutations at r = 2
        sw = F.copy()
        sw[:, [0, 1]] = sw[:, [1, 0]]
        assert {tuple(np.round(row, 12)) for row in sw} == key


_MU694 = np.array([[0.6094341595088627, -2.079968212480991,
                    1.5009023916129145, 1.8811294327824277,
                    -3.902070377307673]])
_V694 = np.array([
    [-2.7227514556621832, 0.2673038870461732, -0.6612375437551032,
     -0.03512985407199214],
    [-1.783645483057296, 1.8387496528290126, 1.6262996869719497,
     0.13806481898674158],
    [2.356969696558986, 0.9775240168074725, -1.7967106622591427,
     0.771027902717065],
    [-2.004945542037578, 1.8367681444891217, -0.10439101989911242,
     -0.386532169172916],
    [-1.4237683041845364, 2.5562345221532174, -0.3231085806743851,
     -0.8955986445411943]])
# scipy MVN contrast CDFs, three seeds at maxpts 2e6, abseps 1e-9
_REF694 = np.array([0.17760492, 0.02211342, 0.43580930, 0.36438589,
                    0.00008646])


def test_rank_four_sharp_rule_keeps_its_resolution_694():
    """#694: the r = 4 orbit of 384 left three own-noise values; raw
    masses summed to 1.216 and the normalised error was 0.042."""
    raw = np.array([_prob_of(_MU694, _V694, k)[0] for k in range(5)])
    assert abs(raw.sum() - 1.0) < 0.01
    assert np.abs(raw / raw.sum() - _REF694).max() < 0.01


def _fixture(T=300, J=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, J, 2))
    U = X @ np.array([0.8, -0.5]) + rng.normal(size=(T, J))
    return X, U.argmax(axis=1)


def test_both_normalisations_and_the_mass_defect():
    """#376: fit maximizes raw masses, prediction normalizes; both are
    available and the gap is reported."""
    X, ch = _fixture()
    m = MNProbit(X, ch, r=1).fit()
    raw = m.predict_proba(normalize=False)
    P = m.predict_proba()
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.allclose(m.mass_defect(), 1.0 - raw.sum(axis=1))
    assert m.mass_defect_ == pytest.approx(np.abs(m.mass_defect()).max())
    assert m.mass_defect_ < 1e-3
    # the raw score is the fitted objective (on the fitting rule)
    clf = MNProbitClassifier(r=1)
    clf.model_ = m
    ll = choice_loglik_and_score(m.X @ m.params_, m.V_, ch)[0]
    assert clf.score(X, ch, normalize=False) * len(ch) == pytest.approx(ll)


def test_one_alternative_is_refused():
    """#581 comment: J = 1 returned the initializer as converged."""
    X = np.arange(8, dtype=float).reshape(8, 1, 1)
    with pytest.raises(ValueError, match="two alternatives"):
        MNProbit(X, np.zeros(8, dtype=int), intercepts=False, r=0)


def _separated(T=100):
    X = np.empty((T, 2, 1))
    ch = np.empty(T, dtype=int)
    X[:T // 2, :, 0] = [1.0, -1.0]
    ch[:T // 2] = 0
    X[T // 2:, :, 0] = [-1.0, 1.0]
    ch[T // 2:] = 1
    return X, ch


@pytest.mark.parametrize("T", [20, 100])
def test_separated_design_is_not_reported_converged(T):
    """#642: every alternative chosen, yet beta -> +inf separates."""
    X, ch = _separated(T)
    with pytest.warns(RuntimeWarning, match="separated"):
        m = MNProbit(X, ch, intercepts=False, r=0).fit(maxiter=500)
    assert not m.converged_ and m.boundary_ and m.separated_
    assert m.separation_direction_[0] == pytest.approx(1.0)


def test_separation_three_alternatives_quasi_and_control():
    from winning.mnprobit import _separation_direction
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3, 2))
    ch = X[:, :, 0].argmax(axis=1)
    d = _separation_direction(X, ch)
    assert d is not None and d[0] > 0.99
    Xq = X.copy()
    Xq[:10, :, 0] = 0.0                     # ties: quasi-separation
    assert _separation_direction(Xq, ch) is not None
    ch2 = ch.copy()
    ch2[:5] = (ch2[:5] + 1) % 3             # near-separated, finite MLE
    assert _separation_direction(X, ch2) is None
    assert _separation_direction(X * 7.0, ch) is not None    # rescaling
    Xr, chr_ = np.concatenate([X, X]), np.concatenate([ch, ch])
    assert _separation_direction(Xr, chr_) is not None       # replication
    X3, ch3 = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = MNProbit(X3, ch3, r=1).fit()
    assert m.converged_ and not m.separated_
