"""The named bases floor S at 1e-300 as a guard for log(0); summed as
mass it plateaued rare winners at ~1e-300 while exact tails stayed
representable (#502). Floored points now take the exact log survival,
so nothing above that range moves."""

import numpy as np
import pytest
from scipy.special import ndtr

from winning.factor import races as R
from winning.factor.topk import rank_probabilities, top_k_probabilities

NORMAL_53 = float(ndtr(-53 / np.sqrt(2)))       # 1.1054538321e-307


@pytest.fixture
def pure(monkeypatch):
    monkeypatch.setattr(R, "_HAVE_RUST", False)


def test_three_runner_normal_race_below_the_floor(pure):
    p = R.race_probabilities(np.array([0.0, 53.0, 106.0]))
    assert p[1] == pytest.approx(NORMAL_53, rel=1e-6)


@pytest.mark.parametrize("base,gap,exact", [
    ("laplace", 500.0, np.exp(-500 * np.sqrt(2)) * (500 * np.sqrt(2) / 4 + .5)),
    ("logistic", 400.0, 5.8996759851e-313)])
def test_named_bases_below_the_floor(pure, base, gap, exact):
    p = R.race_probabilities(np.array([0.0, gap]), base=base, window="span",
                             points=65537)
    assert p[1] == pytest.approx(exact, rel=1e-5)


def test_top1_and_rank_marginals_below_the_floor():
    assert top_k_probabilities(np.array([0.0, 53.0, 106.0]), 1)[1] == \
        pytest.approx(NORMAL_53, rel=1e-6)
    P = rank_probabilities(np.array([0.0, 53.0, 106.0]))
    assert P[0, 1] == pytest.approx(NORMAL_53, rel=1e-6)


def test_ordinary_results_do_not_move(pure):
    mu = np.array([0.0, 0.7, 1.5, 3.0])
    for base in ("normal", "gumbel", "logistic", "laplace"):
        fn = R.BASES[base]
        z = np.linspace(-30, 30, 101)
        S = fn(z)[0]
        assert np.array_equal(R._log_survival(fn, z, S)[S > 1e-300],
                              np.log(S[S > 1e-300]))
        assert np.isfinite(R.race_probabilities(mu, base=base)).all()
