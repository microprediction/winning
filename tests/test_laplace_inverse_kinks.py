"""#267: laplace inverses on exact self-generated targets, and the
factor-rating likelihood above one.

The uniform lattice's slope integrand -f'(z) jumps at every laplace
runner's kink, so its rectangle-sum slopes and Jacobians are first
order (an own-slope came out 6x too large) and the inverses that
consumed them stalled. The inverses now take their derivatives from
the split lattice (races.kink_lattice) and keep the residual on the map
being inverted; the default FORWARD maps are unchanged. kink_split=True
opts the whole calibration (forward and inverse) onto the split lattice.
"""
import warnings

import numpy as np
import pytest

from winning.factor.races import (race_probabilities, abilities_from_race,
                                  exponential_power_base)
from winning.factor.topk import (top_k_probabilities, abilities_from_topk,
                                 loc_scale_from_topk_pair,
                                 abilities_from_rank_marginal,
                                 rank_probabilities, top_k_jacobians,
                                 bottom_k_probabilities)


PAIR_MU = np.array([-1.0758897613734006, 0.8221703173592687,
                    1.2747228138148785, -1.7811144122853875,
                    0.7601110424846411])
PAIR_SD = np.array([0.5289511696798439, 1.2097101565229562,
                    0.9433803641440728, 1.3242591310799288,
                    1.2509595951816537])
TOPK_MU = np.array([-1.7151211049140525, 1.6647723741888125,
                    0.5038679121443425, -0.5829490360565223,
                    0.12942985463742007])
TOPK_D = np.array([0.7813762250327798, 0.7548860941612756,
                   1.0407350057755838, 0.5488380588062226,
                   2.0439199583301906])


@pytest.mark.parametrize("points", [257, 513])
def test_race_inverse_round_trip_laplace(points):
    # was converged=False and a 0.099 repricing miss at 513
    mu, D = np.array([-1.5, 1.5, 0.0]), np.array([0.1, 6.0, 2.0])
    t = race_probabilities(mu, D=D, base="laplace", points=points)
    a, info = abilities_from_race(t, D=D, base="laplace", points=points,
                                  return_info=True)
    assert info["converged"]
    back = race_probabilities(a, D=D, base="laplace", points=points)
    assert np.abs(back - t).max() < 1e-8


def test_topk_inverse_round_trip_laplace():
    # was converged=False, logit residual 2.03, miss 0.0135
    q = top_k_probabilities(TOPK_MU, 3, D=TOPK_D, base="laplace")
    a, info = abilities_from_topk(q, 3, D=TOPK_D, base="laplace",
                                  return_info=True)
    assert info["converged"]
    back = top_k_probabilities(a, 3, D=TOPK_D, base="laplace")
    assert np.abs(back - q).max() < 1e-8


def test_loc_scale_pair_round_trip_laplace():
    # the report's fixture: converged=False at logit residual 1.94e-3
    D = PAIR_SD ** 2
    q1 = top_k_probabilities(PAIR_MU, 1, D=D, base="laplace")
    q2 = top_k_probabilities(PAIR_MU, 2, D=D, base="laplace")
    m, s, info = loc_scale_from_topk_pair(q1, 1, q2, 2, base="laplace",
                                          return_info=True)
    assert info["converged"] and info["max_logit_residual"] < 1e-8
    r1 = top_k_probabilities(m, 1, D=s ** 2, base="laplace")
    r2 = top_k_probabilities(m, 2, D=s ** 2, base="laplace")
    assert max(np.abs(r1 - q1).max(), np.abs(r2 - q2).max()) < 1e-8


def test_rank_marginal_round_trip_laplace():
    # every self-generated laplace target failed: the solver fitted the
    # RAW column while rank_probabilities normalises rows by the
    # lattice mass, 1.6e-3 off one for a laplace row
    mu = np.array([-0.32582296524196863, 0.47305250354111195,
                   0.6627286607399583, 1.3682900704443455,
                   1.9737782655283809])
    sd = np.array([1.1240951818414033, 1.6630497237201780,
                   0.37187049128115174, 0.4599949211347848,
                   0.7765970396809280])
    mu0 = np.array([-0.34953938962426034, 0.5153643666766584,
                    0.6403211534721777, 1.3358619652222843,
                    2.0055403112666683])
    D = sd ** 2
    t = rank_probabilities(mu, D=D, base="laplace")[:, 1]
    a, info = abilities_from_rank_marginal(t, 2, mu0=mu0, D=D,
                                           base="laplace", return_info=True)
    assert info["converged"]
    assert np.abs((a - a.mean()) - (mu - mu.mean())).max() < 1e-6


@pytest.mark.parametrize("base", ["laplace", "ep15"])
def test_kink_split_calibration_is_consistent(base):
    b = exponential_power_base(1.5) if base == "ep15" else base
    D = PAIR_SD ** 2
    q1 = top_k_probabilities(PAIR_MU, 1, D=D, base=b, kink_split=True)
    q2 = top_k_probabilities(PAIR_MU, 2, D=D, base=b, kink_split=True)
    m, s, info = loc_scale_from_topk_pair(q1, 1, q2, 2, base=b,
                                          return_info=True, kink_split=True)
    assert info["converged"]
    r2 = top_k_probabilities(m, 2, D=s ** 2, base=b, kink_split=True)
    assert np.abs(r2 - q2).max() < 1e-8
    q = top_k_probabilities(TOPK_MU, 3, D=TOPK_D, base=b, kink_split=True)
    a, info = abilities_from_topk(q, 3, D=TOPK_D, base=b, return_info=True,
                                  kink_split=True)
    assert info["converged"]
    t = race_probabilities(TOPK_MU, D=TOPK_D, base=b, kink_split=True)
    a, info = abilities_from_race(t, D=TOPK_D, base=b, return_info=True,
                                  kink_split=True)
    assert info["converged"]


def test_split_topk_matches_race_and_jacobian_matches_differences():
    # the split top-1 is the split win race, and its Jacobians are the
    # derivatives of the split map (the uniform ones are not, #267)
    D = TOPK_D
    q = top_k_probabilities(TOPK_MU, 1, D=D, base="laplace", kink_split=True)
    p = race_probabilities(TOPK_MU, D=D, base="laplace", points=2049,
                           kink_split=True)
    assert np.abs(q - p).max() < 1e-9
    Jm, Js = top_k_jacobians(TOPK_MU, 2, D=D, base="laplace",
                             kink_split=True)
    h = 1e-5
    sd = np.sqrt(D)
    for j in range(len(sd)):
        e = np.zeros(len(sd))
        e[j] = h
        fm = (top_k_probabilities(TOPK_MU + e, 2, D=D, base="laplace",
                                  kink_split=True)
              - top_k_probabilities(TOPK_MU - e, 2, D=D, base="laplace",
                                    kink_split=True)) / (2 * h)
        fs = (top_k_probabilities(TOPK_MU, 2, D=(sd + e) ** 2,
                                  base="laplace", kink_split=True)
              - top_k_probabilities(TOPK_MU, 2, D=(sd - e) ** 2,
                                    base="laplace", kink_split=True)) / (2 * h)
        assert np.abs(fm - Jm[:, j]).max() < 1e-6
        assert np.abs(fs - Js[:, j]).max() < 1e-6


def test_kink_split_off_and_smooth_bases_unchanged():
    mu, D = TOPK_MU, TOPK_D
    for base in ("normal", "logistic"):
        a = top_k_probabilities(mu, 2, D=D, base=base)
        b = top_k_probabilities(mu, 2, D=D, base=base, kink_split=True)
        assert np.array_equal(a, b)
    a = top_k_probabilities(mu, 2, D=D, base="laplace")
    b = top_k_probabilities(mu, 2, D=D, base="laplace", kink_split=False)
    assert np.array_equal(a, b)
    a = bottom_k_probabilities(mu, 2, D=D, base="laplace")
    b = bottom_k_probabilities(mu, 2, D=D, base="laplace", kink_split=True)
    assert np.abs(a - b).max() < 1e-2 and np.abs(b.sum() - 2) < 1e-9
    # columns: 1.3e-4 off one on the uniform lattice, 4.8e-6 split (the
    # rest is the count window's truncation, the same at 2049 points)
    R = rank_probabilities(mu, D=D, base="laplace", kink_split=True)
    assert np.abs(R.sum(axis=0) - 1).max() < 1e-5


def test_design_loglik_never_above_zero():
    from winning.ratings.factor_ratings import design_loglik
    theta = np.array([29.815, 0.0, -0.3, 1.0])
    events = [(np.eye(4), [0])]
    # was +3.05e-5: a winner probability of 1.0000305
    for s in np.linspace(5.0, 40.0, 141):
        th = theta.copy()
        th[0] = s
        for ev in ([0], [0, 3]):
            assert design_loglik(th, [(np.eye(4), ev)], base="laplace")[0] <= 0
    ll = design_loglik(theta, events, base="laplace")[0]
    h = 1e-3
    a, b = theta.copy(), theta.copy()
    a[0] += h
    b[0] -= h
    fd = (design_loglik(a, events, base="laplace")[0]
          - design_loglik(b, events, base="laplace")[0]) / (2 * h)
    assert ll == 0.0 and fd == 0.0


def test_inverse_reports_rather_than_raises_on_refused_iterate():
    # a top-3 laplace target near saturation whose iterates the uniform
    # lattice prices above one: honest non-convergence, not an exception
    # and not a false convergence
    mu = np.array([0.35680389, -0.68568108, 1.10259497, -0.77532861,
                   -3.05838125, 0.79406476, -0.61725912])
    D = np.array([1.06547873, 2.41853464, 2.877514, 0.29991029, 0.26024292,
                  2.28966514, 1.51227013])
    q = top_k_probabilities(mu, 3, D=D, base="laplace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, info = abilities_from_topk(q, 3, D=D, base="laplace",
                                      return_info=True)
    if info["converged"]:
        back = top_k_probabilities(a, 3, D=D, base="laplace")
        assert np.abs(back - q).max() < 1e-7
    else:
        assert np.isfinite(info["max_logit_residual"])
    # the split lattice inverts the same field cleanly
    qs = top_k_probabilities(mu, 3, D=D, base="laplace", kink_split=True)
    a, info = abilities_from_topk(qs, 3, D=D, base="laplace",
                                  return_info=True, kink_split=True)
    assert info["converged"]
