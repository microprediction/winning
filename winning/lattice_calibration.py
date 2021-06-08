from winning.lattice import state_prices_from_offsets, densities_and_coefs_from_offsets,\
    winner_of_many, expected_payoff, densities_from_offsets, implicit_state_prices
import pandas as pd
import numpy as np


#################################################################
#                                                               #
#      Implements the "method of multiplicity inversion" that   #
#      quickly solves the ratings race problem                    #
#                                                               #
#################################################################


def convert_nan_to(x,nan_value=2000):
    """ Longshots """
    if pd.isnull(x):
        return nan_value
    else:
        return x


def normalize(p):
    """ Naive renormalization of probabilities """
    S = sum(p)
    return [pr / S for pr in p]

def prices_from_dividends(dividends,nan_value=2000):
    """ Risk neutral probabilities using naive renormalization """
    return normalize([1. / convert_nan_to(x,nan_value=nan_value) for x in dividends])


def dividends_from_prices(prices, multiplicity=1.0):
    """ Australian style dividends """
    return [1.0 / (multiplicity*d) if not(np.isnan(d)) and d>0 else np.nan for d in normalize(prices)]

def normalize_dividends(dividends):
    return dividends_from_prices(prices_from_dividends(dividends))


def dividend_implied_ability(dividends, density, nan_value=2000):
    """ Infer risk-neutral implied_ability from Australian style dividends

    :param dividends:    [ 7.6, 12.0, ... ]
    :return: [ float ]   Implied ability

    """
    p = prices_from_dividends(dividends,nan_value=nan_value)
    return state_price_implied_ability(prices=p, density=density)

def state_price_implied_ability(prices, density):
    """ Calibrate offsets (translations of the performance density) to match state prices """
    implied_offsets_guess = [0 for _ in prices]
    L = int((len(density) - 1) / 2)
    offset_samples = list(range(int(-L / 2), int(L / 2)))[::-1]
    ability = implied_ability(prices=prices, density=density, \
                              offset_samples=offset_samples, implied_offsets_guess=implied_offsets_guess, nIter=3)
    return ability


def ability_implied_dividends(ability, density):
    """ Return inverse state prices
    :param ability:   [ float ]
    :param density:  [ float ]
    :return: [ 7.6, 12.3, ... ]
    """
    state_prices = state_prices_from_offsets(density=density, offsets=ability)
    return [1. / sp for sp in state_prices]


def implied_ability(prices, density, offset_samples=None, implied_offsets_guess=None, nIter=3, verbose=False,
                    visualize=False):
    """
    This is the main routine.
    It finds location translations of a fixed performance densitym so as to replicate given state prices for winning
    See the paper for details.

        offset_samples   Optionally supply a list of offsets which are used in the interpolation table  a_i -> p_i

    """

    L = int((len(density) - 1) / 2)
    if offset_samples is None:
        offset_samples = list(range(int(-L / 2), int(L / 2)))[
                         ::-1]
    else:
        _assert_descending(offset_samples)

    if implied_offsets_guess is None:
        implied_offsets_guess = list(range(int(L/3)))

    # First guess at densities
    densities, coefs = densities_and_coefs_from_offsets(density, implied_offsets_guess)
    densityAllGuess, multiplicityAllGuess = winner_of_many(densities)
    densityAll = densityAllGuess.copy()
    multiplicityAll = multiplicityAllGuess.copy()
    guess_prices = [np.sum(expected_payoff(density, densityAll, multiplicityAll, cdf=None, cdfAll=None)) for density in
                    densities]

    for _ in range(nIter):
        if visualize:
            from winning.lattice_plot import densitiesPlot
            # temporary hack to check progress of optimization
            densitiesPlot([densityAll] + densities, unit=0.1)
        implied_prices = implicit_state_prices(density=density, densityAll=densityAll, multiplicityAll=multiplicityAll,
                                               offsets=offset_samples)
        implied_offsets = np.interp(prices, implied_prices, offset_samples)
        densities = densities_from_offsets(density, implied_offsets)
        densityAll, multiplicityAll = winner_of_many(densities)
        guess_prices = [np.sum(expected_payoff(density, densityAll, multiplicityAll, cdf=None, cdfAll=None)) for density
                        in densities]
        approx_prices = [np.round(pri, 3) for pri in prices]
        approx_guesses = [np.round(pri, 3) for pri in guess_prices]
        if verbose:
            print(zip(approx_prices, approx_guesses)[:5])

    return implied_offsets


def _assert_descending(xs):
    for d in np.diff(xs):
        if d > 0:
            raise ValueError("Not descending")
