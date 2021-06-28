from winning.lattice import state_prices_from_offsets, densities_and_coefs_from_offsets, \
    winner_of_many, expected_payoff, densities_from_offsets, implicit_state_prices
import numpy as np
from winning.lattice_conventions import NAN_DIVIDEND


#################################################################
#                                                               #
#      Implements a fast algorithm for inferring                #
#      relative location parameters of performance              #
#      distributions, given contest win probabilities           #
#                                                               #
#################################################################

# The main two functions are listed first. They can be used to solve the horse race problem when
# provided "dividends" (i.e. decimal prices).

# The inverse of a dividend is called a state price. So if you want to provide winning probabilities
# and ignore the possibility of dead-heats, you are well served by 'state_price_implied_ability'




def dividend_implied_ability(dividends, density, nan_value=NAN_DIVIDEND, unit=1.0):
    """ Infer risk-neutral solve_for_implied_offsets from Australian style dividends

    :param dividends:    [ 7.6, 12.0, ... ]
    :return: [ float ]   Implied ability

    """
    # By default this returns scale free offsets.
    # User should supply the lattice unit if they wish ability to be commensurate with some latice
    # width that was assumed when generating the density
    p = prices_from_dividends(dividends, nan_value=nan_value)
    return state_price_implied_ability(prices=p, density=density, unit=unit)


def state_price_implied_ability(prices, density, unit=1.0):
    """ Calibrate offsets (translations of the performance density) to match state prices """
    # By default this returns scale free offsets.
    # User should supply the lattice unit if they wish ability to be commensurate with some latice
    # width that was assumed when generating the density
    implied_offsets_guess = [0 for _ in prices]
    L = int((len(density) - 1) / 2)
    offset_samples = list(range(int(-L / 2), int(L / 2)))[::-1]
    scale_free_ability = solve_for_implied_offsets(prices=prices, density=density, \
                                                   offset_samples=offset_samples, implied_offsets_guess=implied_offsets_guess, nIter=3)
    return [ sfa*unit for sfa in scale_free_ability ]


# Although a unit can be provided, these calculations are morally scale free - which is to say that
# the user is providing densities defined on the natural numbers. There is clearly no loss of
# generality in that ... but see std_calibration or skew_calibration if you have specific performance
# distributions in mind.

# Here are the inverse operations...


def ability_implied_state_prices(ability, density, unit=1.0):
    """ Return inverse state prices from (by default scale free) ability
        If ability is instead interpreted in reference to an implied lattice width, then user must supply that unit length
        This should be the unit that was assumed when creating the densities.
    :param ability:   [ float ]
    :param density:  [ float ]
    :return: [ 7.6, 12.3, ... ]
    """
    scale_free_offsets = [ a/unit for a in ability ]
    return state_prices_from_offsets(density=density, offsets=scale_free_offsets)


def safe_inv(x, nan_value=np.nan):
    try:
        return 1/x
    except:
        return nan_value


def ability_implied_dividends(ability, density, unit=1.0, nan_value=NAN_DIVIDEND):
    """ Return inverse state prices from (by default scale free) ability
        If ability is instead interpreted in reference to an implied lattice width, then user must supply that unit length
        This should be the unit that was assumed when creating the densities.
    :param ability:  [ float ]
    :param density:  [ float ]
    :return: [ 7.6, 12.3, ... ]
    """
    state_prices = ability_implied_state_prices(ability=ability, density=density, unit=unit)
    return [ safe_inv(sp) for sp in state_prices]



def convert_nan_to(x, nan_value=NAN_DIVIDEND):
    """ Often horses that have no "realistic" odds might appear as nan, since the maximum
        price on Betfair is 1000.0, for example
    """
    if np.isnan(x):
        return nan_value
    else:
        return x


def normalize(p):
    """ Naive renormalization of probabilities """
    S = sum(p)
    return [pr / S for pr in p]


def prices_from_dividends(dividends, nan_value=NAN_DIVIDEND):
    """ Risk neutral probabilities using naive renormalization """
    return normalize([1. / convert_nan_to(x, nan_value=nan_value) for x in dividends])


def dividends_from_prices(prices, multiplicity=1.0):
    """ Australian style "dividends" """
    return [1.0 / (multiplicity * d) if not (np.isnan(d)) and d > 0 else np.nan for d in normalize(prices)]


def normalize_dividends(dividends):
    return dividends_from_prices(prices_from_dividends(dividends))




def solve_for_implied_offsets(prices, density, offset_samples=None, implied_offsets_guess=None, nIter=3, verbose=False,
                              visualize=False):
    """
    This is the main routine.

    See the paper for details, in the /doc folder
    https://github.com/microprediction/winning/blob/main/docs/Horse_Race_Problem__SIAM_updated.pdf

        offset_samples   Optionally supply a list of offsets which are used in the interpolation table  a_i -> p_i

    """

    L = int((len(density) - 1) / 2)
    if offset_samples is None:
        offset_samples = list(range(int(-L / 2), int(L / 2)))[
                         ::-1]
    else:
        _assert_descending(offset_samples)

    if implied_offsets_guess is None:
        implied_offsets_guess = list(range(int(L / 3)))

    # First guess at densities
    densities, coefs = densities_and_coefs_from_offsets(density, implied_offsets_guess)
    densityAllGuess, multiplicityAllGuess = winner_of_many(densities)
    densityAll = densityAllGuess.copy()
    multiplicityAll = multiplicityAllGuess.copy()

    if verbose:
        guess_prices = [np.sum(expected_payoff(density, densityAll, multiplicityAll, cdf=None, cdfAll=None)) for density in
                    densities]

    for _ in range(nIter):
        if visualize:
            from winning.lattice_plot import densitiesPlot
            # temporary hack to check progress of optimization
            densitiesPlot([densityAll] + densities, unit=0.1)

        # Main iteration...
        implied_prices = implicit_state_prices(density=density, densityAll=densityAll, multiplicityAll=multiplicityAll,
                                               offsets=offset_samples)
        implied_offsets = np.interp(prices, implied_prices, offset_samples)
        densities = densities_from_offsets(density, implied_offsets)
        densityAll, multiplicityAll = winner_of_many(densities)

        if verbose:
            guess_prices = [np.sum(expected_payoff(density, densityAll, multiplicityAll, cdf=None, cdfAll=None)) for density
                            in densities]
            approx_prices  = [np.round(pri, 3) for pri in prices]
            approx_guesses = [np.round(pri, 3) for pri in guess_prices]

            print(zip(approx_prices, approx_guesses)[:5])

    return implied_offsets


def _assert_descending(xs):
    for d in np.diff(xs):
        if d > 0:
            raise ValueError("Not descending")
