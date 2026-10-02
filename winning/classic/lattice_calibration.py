from winning.classic.lattice import state_prices_from_extended_offsets, densities_and_coefs_from_offsets, \
    winner_of_many, expected_payoff, densities_from_offsets, implicit_state_prices, implied_L, cdf_to_pdf, \
    _exact_offset_cdfs, _exact_implicit_prices, _exact_shifted_cdf, exact_state_prices_from_cdfs, \
    as_classic_density, as_classic_prices
import numpy as np
from winning.classic.lattice_conventions import NAN_DIVIDEND

from ..rustconfig import load_fastrace

# compiled kernels (rust/fastrace); honours WINNING_PURE and use_rust()
_fastrace, _RUST_OK, _HAVE_RUST = load_fastrace('classic_exact_calibrate')


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
    # User should supply the lattice unit if they wish ability to be commensurate with some lattice
    # width that was assumed when generating the density
    implied_offsets_guess = [0 for _ in prices]
    L = implied_L(density)
    offset_samples = list(range(int(-L / 2), int(L / 2)))[::-1]
    scale_free_ability = solve_for_implied_offsets(prices=prices, density=density, \
                                                   offset_samples=offset_samples,
                                                   implied_offsets_guess=implied_offsets_guess,
                                                   nIter=3)
    return [ sfa*unit for sfa in scale_free_ability ]


# Although a unit can be provided, these calculations are morally scale free - which is to say that
# the user is providing densities defined on the natural numbers. There is clearly no loss of
# generality in that ... but see std_calibration or skew_calibration if you have specific performance
# distributions in mind.

# Here are the inverse operations...


def ability_implied_state_prices(ability, density, unit=1.0, max_depth=3):
    """ Return inverse state prices from (by default scale free) ability
        If ability is instead interpreted in reference to an implied lattice width, then user must supply that unit length
        This should be the unit that was assumed when creating the densities.
    :param ability:   [ float ]
    :param density:  [ float ]
    :param max_depth 
    :return: [ 7.6, 12.3, ... ]
    """
    scale_free_offsets = [ a/unit for a in ability ]
    return state_prices_from_extended_offsets(density=density, offsets=scale_free_offsets, max_depth=max_depth)


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
    """ Risk neutral probabilities using naive renormalization

    Only a MISSING quote (None or NaN) becomes ``nan_value``. A
    non-positive dividend, -inf with it, is worth nothing and prices at 0;
    +inf prices at 1/inf = 0. The total is normalised only when it is
    positive, so an all-infinite book is all zeros. This took 1/x
    unconditionally: a zero dividend raised ZeroDivisionError, a negative
    one became a NEGATIVE probability that dividend_implied_ability then
    calibrated to, and an all-infinite book divided 0 by 0 (#423). The
    rule is StatePricer.prices_from_dividends' and the R/browser ports'.
    """
    inv = []
    for x in dividends:
        v = nan_value if (x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))) else float(x)
        inv.append(0.0 if v <= 0 else 1.0 / v)
    S = sum(inv)
    return [pr / S for pr in inv] if S > 0 else inv


def dividends_from_prices(prices, multiplicity=1.0):
    """ Australian style "dividends" """
    return [1.0 / (multiplicity * d) if not (np.isnan(d)) and d > 0 else np.nan for d in normalize(prices)]


def normalize_dividends(dividends):
    return dividends_from_prices(prices_from_dividends(dividends))




def solve_for_implied_offsets(prices, density, offset_samples=None,
                              implied_offsets_guess=None,
                              nIter=3, verbose=False,
                              visualize=False):
    """
    This is the main routine.

    See the paper for details, in the /doc folder
    https://github.com/microprediction/winning/blob/main/docs/Inferring_Relative_Ability_SIAM_updated.pdf

        offset_samples   Optionally supply a list of offsets which are used in the interpolation table  a_i -> p_i
        verbose, visualize   Per-iteration diagnostics. Requesting either
                         runs the pure-Python iteration even when the
                         compiled kernel is installed (same answer).

    """

    density = as_classic_density(density)
    prices = as_classic_prices(prices)
    L = implied_L(density)
    if offset_samples is None:
        offset_samples = list(range(int(-L / 2), int(L / 2)))[
                         ::-1]
    else:
        if len(offset_samples) == 0:
            raise ValueError('offset_samples is empty; there is nothing to interpolate against')
        if not np.all(np.isfinite(np.asarray(offset_samples, dtype=float))):
            raise ValueError('offset_samples has a non-finite offset')
        _assert_descending(offset_samples)

    # One starting offset per target price. The default was
    # range(int(L/3)) -- lattice WIDTH as contestant count, a five-runner
    # first field for a two-runner target at L=15 -- and a guess of any
    # length was accepted (#369). The high-level wrappers always passed
    # zeros; now the low-level default agrees and a mismatch is refused.
    if implied_offsets_guess is None:
        implied_offsets_guess = [0.0 for _ in prices]
    elif len(implied_offsets_guess) != len(prices):
        raise ValueError('implied_offsets_guess must have one starting offset per price: got '
                         + str(len(implied_offsets_guess)) + ' for ' + str(len(prices)) + ' prices')

    # Diagnostics (verbose / visualize) need the per-iteration state, so
    # requesting them selects the Python backend; the answer is the same.
    if _HAVE_RUST and not verbose and not visualize:
        return list(_fastrace.classic_exact_calibrate(
            [float(d) for d in density], [float(p) for p in prices],
            [float(o) for o in offset_samples],
            [float(o) for o in implied_offsets_guess], nIter))

    # The paper's fixed point: tabulate offset -> price against the
    # current field and read the targets off the table. The field and
    # the table are priced by the exact dead-heat engine
    # (winning.classic.lattice, #418/#362/#348/#373), and each step is a
    # DEFECT CORRECTION
    #
    #     a_i  <-  a_i + T^{-1}(p_i) - T^{-1}(P_i(a)),
    #
    # P_i(a) the exact price of runner i in the current field. The table
    # T is exact only for a field member, and between integer samples it
    # is a linear interpolation, so reading p_i straight off it
    # (a_i <- T^{-1}(p_i)) converges to the fixed point of the TABLE: on
    # a three-atom law that missed the exact forward by 1e-2 however many
    # iterations ran. Subtracting the table's own reading of the current
    # price cancels that bias, so the fixed point is P(a) = p exactly;
    # where the table is exact the step is the paper's.
    base, cdfs, L = _exact_offset_cdfs(density, implied_offsets_guess)
    implied_offsets = np.asarray(implied_offsets_guess, dtype=float)
    for _ in range(nIter):
        if visualize:
            from winning.classic.lattice_plot import densitiesPlot
            densitiesPlot([cdf_to_pdf(c) for c in cdfs], unit=0.1)
        implied_prices = _exact_implicit_prices(base, cdfs, offset_samples, L)
        current = exact_state_prices_from_cdfs(cdfs)
        implied_offsets = implied_offsets + (
            np.interp(prices, implied_prices, offset_samples)
            - np.interp(current, implied_prices, offset_samples))
        cdfs = [_exact_shifted_cdf(base, o, L) for o in implied_offsets]
        if verbose:
            print(list(zip(np.round(prices, 3),
                           np.round(exact_state_prices_from_cdfs(cdfs), 3)))[:5])

    return implied_offsets


def _assert_descending(xs):
    for d in np.diff(xs):
        if d > 0:
            raise ValueError("Not descending")
