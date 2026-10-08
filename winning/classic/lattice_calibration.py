from winning.classic.lattice import state_prices_from_extended_offsets, densities_and_coefs_from_offsets, \
    winner_of_many, expected_payoff, densities_from_offsets, implicit_state_prices, implied_L, cdf_to_pdf, \
    _exact_offset_cdfs, _exact_implicit_prices, _exact_shifted_cdf, exact_state_prices_from_cdfs, \
    as_classic_density, as_classic_prices, state_prices_from_offsets
import warnings

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
    """ Calibrate offsets (translations of the performance density) to match state prices

    An entrant priced at exactly zero -- a scratched runner, or a zero,
    negative or infinite dividend, which prices_from_dividends maps to 0 --
    gets ability +inf (never wins; ability_implied_state_prices prices it
    at 0) and the rest are calibrated among themselves. Handing the zero
    to the finite lattice inverse returned an ordinary offset that
    repriced the scratched runner at 4.5% (16% at L=3) (#589).
    """
    # By default this returns scale free offsets.
    # User should supply the lattice unit if they wish ability to be commensurate with some lattice
    # width that was assumed when generating the density
    p = as_classic_prices(prices)
    live = [i for i, x in enumerate(p) if x > 0]
    ability = [float('inf')] * len(p)
    if len(live) == 1:
        ability[live[0]] = 0.0
    else:
        offsets = solve_for_implied_offsets(prices=[p[i] for i in live], density=density, nIter=3)
        for i, o in zip(live, offsets):
            ability[i] = float(o) * unit
    return ability


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
    core = None
    if offset_samples is None:
        offset_samples = default_offset_samples(L)
        core = _core_offset_samples(L)
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
    #
    # After each step the field is re-centred by the integer part of its
    # mean (the gauge; integer shifts are exact on the lattice). The table
    # is absolute, so a field that drifted to one side wasted half of it
    # and a 97/3 book stalled at 88/12 however many iterations ran (#498).
    implied_offsets = np.asarray(implied_offsets_guess, dtype=float)
    # Diagnostics (verbose / visualize) need the per-iteration state, so
    # requesting them selects the Python backend; the answer is the same.
    if _HAVE_RUST and not verbose and not visualize:
        d_list = [float(d) for d in density]
        p_list = [float(p) for p in prices]
        s_list = [float(o) for o in offset_samples]
        for _ in range(nIter):
            # one step per call, centred here, so an older fastrace
            # without the centring gives the same iterates
            implied_offsets = _gauge_centred(np.asarray(_fastrace.classic_exact_calibrate(
                d_list, p_list, s_list, [float(o) for o in implied_offsets], 1), dtype=float))
    else:
        base, cdfs, L = _exact_offset_cdfs(density, implied_offsets)
        for _ in range(nIter):
            if visualize:
                from winning.classic.lattice_plot import densitiesPlot
                densitiesPlot([cdf_to_pdf(c) for c in cdfs], unit=0.1)
            current = exact_state_prices_from_cdfs(cdfs)
            samples, implied_prices = _table(base, cdfs, offset_samples, core, L, prices, current)
            implied_offsets = _gauge_centred(implied_offsets + (
                np.interp(prices, implied_prices, samples)
                - np.interp(current, implied_prices, samples)))
            cdfs = [_exact_shifted_cdf(base, o, L) for o in implied_offsets]
            if verbose:
                print(list(zip(np.round(prices, 3),
                               np.round(exact_state_prices_from_cdfs(cdfs), 3)))[:5])

    if nIter > 0:
        _warn_if_unconverged(state_prices_from_offsets(density, implied_offsets), prices)
    return implied_offsets


# A calibration whose own reprice misses a target by more than this is
# reported. The table iteration on the full representable grid reaches
# ~1e-4 or better in three steps on ordinary books; a miss beyond 1e-3
# means the target is outside what the lattice can represent (an exact
# zero, a longshot past the lattice edge) or nIter was too small (#498).
CALIBRATION_WARN_TOL = 1e-3


def default_offset_samples(L):
    """The default interpolation table: every integer offset the lattice
    represents, L-2 down to -(L-2) (low_high pins anything beyond). The
    half-lattice grid range(-L/2, L/2) endpoint-clamped targets the
    forward could reach: a 97/3 book repriced at 88/12 (#498)."""
    return list(range(-(L - 2), L - 1))[::-1]


def _core_offset_samples(L):
    """The old half-lattice table, a contiguous slice of the default."""
    return list(range(int(-L / 2), int(L / 2)))[::-1]


def _table(base, cdfs, offset_samples, core, L, prices, current):
    """(samples, prices) for one step. With the default table, price the
    central slice first and extend to the full range only when a lookup
    reaches the slice's ends: inside it the interpolation is identical
    (same integer samples, same field), so the answer is the full table's
    at the half table's cost on ordinary books."""
    if core is None:
        return offset_samples, _exact_implicit_prices(base, cdfs, offset_samples, L)
    t = _exact_implicit_prices(base, cdfs, core, L)
    lo, hi = t[0], t[-1]
    if all(lo < x < hi for x in prices) and all(lo < x < hi for x in current):
        return core, t
    top = [k for k in offset_samples if k > core[0]]
    bottom = [k for k in offset_samples if k < core[-1]]
    return offset_samples, (_exact_implicit_prices(base, cdfs, top, L) + list(t)
                            + _exact_implicit_prices(base, cdfs, bottom, L))


def _gauge_centred(offsets):
    """Shift by the integer part of the mean (toward zero, as int()), an
    exact lattice translation that keeps the field over the table."""
    offsets = np.asarray(offsets, dtype=float)
    return offsets - float(int(np.mean(offsets))) if offsets.size else offsets


def _warn_if_unconverged(repriced, prices):
    miss = float(np.max(np.abs(np.asarray(repriced, dtype=float) - np.asarray(prices, dtype=float))))
    if miss > CALIBRATION_WARN_TOL:
        warnings.warn('solve_for_implied_offsets did not reach the target: max |price error| = '
                      + format(miss, '.3g') + ' after calibration. The target may lie outside what '
                      'this lattice represents (an exact zero, or a longshot past the lattice edge: '
                      'use a wider L or finer unit), or nIter is too small.', RuntimeWarning, stacklevel=3)


def _assert_descending(xs):
    for d in np.diff(xs):
        if d > 0:
            raise ValueError("Not descending")
