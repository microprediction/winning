from winning.lattice_calibration import state_price_implied_ability, dividend_implied_ability, ability_implied_dividends
from winning.lattice import center_density, skew_normal_density
from winning.lattice_conventions import ALT_UNIT, ALT_L, ALT_SCALE, ALT_A

# Specializing the ratings race problem to skew-normally distributed horses


def skew_state_price_implied_ability(prices: [float]) -> [float]:
    """ Calibrate to a skewed performance distribution

            prices:    state prices (very similar to probabilities of winning)
            returns:   implied ratings corresponding to location parameters (i.e. relative translations of perf distribution)

    """
    density = skew_normal_density(L=ALT_L, unit=ALT_UNIT, loc=0., scale=ALT_SCALE, a=ALT_A)
    return state_price_implied_ability(prices=prices, density=density)

def skew_density(loc):
    """ An canonical choice of skew-normal density using hard-coded parameters """
    density = skew_normal_density(L=ALT_L, unit=ALT_UNIT, loc=loc, scale=ALT_SCALE, a=ALT_A)
    return center_density( density )

def skew_dividend_implied_ability(dividends):
    """ Implied ability using skewed normal performances, from inverse win probabilities """
    return dividend_implied_ability(dividends=dividends, density=skew_density(loc=0.0))

def skew_ability_implied_dividends(ability):
    """ Inverse probabilities based on a canonical choice of skew normal density """
    return ability_implied_dividends(ability, density=skew_density(loc=0.0))


