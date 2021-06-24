from winning.lattice_calibration import state_price_implied_ability, dividend_implied_ability, \
    ability_implied_dividends, ability_implied_state_prices
from winning.lattice import skew_normal_density
from winning.lattice_conventions import ALT_UNIT, ALT_L, ALT_SCALE, ALT_A

# Specializing the horse race problem to skew-normally distributed horse performance


def skew_state_price_implied_ability(prices: [float], unit:float=ALT_UNIT, L:int=ALT_L,
                                     a:float = ALT_A, scale:float=ALT_SCALE) -> [float]:
    """ Calibrate 'win probabilities' to a skewed performance distribution

            prices:    state prices (very similar to probabilities of winning)
            unit:                The lattice width
            L:                   The lattice is length 2*L+1
            scale:               The skew-normal scale parameter
            a:                   The skew-normal skew parameter
            returns:   implied ratings corresponding to location parameters (i.e. relative translations of perf distribution)

    """
    assert scale>0, 'scale parameter should be positive'
    assert unit>0,  'unit parameter should be positive'
    density = skew_normal_density(L=L, unit=unit, loc=0., scale=scale, a=a)
    return state_price_implied_ability(prices=prices, density=density, unit=unit)


def skew_dividend_implied_ability(dividends, unit:float=ALT_UNIT, L:int=ALT_L,
                                     a:float = ALT_A, loc=0, scale:float=ALT_SCALE) -> [float]:
    """ Implied ability using skewed normal performances, from inverse win probabilities """
    density = skew_normal_density(L=L,a=a,loc=loc,scale=scale, unit=unit)
    return dividend_implied_ability(dividends=dividends, density=density, unit=unit)


def skew_ability_implied_state_prices(ability, unit:float=ALT_UNIT, L:int=ALT_L,
                                     a:float = ALT_A, scale:float=ALT_SCALE):
    """ Inverse probabilities based on a canonical choice of skew normal density """
    density = skew_normal_density(L=L, a=a, loc=0, scale=scale, unit=unit)
    return ability_implied_state_prices(ability, density=density, unit=unit)


def skew_ability_implied_dividends(ability, unit:float=ALT_UNIT, L:int=ALT_L,
                                     a:float = ALT_A, scale:float=ALT_SCALE):
    """ Inverse probabilities based on a canonical choice of skew normal density """
    density = skew_normal_density(L=L, a=a, loc=0, scale=scale, unit=unit)
    return ability_implied_dividends(ability, density=density, unit=unit)

