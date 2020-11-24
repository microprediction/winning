from winning.std_calibration import std_dividend_implied_ability, std_density
from winning.lattice_calibration import ability_implied_dividends, normalize_dividends

def test_dividend_implied_ability():
    raw_dividends = [2.,3.,6.,10.,12.,20.,66.]
    dividends = normalize_dividends(raw_dividends)
    abilities = std_dividend_implied_ability(dividends=dividends)
    divs_back = ability_implied_dividends(abilities, density=std_density())
    assert all( [ abs(d1-d2)<1e-1 for d1,d2 in zip(divs_back,dividends)])
    abilities_back = std_dividend_implied_ability(divs_back)
    assert all( [abs(a1-a2)<1e-2 for a1,a2 in zip(abilities,abilities_back)] )
    divs_back_again = ability_implied_dividends(abilities_back, density=std_density())
    assert all([abs(d1 - d2) < 1e-1 for d1, d2 in zip(divs_back, divs_back_again)])
