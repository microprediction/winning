
A fast numerical algorithm for inferring relative ability from multi-entrant contest winning probabilities. 

The paper is published in SIAM Journal on Quantitative Finance ([draft](https://github.com/microprediction/winning/blob/main/docs/Horse_Race_Problem__SIAM_updated.pdf)
 

### Usage

To use a default skew-normal performance distribution:

    from winning.skew_calibration import skew_dividend_implied_ability
    dividends = [2.0, 3.0, 6.0] 
    ability  = skew_dividend_implied_ability(dividends=dividends)
    prices   = skew_ability_implied_dividends(ability)

Alternatively see winning.lattice_calibration and use functions such as state_price_implied_ability(prices, density) which allow
you to specify whatever performance distribution you like. 

### Pricing show and place from win prices:

    from winning.lattice_pricing import skew_normal_simulation
    from pprint import pprint
    dividends = [2.0,3.0,12.0,12.0,24.0,24.0]
    pricing = skew_normal_simulation(dividends=dividends,longshot_expon=1.15,skew_parameter=1.0,nSamples=1000)
    pprint(pricing)

### Practical use

See the  ([paper](https://github.com/microprediction/winning/blob/main/docs/Horse_Race_Problem__SIAM_.pdf)) for why this is useful in lots of places.


![](https://i.imgur.com/83iFzel.png)


### Cite

    
        @article{doi:10.1137/19M1276261,
        author = {Cotton, Peter},
        title = {Inferring Relative Ability from Winning Probability in Multientrant Contests},
        journal = {SIAM Journal on Financial Mathematics},
        volume = {12},
        number = {1},
        pages = {295-317},
        year = {2021},
        doi = {10.1137/19M1276261},
        URL = { 
                https://doi.org/10.1137/19M1276261
        },
        eprint = { 
                https://doi.org/10.1137/19M1276261
        }
        }

### Overview 

The lattice_calibration module allows the user to infer relative abilities from state prices in a multi-entrant contest. The assumption
made is that the performance distribution of one competitor is a translation of the performance distribution of another. 

At the racetrack, this would mean looking at the win odds and infering a relative ability of horses. The algorithm is:

- Fast 

- Scalable (to contests with hundreds of thousands of entrants)

- General (it works for any performance distribution). 


### Nomenclature 

The algorithm takes state prices as inputs. These are for practical purposes equivalent to winning probabiliites (as the lattice size grows and ties are less common).

- State prices. The expectation of an investment that has a payoff equal to 1 if there is only one winner, 1/2 if two are tied, 1/3 if three are tied and so forth. 

- Relative ability refers to how much one performance distribution needs to be 
translated in order to match another. 

- Implied abilities are vectors of relative abilities consistent with a collection of state prices.

- Dividends are the inverse of state prices.   


### Special cases

Two natural choices are:

- Standard normal, as per normal_calibration module. 

- Skew-normal, as per skew_calibration module.  
