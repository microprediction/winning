
A fast numerical algorithm for inferring relative ability from multi-entrant contest winning probabilities. 

The paper is published in SIAM Journal on Quantitative Finance ([draft](https://github.com/microprediction/winning/blob/main/docs/Horse_Race_Problem__SIAM_updated.pdf)
 

### Usage

We choose a performance density

    density = centered_std_density()

We set 'dividends', a.k.a. 'decimal prices', almost the same as inverse probability

    dividends = [2,6,np.nan, 3]

The algorithm implies relative ability (i.e. how much to translate the performance distributions)
Horses with no bid are assigned odds of 1999:1 ... or you can leave them out.

    abilities = dividend_implied_ability(dividends=dividends,density=density, nan_value=2000)

### Generality

The density is just a vector. So any 'numerical' performance distribution can be used. 

### Plotting. 

See 

    L = 600
    unit = 0.01
    density = centered_std_density(L=L, unit=unit)
    dividends = [2,6,np.nan, 3]
    abilities = dividend_implied_ability(dividends=dividends,density=density, nan_value=2000, unit=unit)
    densities = [skew_normal_density(L=L, unit=unit, loc=a, a=0, scale=1.0) for a in abilities]
    legend = [ str(d) for d in dividends ]
    densitiesPlot(densities=densities, unit=unit, legend=legend)
    plt.show()
    
  

    ![](https://i.imgur.com/tYsrAWY.png){width="200"}

### Pricing show and place from win prices:

    from winning.lattice_pricing import skew_normal_simulation
    from pprint import pprint
    dividends = [2.0,3.0,12.0,12.0,24.0,24.0]
    pricing = skew_normal_simulation(dividends=dividends,longshot_expon=1.15,skew_parameter=1.0,nSamples=1000)
    pprint(pricing)

### Practical use

See the  ([paper](https://github.com/microprediction/winning/blob/main/docs/Horse_Race_Problem__SIAM_.pdf)) for why this is useful in lots of places.

![](https://i.imgur.com/tYsrAWY.png "Implied performance distributions")


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

If you're reading the code...

- State prices. The expectation of an investment that has a payoff equal to 1 if there is only one winner, 1/2 if two are tied, 1/3 if three are tied and so forth. State prices are synomymous with winning probability, except for dead heats. However in the code a lattice is used so dead-heats must be accomodated and the distinction is important. 

- Relative ability refers to how much one performance distribution needs to be 
translated in order to match another. Implied abilities are vectors of relative abilities consistent with a collection of state prices.

- Dividends are the inverse of state prices. This is Australian tote vernacular. Dividends are 'decimal odds'. A dividend of 9.0 corresponds to a state price of 1/9.0, and a bookmaker quote of 8/1. Don't ask me to translate to American odds conventions because they are so utterly ridiculous!      


### Special cases

The core algorithm is entirely ambivalent to the choice of performance distribution, and that certainly need not correspond to some analytic distribution with known properties. However, to make things convenient, there is some sugar provided:

- std_calibration module. 
- skew_calibration module.  

See the examples_basic for a gentle introduction. 
