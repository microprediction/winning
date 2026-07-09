from winning.lattice import skew_normal_density, densities_from_offsets, pdf_to_cdf, sample_from_cdf
from winning.lattice_calibration import dividend_implied_ability, prices_from_dividends
import numpy as np
from collections import Counter
import heapq
from pprint import pprint
try:
    import pandas as pd
except ImportError:
    raise('pip install pandas')

# An example of pricing exotics by Monte Carlo

def _normalize(ps):
    s = sum(ps)
    if s>1e-6:
        return [ p/s for p in ps ]
    else:
        return [ 0.0 for p in ps ]


def placegetter(scores, position):
    return heapq.nsmallest(position+1, range(len(scores)), key=scores.__getitem__)[position]

def sample_from_cdf_with_noise(cdf, nSamples):
    # Break ties
    samples = sample_from_cdf(cdf=cdf, n_samples=nSamples)
    noise = 0.00001*np.random.randn(nSamples)
    return [ s+x for s,x in zip(samples,noise) ]

def sample_exotics( dividends, density, nSamples = 5000 ):
    """ Return counts of ordered results """

    offsets = dividend_implied_ability(dividends=dividends, density=density)
    densities = densities_from_offsets(density=density,offsets=offsets)
    cdfs = [pdf_to_cdf(density) for density in densities]
    cols = [sample_from_cdf_with_noise(cdf, nSamples) for cdf in cdfs]

    rows = list( map( list, zip( *cols )))
    winner   = [ placegetter(row,0) for row in rows ]
    second   = [ placegetter(row,1) for row in rows ]
    third    = [ placegetter(row,2) for row in rows ]
    return exotic_count(winner=winner, second=second, third=third)


def exotic_count(winner, second, third):
    win = Counter(winner)
    place = Counter(second)
    place.update(win)
    show = Counter(third)
    show.update(place)
    exacta = Counter(zip(winner, second))
    trifecta = Counter(zip(winner, second, third))
    return {"win":win, "exacta":exacta, "trifecta":trifecta, "place":place, "show":show}

def bookmaker_ratios( dividends, density, nSamples=5000 ):
    """
       Comparision to the rule of 1/4
    """
    probabilities = prices_from_dividends(dividends)
    n = len(probabilities)
    monte_carlo = sample_exotics( dividends=dividends, density=density, nSamples=nSamples )
    win  = monte_carlo['win']
    show = monte_carlo["show"]

    nTotal = nSamples
    while True:
        monte_carlo_ = sample_exotics(dividends=dividends, density=density, nSamples=nSamples)
        win_  = monte_carlo_['win']
        show_ = monte_carlo_["show"]
        win.update(win_)
        show.update(show_)
        nTotal += nSamples
        rows = list()
        for k in range(n):
            p = probabilities[k]
            b = 1/p-1              # Bookmaker quoted odds
            b_show  = b/4          # Bookmaker quoted show odds
            p_show  = 1/(b_show+1) # Bookmaker show probability using rule of 1/4
            b_ratio = p_show/p     # Ratio of show probability to win probability
            p_show_model = show[k]/nTotal
            increase = round(100*(p_show_model/p_show-1),1)
            row_data = [ round(x,3) for x in (b,b_show,p_show,p_show_model,increase)]
            rows.append(row_data)

        df = pd.DataFrame.from_records(data=rows,columns=['Win','Show','Bookmaker','Model','Model ratio'])
        df.to_csv('rule_of_a_quarter.csv')
        pprint(df)
    return {"show":df}


def exotic_ratios( dividends, density, nSamples=5000 ):
    """
         By Monte Carlo, estimate difference in conditional second place probabilities versus axiom of choice
    """

    probabilities = prices_from_dividends(dividends)
    n = len(probabilities)
    monte_carlo = sample_exotics( dividends=dividends, density=density, nSamples=nSamples )
    win = monte_carlo['win']
    exacta = monte_carlo["exacta"]
    trifecta = monte_carlo["trifecta"]

    nTotal = nSamples
    while True:
        monte_carlo_ = sample_exotics(dividends=dividends, density=density, nSamples=nSamples)
        win_ = monte_carlo_['win']
        exacta_ = monte_carlo_["exacta"]
        trifecta_ = monte_carlo_["trifecta"]
        win.update(win_)
        exacta.update(exacta_)
        trifecta.update(trifecta_)
        nTotal += nSamples
        exacta_ratios = [[0.] * n for _ in range(n)]
        for ex in exacta:
            winner = ex[0]
            second = ex[1]
            p1 = probabilities[winner]
            p2 = probabilities[second]
            conditional_prob = (exacta[ex]/win[winner])
            harville_conditional_prob = p2/(1-p1)
            exacta_ratios[winner][second] = round(conditional_prob/harville_conditional_prob-1,3)
        pprint(exacta_ratios)
        np.savetxt("derby.csv", np.array(exacta_ratios), delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')
    return {"exacta":exacta_ratios}


####################################
#   Harville stuff as a benchmark  #
####################################

def harville_exacta(p1,p2):
    return p1*p2/(1-p1)

def harville_trifecta(p1,p2,p3):
    return p1*p2*p3/((1-p1)*(1-p2))

def harville_probabilities( dividends ):
    """
        Apply axiom of choice
    """

    probabilities = _normalize( [ 1/dividend if dividend>0 else 0.0 for dividend in dividends ])
    n = len(probabilities)
    exacta   = [ [0.]*n for _ in range(n) ]
    quinella = [ [0.]*n for _ in range(n) ]
    trifecta = [ [ [0.]*n for _ in range(n) ] for _ in range(n) ]

    win      = probabilities
    second   = [ 0. ]*n
    third    = [ 0. ]*n

    for k1,p1 in enumerate(probabilities):
        for k2,p2 in enumerate(probabilities):
            if k1 != k2:
                exacta[k1][k2]=harville_exacta(p1=p1,p2=p2)
                second[k2] += exacta[k1][k2]
                if k2>k2:
                    quinella[k1][k2] = harville_exacta(p1=p1,p2=p2)+harville_exacta(p1=p2,p2=p1)
            for k3,p3 in enumerate(probabilities):
                trifecta[k1][k2][k3] = harville_trifecta(p1=p1,p2=p2,p3=p3)
                third[k3] += trifecta[k1][k2][k3]

    show  = [ f+s+t for f,s,t in zip(win,second,third)]
    place = [ f+s  for f, s in zip(win, second)]

    return exacta, quinella, trifecta, win, place, show


DERBY = sorted( [150,6,66,10,30,55,33,50,80,125,15/2,100,25,80,40,125,28,66,100,150,40,100,20,10/13,20 ] )


if __name__=='__main__':
    from winning.lattice_conventions import STD_UNIT, STD_SCALE, STD_L, STD_A
    dividends = [ o+1.0 for o in DERBY ]
    bookmaker_ratios(nSamples=100, dividends=dividends, density=skew_normal_density(L=STD_L, unit=STD_UNIT, scale=STD_SCALE, a=2.0))