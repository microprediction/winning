from winning.classic.normaldist import normcdf, normpdf
import numpy as np
import math

from ..rustconfig import load_fastrace

# compiled kernels (rust/fastrace); honours WINNING_PURE and use_rust()
_fastrace, _RUST_OK, _HAVE_RUST = load_fastrace('classic_exact_state_prices')

#########################################################################################
#   Operations on univariate atomic distributions supported on evenly spaced points     #
#########################################################################################

# A density is just a list of numbers interpreted as a density on the integers
# Where a density is to be interpreted on a lattice with some other unit, the unit parameter is supplied and
# this is equal to the spacing between lattice points. However, most operations are implemented on the
# lattice that is the natural numbers.




def integer_shift(cdf, k):
    """ Shift cdf to the *right* so it represents the cdf for Y ~ X + k*unit
    :param cdf:
    :param k:     int    Number of lattice points
    :return:
    """
    if k < 0:
        return np.append(cdf[abs(k):], cdf[-1] * np.ones(abs(k)))
    elif k == 0:
        return cdf
    else:
        # Mass shifted past the top atom lumps on it, mirroring the
        # negative branch (which lumps the bottom). Truncating left the
        # CDF ending below its total, a sub-probability law (#373).
        out = np.append(np.zeros(k), cdf[:-k])
        out[-1] = cdf[-1]
        return out


def fractional_shift(cdf, x):
    """ Shift cdf to the *right* so it represents the cdf for Y ~ X + x*unit
    :param cdf:
    :param x:     float    Number of lattice points to shift (need not be integer)
    :return:
    """
    L = implied_L(cdf)
    (l, lc), (u, uc) = _low_high(x, L=L)
    try:
        return lc * integer_shift(cdf, l) + uc * integer_shift(cdf, u)
    except:
        raise Exception('nasty bug')


def _low_high(offset, L):
    """ Represent a float offset as combination of two discrete ones """
    if -L+2 < offset < L-2:
        l = math.floor(offset)
        u = math.ceil(offset)
        r = offset - l
        return (l, 1 - r), (u, r)
    elif offset >= L-2:
        return (L-2,1), (L-1,0)
    elif offset <= -L+2:
        return (-L+1,0), (-L+2,1)


def density_from_samples(x: [float], L: int, unit=1.0):
    low_highs = [ _low_high(xi / unit, L=L) for xi in x]
    density = [0 for _ in range(2 * L + 1)]
    mass = 0
    for lh in low_highs:
        for (lc, wght) in lh:
            rel_loc = min(2 * L, max(lc + L, 0))
            mass += wght
            density[rel_loc] += wght
    total_mass = sum(density)
    return [d / total_mass for d in density]





def fractional_shift_density(density, x):
    """ Shift pdf to the *right* so it represents the pdf for Y ~ X + x*unit """
    cdf = pdf_to_cdf(density)
    shifted_cdf = fractional_shift(cdf, x)
    return cdf_to_pdf(shifted_cdf)


def center_density(density):
    """ Shift density to near its mean """
    m = mean_of_density(density, unit=1.0)
    return fractional_shift_density(density, -m)


### Interpretation of density on a grid with known spacing

def implied_L(density):
    return int((len(density) - 1) / 2)


def approximate_support(density, tol=1e-12):
    return [ i  for i in range(len(density)) if density[i]>tol ]


def approximate_support_width(density, tol=1e-12):
    supp = approximate_support(density=density, tol=tol)
    return max(supp)-min(supp)


def mean_of_density(density, unit):
    L = implied_L(density)
    pts = symmetric_lattice(L=L, unit=unit)
    return np.inner(density, pts)


def symmetric_lattice(L, unit):
    assert isinstance(L, int), "Expecting L to be integer"
    return unit * np.linspace(-L, L, 2 * L + 1)


def middle_of_density(density, L:int, do_padding=False):
    """
    :param density:
    :param L:
    :return: density of len 2*L+1
    """
    L0 = implied_L(density)
    if (L0==L) or ((L0<L) and not do_padding):
        return density
    elif L0<L:
        L0 = implied_L(density)
        n_extra=L-L0
        padding = [ 0 for _ in range(n_extra) ]
        return padding + list(density) + padding
    else:
        # Crop through the CDF: the mass below the window lands on its
        # first atom, the mass above it is dropped. This line used to
        # read cdf_to_pdf(density) -- a PDF differenced twice -- so the
        # "crop" was a signed second difference with mass ~0 and every
        # convolution that actually cropped raised "too much mass loss"
        # (#404).
        n_extra = L0-L
        cdf = pdf_to_cdf(density)
        cdf_truncated = cdf[n_extra:-n_extra]
        pdf_truncated = cdf_to_pdf(cdf_truncated)
        return pdf_truncated

def convolve_two(density1, density2, L=None, do_padding=False):
    """
       If X ~ density1 and Y ~ density2 are represented on symmetric lattices
       then the returned density will approximate X+Y, albeit not perfectly if
       the support of X or Y comes too close to the end of the lattice. Either way
       the mean will be preserved.

    :param density1:  2L+1
    :param density2:  Any odd length
    :return:

    Note that if, on the other hand, you wish to preserve densities exactly then you can simply
    use np.convolve

    """
    assert len(density1) % 2 ==1,  'Expecting odd length density1 '
    assert len(density2) % 2 == 1, 'Expecting odd length density2 '
    if L is None:
        L = implied_L(density1)

    mu1 = mean_of_density(density1, unit=1)
    mu2 = mean_of_density(density2, unit=1)
    density = np.convolve(density1, density2)
    middle = middle_of_density(density=density, L=L, do_padding=do_padding)
    mu = mean_of_density(middle, unit=1)
    mu_diff = mu-(mu1+mu2)
    pdf_shifted = fractional_shift_density(middle,-mu_diff)
    if sum(pdf_shifted)<0.9:
        raise ValueError('Convolution of two densities caused too much mass loss - increase L')
    return pdf_shifted


def convolve_many(densities, L=None, do_padding=True):
    """

    :param density[0] has length  2L+1
    :return:

    Note that if, on the other hand, you wish to preserve densities exactly then you can simply
    use np.convolve

    """
    densities = list(densities)
    if not densities:
        raise ValueError('convolve_many needs at least one density')
    for k,d in enumerate(densities):
        assert len(d) % 2 ==1,  'Expecting odd length density['+str(k)+']'

    mu_sum = sum( [ mean_of_density(density, unit=1) for density in densities ])
    if L is None:
        L = implied_L(densities[0])
    if len(densities) == 1:
        # The convolution of one law is that law. The loop below always
        # ended with convolve_two(full_density, densities[-1]), which for
        # a singleton is the density convolved WITH ITSELF: mass and mean
        # survive, the variance doubles (#405). Only the requested
        # crop/padding applies, and the mean is restored only if a crop
        # actually moved it.
        d0 = np.asarray(densities[0], dtype=float)
        out = np.asarray(middle_of_density(d0, L=L, do_padding=do_padding), dtype=float)
        if implied_L(d0) > L:
            mu_diff = mean_of_density(out, unit=1) - mu_sum
            out = fractional_shift_density(out, -mu_diff)
        return out
    full_density = [ p for p in densities[0] ]
    for density in densities[1:-1]:
        full_density = convolve_two(density1=full_density, density2=density, L=L, do_padding=False)
    full_density = convolve_two(density1=full_density, density2=densities[-1], L=L, do_padding=do_padding)

    mu = mean_of_density(full_density, unit=1)
    mu_diff = mu-mu_sum
    pdf_shifted = fractional_shift_density(full_density,-mu_diff)
    return pdf_shifted

#############################################
#   Simple family of skewed distributions   #
#############################################

# As noted above, densities are simply vectors. So if these utility functions are used
# to create densities with unit=0.02, say, then the user must keep the unit chosen for
# later interpretation.


def skew_normal_density(L, unit, loc=0, scale=1.0, a=2.0):
    """ Skew normal as a lattice density """
    lattice = symmetric_lattice(L=L, unit=unit)
    density = np.array([_unnormalized_skew_cdf(x, loc=loc, scale=scale, a=a) for x in lattice])
    density = density / np.sum(density)
    density = center_density(density)
    density = fractional_shift(density, loc/unit )
    return density


def _unnormalized_skew_cdf(x, loc=0, scale=1.0, a=2.0):
    """ Proportional to skew-normal density
    :param x:
    :param loc:    location
    :param scale:  scale
    :param a:      controls skew (a>0 means fat tail on right)
    :return:  np.array length 2*L+1
    """
    t = (x - loc) / scale
    return 2 / scale * normpdf(t) * normcdf(a * t)


def sample_from_cdf(cdf, n_samples, unit=1.0, add_noise=False):
    """ Monte Carlo sample """
    rvs = np.random.rand(n_samples)
    performances = [unit * sum([rv > c for c in cdf]) for rv in rvs]
    if add_noise:
        noise = 0.00001 * unit * np.random.randn(n_samples)
        return [s + x for s, x in zip(performances, noise)]
    else:
        return performances


def sample_from_cdf_with_noise(cdf, n_samples, unit=1.0):
    return sample_from_cdf(cdf=cdf, n_samples=n_samples, unit=unit, add_noise=True)


#############################################
#  Order statistics on lattices             #
#############################################
# -------  Nothing below here depends on the unit chosen -------------




def pdf_to_cdf(density):
    """  Prob( X <= k )    """
    # If pdf's were created with unit in mind, this is really Prob( X<=k*unit )
    return np.cumsum(density)


def cdf_to_pdf(cumulative):
    """ Given cumulative distribution on lattice, return the pdf """
    prepended = np.insert(cumulative, 0, 0.)
    return np.diff(prepended)


def winner_of_many(densities, multiplicities=None):
    """ The PDF of the minimum of the random variables represented by densities
    See https://medium.com/@mike.roweprediger/density-of-the-minimum-of-n-random-variables-on-a-lattice-45583cc0c2d0
    See https://github.com/microprediction/winning/blob/main/density_of_minimum_of_many.ipynb

    :param   densities:  [ np.array   ]
    :return: np.array
    """
    if len(densities) == 0:
        raise ValueError('winner_of_many needs at least one density')
    d = densities[0]
    multiplicities = multiplicities or [None for _ in densities]
    m = multiplicities[0]
    if m is None:
        # A lone entrant is its own winner with multiplicity exactly one.
        # This stayed None when the fold below never ran, and the None
        # reached the multiplicity arithmetic of state_prices_from_densities
        # as a TypeError; R and Rust start the fold at ones (#406).
        m = np.ones(len(d))
    for d2, m2 in zip(densities[1:], multiplicities[1:]):
        d, m = _winner_of_two_pdf(d, d2, multiplicityA=m, multiplicityB=m2)
    return d, m


def sample_winner_of_many(densities, nSamples=5000):
    """ The PDF of the minimum of the integer random variables represented by densities, by Monte Carlo """
    cdfs = [pdf_to_cdf(density) for density in densities]
    cols = [sample_from_cdf(cdf, nSamples) for cdf in cdfs]
    rows = map(list, zip(*cols))
    D = [int(min(row)) for row in rows]
    density = np.bincount(D, minlength=len(densities[0])) / (1.0 * nSamples)
    return density


def get_the_rest(density, densityAll, multiplicityAll, cdf=None, cdfAll=None):
    """ Returns expected _conditional_payoff_against_rest broken down by score,
       where _conditional_payoff_against_rest is:
                     1 if we are better than rest (lower)
                     and 1/(1+multiplicity) if we are equal
    """
    # Use np.sum( expected_payoff ) for the expectation
    if cdf is None:
        cdf = pdf_to_cdf(density)
    if cdfAll is None:
        cdfAll = pdf_to_cdf(densityAll)
    if density is None:
        density = cdf_to_pdf(cdf)
    if densityAll is None:  # Why do we need this??
        densityAll = cdf_to_pdf(cdfAll)

    S = 1 - cdfAll
    S1 = 1 - cdf
    Srest = (S + 1e-18) / (S1 + 1e-6)
    cdfRest = 1 - Srest

    # Multiplicity inversion (uses notation from blog post)
    # This is written up in my blog post and the paper
    m = multiplicityAll
    f1 = density
    m1 = 1.0
    fRest = cdf_to_pdf(cdfRest)
    numer = m * f1 * Srest + m * (f1 + S1) * fRest - m1 * f1 * (Srest + fRest)
    denom = fRest * (f1 + S1)
    multiplicityLeftTail = (1e-18 + numer) / (1e-18 + denom)
    multiplicityRest = multiplicityLeftTail

    T1 = (S1 + 1.0e-18) / (
            f1 + 1e-6)  # This calculation is more stable on the right tail. It should tend to zero eventually
    Trest = (Srest + 1e-18) / (fRest + 1e-6)
    multiplicityRightTail = m * Trest / (1 + T1) + m - m1 * (1 + Trest) / (1 + T1)

    k = list(f1 == max(f1)).index(True)
    multiplicityRest[k:] = multiplicityRightTail[k:]
    
    # Force positivity
    cdfRestMax = np.maximum.accumulate(cdfRest)

    return cdfRestMax, multiplicityRest


def expected_payoff(density, densityAll, multiplicityAll, cdf=None, cdfAll=None):
    cdfRest, multiplicityRest = get_the_rest(density=density, densityAll=densityAll, multiplicityAll=multiplicityAll, cdf=cdf, cdfAll=cdfAll)
    return _conditional_payoff_against_rest(density=density, densityRest=None, multiplicityRest=multiplicityRest, cdf=cdf, cdfRest=cdfRest)


def _winner_of_two_pdf(densityA, densityB, multiplicityA=None, multiplicityB=None, cdfA=None, cdfB=None):
    """ The PDF of the minimum of two random variables represented by densities
    
    See https://medium.com/@mike.roweprediger/probability-of-winning-a-two-horse-race-given-discrete-densities-a0b3fdb50094
    
    :param   densityA:   np.array
    :param   densityB:   np.array
    :return: density, multiplicity
    """
    if cdfA is None:
        cdfA = pdf_to_cdf(densityA)
    if cdfB is None:
        cdfB = pdf_to_cdf(densityB)
    cdfMin = 1 - np.multiply(1 - cdfA, 1 - cdfB)
    density = cdf_to_pdf(cdfMin)
    L = implied_L(density)
    if multiplicityA is None:
        multiplicityA = np.ones(2 * L + 1)
    if multiplicityB is None:
        multiplicityB = np.ones(2 * L + 1)

    winA, draw, winB = _conditional_win_draw_loss(densityA, densityB, cdfA, cdfB)
    try:
        multiplicity = (winA * multiplicityA + draw * (multiplicityA + multiplicityB) + winB * multiplicityB + 1e-18) / (
                winA + draw + winB + 1e-18)
    except ValueError:
        raise Exception('hit nasty bug')
        pass
    return density, multiplicity


def _loser_of_two_pdf(densityA, densityB):
    reverse_density, reverse_multiplicity = _winner_of_two_pdf(np.flip(densityA), np.flip(densityB))
    return np.flip(reverse_density), np.flip(reverse_multiplicity)


def beats(densityA, multiplicityA, densityB, multiplicityB):
    """
        Returns expected _conditional_payoff_against_rest broken down by score
    """
    # use np.sum( _conditional_payoff_against_rest) for the expectation
    cdfA = pdf_to_cdf(densityA)
    cdfB = pdf_to_cdf(densityB)
    win, draw, loss = _conditional_win_draw_loss(densityA, densityB, cdfA, cdfB)
    return sum( win + draw * (1+multiplicityA) / (2 + multiplicityB + multiplicityA ) )



def state_prices_from_densities(densities:[[float]], densityAll=None, multiplicityAll=None)->[float]:
    """
      :param densities: List of performance distributions
      :return: state prices
    """
    # Exact dead heats (see exact_state_prices_from_cdfs). densityAll and
    # multiplicityAll are accepted for compatibility and no longer used:
    # the minimum's density and mean multiplicity do not determine the
    # prices (#418, #362).
    if len(densities) == 0:
        raise ValueError('a race needs at least one runner')
    prices = exact_state_prices_from_cdfs([pdf_to_cdf(d) for d in densities])
    sum_p = sum(prices)
    return [pi / sum_p for pi in prices]


def symmetric_state_prices_from_densities(densities:[[float]], densityAll=None, multiplicityAll=None, with_all=False)->[[float]]:
    if (densityAll is None) or (multiplicityAll is None):
        densityAll, multiplicityAll = winner_of_many(densities, multiplicities=None)
    n = len(densities)
    bi = np.ndarray(shape=(n, n))
    for h0 in range(n):
        density0 = densities[h0]
        cdfRest0, multiplicityRest0 = get_the_rest(density=density0, densityAll=densityAll,
                                                   multiplicityAll=multiplicityAll, cdf=None, cdfAll=None)
        for h1 in range(n):
            if h1 > h0:
                density1 = densities[h1]
                cdfRest01, multiplicityRest01 = get_the_rest(density=density1, densityAll=None,
                                                             multiplicityAll=multiplicityRest0, cdf=None,
                                                             cdfAll=cdfRest0)
                pdfRest01 = cdf_to_pdf(cdfRest01)
                loser01, loser_multiplicity01 = _loser_of_two_pdf(density0, density1)
                bi[h0, h1] = beats(loser01, loser_multiplicity01, pdfRest01, multiplicityRest01)
                bi[h1, h0] = bi[h0, h1]
    if with_all:
        return bi, densityAll, multiplicityAll
    else:
        return bi


def two_prices_from_densities(densities:[[float]], densityAll=None, multiplicityAll=None, with_all=False)->[[float]]:
    q, densityAll, multiplicityAll = symmetric_state_prices_from_densities(densities=densities, densityAll=densityAll, multiplicityAll=multiplicityAll, with_all=True)
    w = state_prices_from_densities(densities=densities, densityAll=densityAll, multiplicityAll=multiplicityAll)
    pl = [0 for _ in w]
    n = len(w)
    for i in range(n):
        for j in range(i+1,n):
            pl[i] += q[i,j]
            pl[j] += q[i,j]
    sum_pl = sum(pl)
    pl = [ 2.0*pli/sum_pl for pli in pl]
    assert abs(sum(pl)-2.0)<0.01
    s = [ bi-wi for bi, wi in zip(pl,w)]
    if with_all:
        return w, s, densityAll, multiplicityAll
    else:
        return w , s


def five_prices_from_five_densities(densities:[[float]])->[[float]]:
    """
        Rank probabilities for five independent contestants
        returns [ [first probs], ..., [fifth probs] ]
    """
    assert(len(densities)==5)
    rdensities = [ np.asarray([ p for p in reversed(d)]) for d in densities ]
    w1, w2 = two_prices_from_densities(densities=densities, with_all=False)
    w5, w4 = two_prices_from_densities(densities=rdensities, with_all=False)
    w3 = [1.0-w1i-w2i-w4i-w5i for w1i,w2i,w4i,w5i in zip(w1,w2,w4,w5) ]
    return [ w1, w2, w3, w4, w5 ]


def densities_from_events(scores:[int], events:[[float]], L:int, unit:float):
    """
    :param scores:  List of current scores
    :param events:  List of densities of future events
    :param L:
    :param unit:
    :return:
    """
    assert len(scores)==len(events)
    low_score = min(scores)
    adjusted_scores = [ s-low_score for s in scores ]
    if True:
        for k,adj_score in enumerate(adjusted_scores):
            adapted_L = int(math.ceil(abs(adj_score/unit)))
            events[k].append( density_from_samples(x=[adj_score],L=adapted_L, unit=unit) )
    densities = [ convolve_many( densities=event, L=L ) for event in events ]
    return densities


def state_prices_from_events(scores:[int], events:[[float]], L:int, unit:float):
    densities = densities_from_events(scores=scores, events=events, L=L, unit=unit )
    return state_prices_from_densities(densities)



def _conditional_win_draw_loss(densityA, densityB, cdfA, cdfB):
    """ Conditional win, draw and loss probability lattices for a two ratings race """
    win = densityA * (1 - cdfB)
    draw = densityA * densityB
    lose = densityB * (1 - cdfA)
    return win, draw, lose


def _conditional_payoff_against_rest(density, densityRest, multiplicityRest, cdf=None, cdfRest=None):
    """ Returns expected _conditional_payoff_against_rest broken down by score, where _conditional_payoff_against_rest is 1 if we are better than rest (lower) and 1/(1+multiplicity) if we are equal """
    # use np.sum( _conditional_payoff_against_rest) for the expectation
    if cdf is None:
        cdf = pdf_to_cdf(density)
    if cdfRest is None:
        cdfRest = pdf_to_cdf(densityRest)
    if density is None:
        density = cdf_to_pdf(cdf)
    if densityRest is None:
        densityRest = cdf_to_pdf(cdfRest)

    win, draw, loss = _conditional_win_draw_loss(density, densityRest, cdf, cdfRest)
    return win + draw / (1 + multiplicityRest)


def densities_and_coefs_from_offsets(density, offsets):
    """ Given a density and a list of offsets (which might be non-integer) this
        returns a list of translated densities

    :param density:  np.ndarray
    :param offsets: [ float ]
    :return: [ np.ndarray ]
    """
    cdf = pdf_to_cdf(density)
    L = implied_L(cdf)
    coefs = [_low_high(offset, L=L) for offset in offsets]
    try:
        cdfs = [lc * integer_shift(cdf, l) + uc * integer_shift(cdf, u) for (l, lc), (u, uc) in coefs]
    except ValueError as e:
        print(e)
        print('')
        raise NotImplementedError('fix this peter')
    pdfs = [cdf_to_pdf(cdf) for cdf in cdfs]
    return pdfs, coefs


def densities_from_offsets(density, offsets):
    return densities_and_coefs_from_offsets(density, offsets)[0]


def dilate_density(density, unit_ratio=2):
    """ Represent density on a new lattice with a larger unit size
        Pretty crude
    See https://github.com/microprediction/winning/blob/main/dilation.ipynb
    Or see  https://medium.com/@mike.roweprediger/how-to-move-a-discrete-density-from-one-unit-size-to-another-27d4ffeab036

    :param density: 
    :param L: 
    :param unit_ratio:  e.g. if 2 the new density will be skinnier 
                        e.g. if 0.5, the new density will be fatter 
    :return: 
    """
    L = implied_L(density)
    x = list(range(-L,L+1))  
    low_highs = [ _low_high(xi / unit_ratio, L=L) for xi in x]
    dilated_density = [0 for _ in range(2 * L + 1)]
    mass = 0
    for lh, p in zip(low_highs, density):
        for (lc, wght) in lh:
            rel_loc = min(2 * L, max(lc + L, 0))
            mass += p*wght
            dilated_density[rel_loc] += p*wght
    total_mass = sum(dilated_density)
    return [d / total_mass for d in dilated_density]


def _state_prices_from_clustered_offsets(density, offsets, fast_ndxs:[int], unit_ratio, max_depth):
    """
         Helper to deal with lattice limitations
         This splits the race into two groups and then
         uses a dilated density to estimate the winning
         share of the lessor group

    :param density: 
    :param offsets: 
    :param unit_ratio: 
    :param fast_ndxs:    Indexes of fast horses 
    :return: 
    """
    # Hold an approximate race involving everyone
    dilated_density = dilate_density(density=density, unit_ratio=unit_ratio)
    dilated_offsets = [o / unit_ratio for o in offsets]
    dilated_state_prices = state_prices_from_extended_offsets(density=dilated_density, offsets=dilated_offsets,
                                                              max_depth=max_depth - 1)
    n = len(offsets)
    if (len(fast_ndxs)==n) or (len(fast_ndxs)==0) or (max_depth <= 0):
        return dilated_state_prices

    # Otherwise ...
    slow_ndxs = [ j for j in range(n) if j not in fast_ndxs ]
    assert slow_ndxs
    if len(slow_ndxs) == 1:
        slow_relative_state_prices = [1.0]
    else:
        # Compute the slown guys at low resolution to reduce recursions
        dilated_slow_offsets = [dilated_offsets[i] for i in slow_ndxs]
        slow_relative_state_prices = state_prices_from_extended_offsets(density=dilated_density,
                                                                        offsets=int_centered(dilated_slow_offsets),
                                                                        max_depth=max_depth - 2)
    # Race the fast horses
    fast_ndxs = [j for j in range(n) if j not in slow_ndxs]
    fast_offsets = [offsets[j] for j in fast_ndxs]
    fast_relative_state_prices = state_prices_from_extended_offsets(density=density,
                                                                    offsets=int_centered(fast_offsets),
                                                                    max_depth=max_depth - 1)

    # It remains to combine the two results in a plausible mannner
    slow_share = sum([p for i, p in enumerate(dilated_state_prices) if i in slow_ndxs])
    fast_share = 1 - slow_share
    slow_prices = [p * slow_share for p in slow_relative_state_prices]
    fast_prices = [p * fast_share for p in fast_relative_state_prices]
    state_prices = [0 for _ in range(n)]
    for i, ndx in enumerate(slow_ndxs):
        state_prices[ndx] = slow_prices[i]
    for i, ndx in enumerate(fast_ndxs):
        state_prices[ndx] = fast_prices[i]
    state_sum = sum(state_prices)
    assert state_sum>0.99,'Surprising fail! l606 lattice.py'
    state_prices = [ si/state_sum for si in state_prices]
    return state_prices


def mean_ignoring_inf(values):
    return np.nanmean([e for e in values if np.isfinite(e)])


def int_centered(offsets):
    int_mean = int(mean_ignoring_inf(offsets))
    return [o - int_mean for o in offsets]


def divide_offsets(centered_offsets, max_best=20):
    n = len(centered_offsets)
    if len(centered_offsets) == 2:
        offset_divider = np.mean(centered_offsets)  # should be 0
    else:
        srt_offsets = sorted(centered_offsets)
        max_best = min(20, int(n/6+2))
        gaps = [ abs(a) for a in np.diff([srt_offsets[0]]+srt_offsets) ][:max_best+1] # [0, 4, 2, ...]
        ndx_gap = max(1,gaps.index(max(gaps)))
        try:
            offset_divider = ( srt_offsets[ndx_gap-1] + srt_offsets[ndx_gap] ) / 2.0
        except IndexError:
            offset_divider = 0
    return offset_divider


def state_prices_from_extended_offsets(density, offsets, max_depth=3, unit_ratio=3):
    """ Imply state prices but allow the offsets to be float('inf') or float('-inf')
    :param density:
    :param offsets:
    :param max_depth: specifies the max number of times to call recursively, not counting
                            the recusion calls that merely get rid of float('inf') or float('-inf')
    :return:
    """
    density = as_classic_density(density)
    # First get rid of float('inf')
    n = len(offsets)
    if n==1:
        return [1.0]

    infinite_ndxs = [i for i in range(n) if offsets[i] == float('inf')]
    if infinite_ndxs:
        finite_ndxs = [ j for j in range(n) if j not in infinite_ndxs ]
        finite_offsets = [ offsets[j] for j in finite_ndxs ]
        if not finite_offsets:
            # Everyone is float('inf')
            return [ 1.0/n for _ in range(n) ]
        else:
            finite_state_prices = state_prices_from_extended_offsets(density=density,
                                                                     offsets=int_centered(finite_offsets),
                                                                     max_depth=max_depth)
            state_prices = [ 0 for _ in range(n)]
            for j, ndx in enumerate(finite_ndxs):
                state_prices[ndx] = finite_state_prices[j]
            return state_prices

    # Then get rid of float('-inf')
    neg_infinite_ndxs = [i for i in range(n) if offsets[i] == float('-inf')]
    if neg_infinite_ndxs:
        # Split pot amongst those who are float('-inf')
        n_pot = len(neg_infinite_ndxs)
        state_prices = [0 for _ in range(n)]
        for j, ndx in enumerate(neg_infinite_ndxs):
            state_prices[ndx] = 1/n_pot
        assert abs(sum(state_prices)-1)<1e-12
        return state_prices

    # Otherwise, having reached this far we have only float abilities, though some might be extremely large
    L = implied_L(density)
    W = int(approximate_support_width(density))
    really_bad_horse_offset = min(offsets) + W # <--- No chance of winning

    # If there are bad but finite horses, set them to float('inf') and call again
    # On the next call, hopefully the centering leaves us inside the lattice
    extremely_bad_ndxs = [i for i, o in enumerate(offsets) if o > really_bad_horse_offset ]
    if any(extremely_bad_ndxs):
        augmented_offsets = [o if i not in extremely_bad_ndxs else float('inf') for i, o in enumerate(offsets)]
        return state_prices_from_extended_offsets(density=density,
                                                  offsets=int_centered(augmented_offsets),
                                                  max_depth=max_depth)

    # If there is a solitary standout horse, short-circuit
    diff_to_best = [ o - min(offsets) for o in offsets]
    is_walkover = min(diff_to_best) > W*math.sqrt(len(offsets))
    if is_walkover:
        winning_ndx = offsets.index(min(offsets))
        state_prices = [0 for _ in offsets]
        state_prices[winning_ndx] = 1.0
        return state_prices

    # At this point in the algorithm we have a race that isn't completely degenerate
    # Let's center and then see if any horses are hanging off the lattice.
    centered_offsets = int_centered(offsets)
    offset_lower_bound = -L + W
    offset_upper_bound = L - W
    hanging_left  = [ i for i,o in enumerate(centered_offsets) if o<offset_lower_bound  ]
    hanging_right = [ i for i,o in enumerate(centered_offsets) if o>offset_upper_bound  ]
    hanging = bool(hanging_right or hanging_left)
    if not hanging:
        # Should be good to go!
        return state_prices_from_offsets(density=density, offsets=centered_offsets)
    else:
        if max_depth==0:
            # If we've already tried pretty hard, just set hangers to inf or -inf and call one more time
            for ndx in hanging_right:
                centered_offsets[ndx]=float('inf')
            for ndx in hanging_left:
                centered_offsets[ndx] = float('-inf')
            return state_prices_from_extended_offsets(offsets=centered_offsets, density=density, max_depth=0)
        else:
            # Otherwise we can split the race into two clusters
            offset_divider = divide_offsets(centered_offsets)
            fast_ndxs = [ i for i,o in enumerate(centered_offsets) if o < offset_divider ]
            if (len(fast_ndxs)==n) or (len(fast_ndxs)==0):
                print('This probably should not happen as the divider should divide! , but its okay...line l717 of lattice.py')

            state_prices = _state_prices_from_clustered_offsets(density=density,
                                                                offsets=centered_offsets,
                                                                fast_ndxs=fast_ndxs,
                                                                unit_ratio=unit_ratio,
                                                                max_depth=max_depth - 1)
            return state_prices


def state_prices_from_offsets(density, offsets):
    """ Returns a list of state prices for a race where all horses have the same density
        up to a translation (the offsets)
    """
    # See the paper for a definition of state price
    # Be aware that this may fail if offsets provided are integers rather than float
    density = as_classic_density(density)
    if _HAVE_RUST:
        return list(_fastrace.classic_exact_state_prices(
            [float(d) for d in density], [float(o) for o in offsets]))
    _, cdfs, _ = _exact_offset_cdfs(density, offsets)
    return exact_state_prices_from_cdfs(cdfs)


def implicit_state_prices(density, densityAll, multiplicityAll=None, cdf=None, cdfAll=None, offsets=None):
    """ Returns the expected _conditional_payoff_against_rest as a function of location changes in cdf """

    L = implied_L(density)
    if cdf is None:
        cdf = pdf_to_cdf(density)
    if cdfAll is None:
        cdfAll = pdf_to_cdf(densityAll)
    if multiplicityAll is None:
        multiplicityAll = np.ones(2 * L + 1)
    if offsets is None:
        offsets = range(int(-L / 2), int(L / 2))
    implicit = list()
    for k in offsets:
        # The integer fast path must obey the SAME clamp the field was
        # built with. The field goes through _low_high, which pins an
        # offset at or past L-2 to the boundary; this called
        # integer_shift(cdf, k) with the raw k, whose own clamp is the
        # much wider +/-(m-1). So a runner could sit in the field at
        # offset L-2 and be PAID at 60, and the state prices stopped
        # being exhaustive claims on one race -- they summed to 1.88,
        # while an epsilon off the integer gave 1.0 because the
        # non-integer path was clamped correctly all along (#292).
        #
        # Guarding the fast path by the interior condition is exactly
        # equivalent where it applies: for an interior integer
        # _low_high returns (k, 1), (k, 0), which is this shift with
        # weight one.
        if k == int(k) and -L + 2 < k < L - 2:
            offset_cdf = integer_shift(cdf, int(k))
            ip = expected_payoff(density=None, densityAll=densityAll, multiplicityAll=multiplicityAll, cdf=offset_cdf,
                                 cdfAll=cdfAll)
            implicit.append(np.sum(ip))
        else:
            (l, l_coef), (r, r_coef) = _low_high(k, L=L)
            offset_cdf_left = integer_shift(cdf, l)
            offset_cdf_right = integer_shift(cdf, r)
            ip_left = expected_payoff(density=None, densityAll=densityAll, multiplicityAll=multiplicityAll,
                                      cdf=offset_cdf_left, cdfAll=cdfAll)
            ip_right = expected_payoff(density=None, densityAll=densityAll, multiplicityAll=multiplicityAll,
                                       cdf=offset_cdf_right, cdfAll=cdfAll)
            implicit.append(l_coef * np.sum(ip_left) + r_coef * np.sum(ip_right))

    return implicit


#########################################################################################
#   Exact dead-heat pricing (#418, #362, #348, #373)                                    #
#########################################################################################
#
# A runner's state price is the expected share of a unit winner claim,
# a dead heat split equally among the tied:
#
#     P_i = sum_t f_i(t) E[ 1{X_j >= t, all j != i} / (1 + M_t) ],
#
# M_t the number of OTHER runners exactly at t. Since 1/(1+M) is the
# integral over [0,1] of u^M, and the runners are independent,
#
#     P_i = sum_t f_i(t) int_0^1 prod_{j != i} ( S_j(t) + u f_j(t) ) du,
#
# with S_j(t) = P(X_j > t). The integrand is a polynomial of degree n-1
# in u, so Gauss-Legendre with n//2 + 1 nodes integrates it EXACTLY.
#
# The field is kept as G_q(t) = prod_j (S_j(t) + u_q f_j(t)) at those
# nodes, and a runner's opponents are G_q / (S_i + u_q f_i). That
# division is exact wherever it matters: where f_i(t) > 0 the divisor
# is at least u_q f_i(t) > 0, and where f_i(t) = 0 the term pays
# nothing. The engine it replaces kept only the minimum's CDF and the
# conditional MEAN multiplicity, which loses information two ways:
#
#   * dividing the minimum's survival by a runner's survival is 0/0
#     once that runner has surely finished, so an opponent's tail could
#     not be recovered: a two-runner compact-support race priced
#     [0.9167, 0.0500] instead of [0.95, 0.05] (#418);
#   * 1/(1 + E[M]) is not E[1/(1 + M)]: twenty iid three-atom entrants
#     were paid 0.04435 each, total 0.887 (#362).
#
# A fractional offset is the CDF mixture that _low_high defines, priced
# AS THAT MIXTURE: the old engine averaged the payoffs of its two
# integer components against a field that contained neither, and two
# identical runners at 0.5 summed to 1.114 (#348).
#
# The lattice is padded by L-1 atoms on each side before shifting, which
# is the largest shift _low_high can return, so a translated runner
# never loses mass off an edge. The old shift dropped whatever moved
# past the top: a singleton at +19 on a uniform 83-atom law was paid
# 0.771 (#373).
#
# (Node count: see Q_START below -- exact rule for small fields, a
# doubling check for big ones.)
#
# The inverse keeps the paper's fixed-point table: a candidate at each
# sample offset is priced against G_q / (S_k + u_q f_k). For a field
# member that is its exact price; for a candidate between members it is
# the same cavity approximation the paper uses, bounded by one and
# nonincreasing in t, which the exact cavity always is.


# The smallest lattice the offset API can represent: _low_high pins
# every offset to [-L+2, L-2], which for L <= 2 is the single point 0 --
# every runner priced as a tie whatever its offset -- and the inverse's
# default table range(-L//2, L//2) is empty at L = 1 (#339).
MIN_CLASSIC_L = 3


def as_classic_density(density, where='density'):
    """The one boundary for a classic atom vector (#339).

    Accepts a finite, nonnegative, odd-length (2L+1, L >= 3) vector of
    atoms with positive total and returns it normalised to unit mass, so
    raw histogram counts are the same law as their frequencies. Entries
    down to -1e-12 of the total are cdf/pdf round-off and are clipped.
    Anything else raises ValueError: mass 10 used to give prices of -61,
    a negative atom a finite tie, and an even length a silently
    truncated lattice.
    """
    d = np.asarray(density, dtype=float)
    if d.ndim != 1 or d.size == 0:
        raise ValueError(where + ' must be a nonempty 1-d vector of lattice atoms')
    if d.size % 2 != 1:
        raise ValueError(where + ' must have odd length 2L+1 on the symmetric lattice; got length '
                         + str(d.size))
    if (d.size - 1) // 2 < MIN_CLASSIC_L:
        raise ValueError(where + ' has L = ' + str((d.size - 1) // 2) + '; the classic lattice needs L >= '
                         + str(MIN_CLASSIC_L) + ' (length >= ' + str(2 * MIN_CLASSIC_L + 1)
                         + ') to represent distinct offsets')
    if not np.all(np.isfinite(d)):
        raise ValueError(where + ' has a non-finite atom')
    total = float(d.sum())
    if not total > 0:
        raise ValueError(where + ' has no positive mass')
    if d.min() < -1e-12 * total:
        raise ValueError(where + ' has a negative atom (' + repr(float(d.min())) + ')')
    return np.maximum(d, 0.0) / total


def as_classic_prices(prices, where='prices'):
    """Target state prices: finite, nonnegative, positive total, normalised.

    A categorical price vector carries only relative mass; the inverse
    used to read p and c*p off its table as different absolute
    ordinates, so a 10% overround moved relative abilities by 0.9 lattice
    units and a 10x book came back as an all-tie race (#377).
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(where + ' must be a nonempty 1-d vector')
    if not np.all(np.isfinite(p)):
        raise ValueError(where + ' has a non-finite entry')
    if p.min() < 0:
        raise ValueError(where + ' has a negative entry (' + repr(float(p.min())) + ')')
    total = float(p.sum())
    if not total > 0:
        raise ValueError(where + ' has no positive mass')
    return p / total


def _gauss_legendre01(n_nodes):
    """Gauss-Legendre nodes and weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    return 0.5 * (x + 1.0), 0.5 * w


def _n_nodes(n_runners):
    """Nodes that integrate a degree n-1 polynomial exactly."""
    return int(n_runners) // 2 + 1


# Exactness costs n//2 + 1 nodes, which makes a big field O(n^2 L): 3000
# runners would need 1501 nodes. The integrand prod_j (S_j + u f_j) is
# only of high degree where many runners carry an atom at once; on a
# smooth lattice it is numerically low-degree and 4 nodes already agree
# with 64 to 1e-17. So the forward map starts at Q_START nodes and
# doubles (capped at the exact count) until two successive rules agree
# on every price to EXACT_TOL, returning the larger rule; a field whose
# exact rule has at most Q_START nodes uses it directly. The inverse's table
# is a preconditioner -- the defect correction makes its fixed point the
# exact forward map whatever the table -- so it uses at most TABLE_NODES.
Q_START = 8
TABLE_NODES = 16
EXACT_TOL = 1e-14


def _node_schedule(n_runners):
    exact = _n_nodes(n_runners)
    q, out = min(Q_START, exact), []
    while True:
        out.append(q)
        if q >= exact:
            return out
        q = min(2 * q, exact)


def _padded_base_cdf(density):
    """CDF of `density` padded by L-1 zero atoms on each side."""
    L = implied_L(density)
    pad = max(L - 1, 0)
    d = np.concatenate([np.zeros(pad), np.asarray(density, dtype=float), np.zeros(pad)])
    return np.cumsum(d)


def _exact_shifted_cdf(padded_cdf, offset, L):
    """The (mixture) CDF _low_high assigns to `offset`, on the padded lattice."""
    (l, lc), (u, uc) = _low_high(offset, L=L)
    return lc * integer_shift(padded_cdf, l) + uc * integer_shift(padded_cdf, u)


def _survival_and_pdf(cdf):
    cdf = np.asarray(cdf, dtype=float)
    f = np.diff(cdf, prepend=0.0)
    S = np.maximum(1.0 - cdf, 0.0)
    return S, f


def _exact_field(cdfs, nodes):
    """G[q, t] = prod_j (S_j(t) + u_q f_j(t))."""
    G = np.ones((len(nodes), len(cdfs[0])))
    for c in cdfs:
        S, f = _survival_and_pdf(c)
        G *= S[None, :] + nodes[:, None] * f[None, :]
    return G


def _exact_payoff(cdf, G, nodes, weights):
    """Expected winner claim of a runner with CDF `cdf` against the field G,
    the runner itself divided out (see the block comment above)."""
    S, f = _survival_and_pdf(cdf)
    den = S[None, :] + nodes[:, None] * f[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(den > 0, G / np.where(den > 0, den, 1.0), 1.0)
    ratio = np.minimum.accumulate(np.minimum(ratio, 1.0), axis=1)
    return float(np.dot(weights, ratio @ f))


def exact_state_prices_from_cdfs(cdfs):
    """Exact state prices (dead heats split equally) of independent runners
    given their CDFs on one common lattice."""
    cdfs = [np.asarray(c, dtype=float) for c in cdfs]
    if not cdfs:
        raise ValueError('a race needs at least one runner')
    prev = None
    for q in _node_schedule(len(cdfs)):
        nodes, weights = _gauss_legendre01(q)
        G = _exact_field(cdfs, nodes)
        p = np.array([_exact_payoff(c, G, nodes, weights) for c in cdfs])
        if prev is not None and np.max(np.abs(p - prev)) <= EXACT_TOL:
            break
        prev = p
    return [float(x) for x in p]


def _exact_offset_cdfs(density, offsets):
    L = implied_L(density)
    base = _padded_base_cdf(density)
    return base, [_exact_shifted_cdf(base, o, L) for o in offsets], L


def _exact_implicit_prices(base, field_cdfs, offset_samples, L):
    """The paper's interpolation table, priced against the exact field."""
    nodes, weights = _gauss_legendre01(min(_n_nodes(len(field_cdfs)), TABLE_NODES))
    G = _exact_field(field_cdfs, nodes)
    return [_exact_payoff(_exact_shifted_cdf(base, k, L), G, nodes, weights)
            for k in offset_samples]
