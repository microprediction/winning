# Prior art: exact factor-correlated Thurstonian ranking likelihood
(Agent report, 2026-09-01. Verdict: (b) leaning (c) -- classical
under independence but treated there as something to APPROXIMATE;
the factor-conditioned chain as a rank-likelihood engine appears
novel AS A COMPOSITION. Frame it as a composition of known
ingredients, never a new identity. All statuses as labeled.)

## The foil, verbatim (read at source by the agent)
Johnson & Kuhn 2013, Behavior Research Methods 45:857-872,
doi:10.3758/s13428-012-0300-3: "Thurstonian ranking models define
the probability of an observed ranking as a (K-1)-dimensional
integral, which cannot be expressed in closed form" -- their remedy
is MCMC. This is the template quote for the short paper.

## Independence case: classical, and still approximated
- Henery 1981 (JRSS B 43:86-91, doi:10.1111/j.2517-6161.1981.tb01153.x)
  defines P(pi) for independent normals and proposes a TAYLOR
  APPROXIMATION around equal means, treating exact evaluation as
  impractical. [Paywalled; characterization corroborated via Ali and
  secondary sources -- verify before quoting.]
- Ali 1998 (J. Applied Statistics 25:221-229, read in full) computes
  win/place/show as single 1-D integrals for independent models --
  MARGINAL position probabilities only, never the full-permutation
  chain, never correlation.
- INID order statistics go through permanents (Vaughan-Venables
  1972; Glueck et al. 2008, PMC2768298). The nested-integral
  identity is implicit folklore; nobody writes the H_m grid
  recursion as a ranking-likelihood tool.

## Correlated case: nobody
- GHK is confirmed simulation (importance sampling; Ridgway
  arXiv:1411.1314 improves it with SMC -- still simulation).
- Yao-Bockenholt 1999 (doi:10.1348/000711099158973): Gibbs.
- Maydeu-Olivares 1999 (Psychometrika 64:325-340) and the 2002
  limited-information line: fit factor-structured Thurstonian
  models from FIRST/SECOND-ORDER MARGINALS precisely to avoid the
  high-dimensional integrals. Approximation by restriction.
- Dunnett-Sobel 1955 (Biometrika 42:258-260) and the product-
  correlation line reduce one-factor ORTHANT/RECTANGLE probabilities
  to 1-D outer integrals; ranking-and-selection (Bechhofer 1954)
  uses equicorrelation for SELECTION EVENTS. Nobody composes the
  factor conditioning with the sequential chain for PERMUTATIONS.
- NEAREST NEIGHBOR, GATE SUBSTANTIALLY CLEARED (abstract read
  verbatim 2026-09-01; full text paywalled): Ennis & Ennis 2013 (J.
  Classification 30:124-147, doi:10.1007/s00357-013-9125-8).
  Abstract: "rank-induced dependencies are specified through
  correlation coefficients among ranked objects that are determined
  by a vector of rank-induced parameters. The ranking model can be
  expressed in terms of univariate normal distribution functions...
  A theorem is proven that shows that the specification given in
  the paper for the dependencies is THE ONLY WAY that this
  simplification can be achieved under the process assumptions of
  the model." So: (a) their dependency structure is a purpose-built
  rank-induced correlation, NOT a factor model VV'+D; (b) their own
  uniqueness theorem says univariate reduction cannot be had more
  generally UNDER THEIR ASSUMPTIONS -- which strengthens our
  positioning: the factor-conditioning route sidesteps the theorem
  by paying one quadrature dimension for arbitrary factor structure.
  Cite prominently, quote the uniqueness sentence, and read the full
  text before final wording (their theorem's exact scope is the one
  remaining unknown). Note: distinct from D.M. Ennis's 2013 Journal
  of Sensory Studies papers (e.g. hedonic mapping) -- same author
  family, same year, different papers; do not conflate.

## Rankograms / scale
Bayesian NMA rank probabilities: posterior resampling; frequentist
netmeta rankogram: MVN resampling; Rucker-Schwarzer 2015 P-scores
are deterministic but pairwise summaries only. Nobody fits
full-likelihood Thurstonian rankings at n >> 20; stated blockers are
the multivariate integrals (Brown & Maydeu-Olivares 2012 line) and
GHK's noisy gradients at small n.

## Related-work anchors for the paper
1. Johnson & Kuhn 2013 -- the "cannot be expressed in closed form"
   foil. 2. Henery 1981 -- independence ancestor. 3. Dunnett-Sobel
   1955 -- factor-reduction ancestor.

## Consequences for the paper plan (SPACINGS.md)
The rank-likelihood result joins spacings/range as a lead exhibit:
the same one-dimensional field machinery prices (i) spacings and
range where Biometrika 1964 stopped at n=4, (ii) any queried
permutation or top-k prefix where the field's own words are "cannot
be expressed in closed form", (iii) exact deterministic rankograms
where NMA resamples. Positioning discipline: composition of known
ingredients (Dunnett-Sobel conditioning + sequential chain), exact
gradients as the fitting payoff, priority language gated on reading
Ennis & Ennis and Henery in full.
