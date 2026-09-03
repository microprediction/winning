# The incumbent is always multinomial logit
(Peter, 2026-09-03. The governing principle for the whole
applications program. Adopt it; re-read the adjudications through it.)

## The claim
For any problem of the form "probability over N alternatives" --
which wins, which is chosen, which fails first, which is best --
the default tool practitioners reach for is a LOGIT: softmax,
Bradley-Terry, Elo, Rasch, conditional logit, or the independence
formula 1-(1-p)^k. Different names, one model, one sin: the Gumbel /
IIA / independence assumption.

## Why it is always logit (the paper's own thesis)
Logit is the ONLY member of the discriminal-race family that is
analytically tractable at scale. Yellott 1977: softmax is exactly
the Gumbel special case of the Thurstone race. So practitioners did
not choose logit because reality is Gumbel; they chose it because it
was the only feasible choice at scale. "The incumbent is always
logit" is a corollary of "logit was the only feasible choice at
scale until now" -- which is precisely what this package removes.

## The consequence for adjudication (the correction to my error)
I kept benchmarking applications against the domain's SPECIALIZED
tool -- GHK for probit, Ford's burst-Markov for durability,
additional-greedy coverage for tests -- and finding "strong
incumbents." That is the wrong baseline. The specialized tools are
the exception; the DEFAULT the median practitioner uses is logit /
softmax / independence. The right question for any application is
not "can we beat the domain's best engine" but:
  1. Is the working incumbent a logit (independence / IIA)? Almost
     always yes.
  2. Does IIA cost something here -- are the alternatives correlated?
  3. Is the correlation STRUCTURED (factor / block / tree)?
If yes/yes/yes, the pitch is uniform and the same sentence every
time: correlated probit at the scale where you are currently forced
to use logit.

## The genuine exceptions (few, and worth naming honestly)
- Purely COMBINATORIAL incumbents that are not logit at all:
  additional-greedy coverage (set cover) in test selection. There
  the observable structure already carries the correlation and logit
  is not the foil. Rare.
- DATA problems, not method problems: Backblaze failed on
  administrative failure dates, not on the incumbent. Logit-framing
  does not rescue a data confound.
Everywhere else the foil is logit, and the market is every place
logit is used as a scale-forced approximation to a correlated race.

## Restated for positioning
The LinkedIn line "probit finally competes with logit at modern
scale" is not a tagline; it is the entire thesis and the entire
market. Every application is one instance of it. Stop apologizing
per-domain; lead with the universal foil.
