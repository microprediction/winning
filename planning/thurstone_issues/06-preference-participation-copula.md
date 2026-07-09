# Discussion: coupling Thurstonian preference to participation (turnout)

**Type:** discussion

From `Democracy_Correlation_Trade.ipynb` (preserved at `attic/notebooks/` in the
winning repo): calibrate a two-alternative Thurstonian model to a headline win
probability, then introduce a latent participation variable (e.g. "appeal of
democracy" — a person votes only if D < 0) correlated with the preference gap
X1 - X2 by rho, and study P(victory | participation) as a function of rho.

Generalization worth discussing: selection effects in *who shows up* are endemic to
contests — electorates, marketplaces (only some buyers see the product), tournaments
with qualification. A Thurstonian model with a participation copula is a clean way to
model "the field you observe is not a random sample of the population", which also
connects to the walkover/hanger handling already in thurstone's `ClusterSplitter`.

The notebook's machinery is one `multivariate_normal` call; the value here is the
framing, not the code. Candidate for a docs essay or a research/ note alongside the
diffeomorphism thread.
