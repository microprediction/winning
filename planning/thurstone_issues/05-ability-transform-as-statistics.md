# Discussion: the ability transform as a general statistical tool

**Type:** discussion

The most speculative and possibly most interesting thread from winning 1.x. The
notebook `Ability_Transforms_Updated.ipynb` (preserved at `attic/notebooks/` in the
winning repo) treats *any* vector of proportions as winning probabilities, inverts
them through the ability transform, and studies the distribution of implied latent
abilities. It runs this on ~30 real datasets: S&P 500 market caps, prize money,
city sizes, COVID case counts, GDP per capita, word frequencies, earthquake
magnitudes, browser market share, and more — asking when implied ability looks
normal, skew-normal, log-normal, or power-law.

Why it might matter: the transform is a principled alternative to "take logs of
shares" for compositional data, with a generative contest story behind it, and the
shape of the implied ability distribution is a one-parameter-family diagnostic
(vary the skew `a` of the base density; see which choice normalizes the sample).

Possible homes:
- a thurstone docs page ("the ability transform outside racing") with 3-4 punchy
  examples,
- a `winning` module (`ability_transform(proportions) -> abilities`) plus a revived,
  slimmed notebook,
- eventually a short methods paper.

The old notebook is Colab-era (uses `powerlaw`, `nevergrad`, `wordfreq`) and needs a
rewrite against the thurstone API either way.
