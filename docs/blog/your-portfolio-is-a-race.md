# Your Portfolio Is a Race

*Or: why the oldest model in psychometrics keeps turning up in allocation, and
what it buys you when it does.*

Portfolio weights are positive and sum to one. So are probabilities. This is
usually treated as a coincidence of bookkeeping, and weights are produced by
optimizing something — variance, drawdown, a utility — then normalizing.

Here is a different reading. Suppose each asset's future performance is a
random variable: an ability plus noise, with the noises correlated in the ways
that matter (sectors move together, styles move together, two ETFs holding the
same stocks move together). Run the contest. The weight of an asset is the
probability that it turns out to be the best one:

    w_i = P(Y_i = max_j Y_j),    Y = mu + correlated noise.

That is Thurstone's 1927 model of comparative judgment, which is also the
random-utility model of discrete choice, which is also how a horse race works.
The claim of this post is not that this is a cute analogy. It is that the race
is a *machine*, and once your weights are its output, a set of chronic
portfolio problems become computations instead of hacks.

## Diversification without a penalty term

The most common complaint about naive scoring rules is concentration: two
nearly identical assets both score well, both get big weights, and you own the
same bet twice. The standard fix is to bolt on a diversity penalty and tune it.

In the race there is nothing to bolt on. If two assets are strongly
correlated, they are running the same race leg, and they *split* one
probability of being best between them. Add a third clone and the three of
them still jointly carry roughly what one carried alone. Diversification is
not a regularizer here; it is a theorem about maxima of correlated variables.
The model cannot double-count a bet, because the event "A is best" and the
event "B is best" are disjoint even when A and B are near-twins.

## Exclusions done right

Every practical mandate deletes things from the menu: ESG screens, sanctioned
names, a client who will not hold tobacco. The standard operation is to zero
the excluded weights and renormalize the rest.

Renormalization silently assumes something false. It assumes the excluded
asset's weight redistributes *proportionally* — which is the independence-of-
irrelevant-alternatives axiom, the same defect that makes softmax the wrong
model of choices and Harville's formula the wrong price for a trifecta. When
you delete a tech stock from the menu, its probability of being best should
flow disproportionately to the *other tech stocks* — the ones that win in the
same states of the world — not pro rata to utilities and bonds.

The race computes the correct redistribution mechanically: delete the runner,
divide it out of the field, and re-read the contest. (In the code this is one
division — the cavity trick — not a re-optimization.) Measured on real
classifier probabilities and on betting markets, renormalization and correct
deletion differ materially, and the direction of the error is systematic: the
excluded asset's correlated neighbors are always underweighted by pro rata.

## Concentration limits without leaving the model

Now the constraint that actually prompted this post. Finance imposes
concentration limits: no name above 5%, no sector above 25%. The standard
operation, again, is clip-and-renormalize. Two things go wrong, and we can
put numbers on both from a twelve-asset example with one factor of
correlation.

Unconstrained, the race puts 36.3% on the strongest name. Cap it at 25% and
cap its four-name sector at 60%. Clipping the name and renormalizing yields a
sector totaling **76.4%** — the name cap leaked straight into the sector,
violating the second constraint while pretending to satisfy the first. And
the clipped vector is no longer the output of *any* race: whatever coherence
the correlation model gave you — the exclusion behavior above, the
diversification theorem, the ability to answer follow-up questions — is gone.

The race alternative: find the **nearest contest that satisfies the caps**.
Perturb the abilities as little as possible, subject to the resulting win
probabilities obeying every limit. This is a small smooth optimization (the
race has an exact, cheap Jacobian), and its output is exactly a race again —
same correlation story, same model, caps binding precisely where active: the
polished portfolio hits 25.0% and 60.0% on the nose, and every downstream
question you ask of it is still answered by the same machine. In the
`winning` package this is one call:

```python
from winning.factor.polish import polish_race
p, mu, info = polish_race(p0=weights, V=V, D=D,
                          name_caps=caps, groups=[(tech_idx, 0.25)])
```

There is a pleasing correspondence here with Schur-complement portfolio
construction, where sub-portfolios are solved against a conditionally-adjusted
environment rather than in isolation. The race version of that idea prices
sector *blocks* — each block a private factor, blocks coupled through a
market factor, hierarchies of blocks at any depth — and the arithmetic stays
linear in the number of assets. A race over one million runners in twenty
thousand sectors prices in about twelve seconds on a laptop, and the block
count is irrelevant to the cost: it is a segment boundary, not a matrix.

## The market's portfolio is a race you can invert

Because the map from abilities to weights is exact and smooth, it runs
backwards. Hand the machine a set of observed weights — a benchmark, a
competitor's disclosed book, the market portfolio — and it returns the
abilities that would have produced them under your correlation model. Views
then live in ability space, where they belong ("I think this company is half
a sigma better than the market thinks"), and the modified weights come back
out coherent, with every constraint applied by polishing rather than surgery.
This is the transport step of allocation, done in one place by one model.

One honest subtlety the inversion taught us: a weight of zero is not a
measurement of an ability, it is only a *bound* on one. The machinery treats
sub-resolution weights accordingly, which is exactly how you should read a
benchmark's zero position in a small-cap you happen to like.

## What this is not

The race functional P(best) is not mean-variance utility, and I will not
pretend a theorem says it allocates optimally for cumulative returns. Where
it is *literally* the right objective is when the payoff is max-like: venture
books, R&D pipelines, drug candidates, best-of-n generation — anywhere one
great outcome redeems the batch and redundancy is the enemy. For conventional
portfolios it is better understood as a transform with unusually good
properties: monotone in views, correlation-aware by construction, closed
under exclusion and conditioning, invertible, and constrainable without
leaving its own model class. Most weight-producing pipelines can claim none
of those.

And the correlations must come from somewhere — a factor model, sectors, a
kernel — as they must in any allocator. The race does not conjure structure;
it *spends* whatever structure you give it, coherently, in every operation.

The machinery described here is in the open-source `winning` package
(github.com/microprediction/winning): exact win probabilities for
independent, factor, block, nested and tree correlation structures, ordered
outcomes, top-m probabilities, exact Jacobians, inversion, and constrained
polishing, with a Rust core where it matters. Your portfolio was already
summing to one. It may as well mean something.
