# Analytic results for Gaussian rank-order contests
(2026-09-24. Every numbered result is checked in `verify_theory.py`;
numbers quoted are from `verify_results.json`. Peter asked for
"analytic game theory results"; the classics are the foil, see
`../NOTES.md`.)

## Model
N contestants. Performance `X_i = mu_i - e_i + s_i eps_i`, eps iid
standard normal, lower is better (min-wins). Prizes `w_1 >= ... >= w_N`
by finishing position, unit purse. Effort `e_i >= 0` costs
`e^2 / (2 kappa)`. Contestant i's payoff is `E[prize_i] - cost`, so the
first-order condition is

    e_i = kappa * inc_i,     inc_i = d E[prize_i] / d e_i = -d E[prize_i] / d mu_i.

Write `L_(k)` for the LUCK of the k-th finisher, `-eps` of whoever
finishes k-th (positive = lucky).

## R0. The incentive is a covariance (Stein's lemma)
For Gaussian noise, `d E[g(X)] / d mu_i = E[g(X) eps_i] / s_i` for any
integrable g. With `g = prize_i`,

    inc_i = -E[prize_i eps_i] / s_i = Cov(prize_i, -eps_i) / s_i.

A contestant's marginal incentive is how much their payout co-moves
with their own luck, per unit of luck. This is the response =
covariance identity of the fluctuation-dissipation chapter (IDEAS.md),
with the prize as the observable and the contestant's own noise as the
conjugate field. The race engine computes the left side exactly; the
right side is what Monte Carlo would estimate. Checked in exp2's notes
(0.1585 vs 0.1577 +- 0.0005 for a favourite; totals 1.169 vs 1.167).

## R1. Total incentive is linear in the prize vector
Summing R0 over contestants with unit noise,

    sum_i inc_i = -E[ sum_i prize_i eps_i ] = sum_k w_k E[L_(k)].

The total incentive of ANY schedule is the prize vector dotted with the
expected luck of each finishing position. The coefficients `E[L_(k)]`
depend on the abilities but not on the prizes, so for a designer
maximising total effort at a fixed field the problem is a linear
programme over the simplex, solved at a vertex:

    pay only the position whose finisher is, on average, luckiest.

By Stein each coefficient is also a sum of exact rank-probability
derivatives, `E[L_(k)] = sum_i d P(rank_i = k) / d(-mu_i)`, which is how
the engine evaluates it.

## R2. Symmetric field: the coefficients are normal order statistics
If all `mu_i` are equal, the k-th finisher's luck IS the k-th largest of
N standard normals, so `E[L_(k)] = E[Z_(k)]` (descending order
statistics). Checked at N = 24: engine vs quadrature agree to 5e-9;
`E[L_(1..4)] = 1.948, 1.503, 1.239, 1.041`.

Consequences for a symmetric field:
- a unit prize step at boundary k buys `c_k = sum_{j<=k} E[Z_(j)]`, which
  peaks mid-field (c_12 = 9.27 at N = 24) and is symmetric, c_k = c_{N-k};
- a top-k equal split has total incentive `c_k / k`, the mean of the
  top-k order statistics, which is strictly decreasing in k;
- hence WINNER-TAKE-ALL maximises total effort in a symmetric field for
  any cost function that makes effort increasing in the marginal
  incentive. Multiple prizes can only win through heterogeneity (R6).
- Lazear-Rosen's `g(0)` is the N = 2 case: `E[Z_(1)] = 1/sqrt(pi)` and
  each of two players faces half of it, `1/(2 sqrt(pi)) = 0.2821`, the
  density of the noise difference at zero.

## R3. Symmetric Nash equilibrium in closed form
In a symmetric field all incentives are equal, so equal efforts shift
every mean equally and leave the race unchanged. The symmetric Nash
effort is therefore exact:

    e* = kappa * (w . E[Z]) / N,   total effort = kappa * (w . E[Z]).

Checked by damped best response at N = 24, kappa = 3: winner-take-all
e* = 0.2435, top-3 (.5/.3/.2) 0.2091, geometric .7 0.1663, all matching
to 1e-13 with zero spread across contestants. With belief variance v
added to every runner the noise scale is `sqrt(1 + v)` and every
coefficient scales by `1/sqrt(1 + v)`: the sharpening measured in exp1.

Participation: each player expects `1/N` of the purse, so the symmetric
field stays whole iff `1/N >= c_part + e*^2 / (2 kappa)`.

## R4. Two unequal players: effort is equal and decays with the gap
N = 2, gap d in ability, winner-take-all. Both players' incentive is the
tie density of the difference at zero,

    inc_1 = inc_2 = phi(d / sqrt 2) / sqrt 2,

so the strong and the weak player exert IDENTICAL effort (the
Lazear-Rosen result with additive noise), and total effort falls with
the gap as a Gaussian: 0.2821, 0.2197, 0.1038, 0.0297 at d = 0, 1, 2, 3
(engine and formula agree to 4 decimals). Equal efforts leave the gap
unchanged, so this is also the equilibrium. Any handicap that closes
the gap raises effort; full handicapping is effort-optimal at N = 2.

## R5. Unequal field: the static rule and the equilibrium disagree
On exp1's 24-player field (abilities N(0,1)), the position-luck
coefficients at zero effort are `1.271, 1.130, 0.970, 0.829, ...` --
still decreasing, so the STATIC rule R1 says winner-take-all. But the
equilibrium total effort by damped best response is

    pay 1st only   2.472
    pay 2nd only   3.189
    pay 3rd only   2.907
    top-3 .5/.3/.2 3.146

Paying second place alone buys 29 percent more equilibrium effort than
winner-take-all. The static coefficients are evaluated at zero effort;
in equilibrium under winner-take-all the favourite's effort widens the
gap, which kills the tie densities at the top boundary, including its
own. Paying second place puts the step where the mid-field is dense and
leaves the favourite -- who finishes first without trying -- out of the
money. That is the mechanism of R6 in a large field.

Caveat that matters: "pay second only" is only an equilibrium because
effort is constrained non-negative. A favourite who could sandbag would
aim for second. Top-3 gets within 1.4 percent of it while still paying
the winner, which is the practical reading.

## R6. Szymanski-Valletti in Gaussian form, with a threshold
Three players: a leader at `-d`, two at 0. Prizes `(1 - s, s, 0)`. By R1
the static total incentive is `(1 - s) E[L_(1)] + s E[L_(2)]`, so a
second prize raises total effort iff

    E[L_(2)] > E[L_(1)]   (equivalently D_2 > 2 D_1 in boundary densities),

the luck of the runner-up exceeds the luck of the winner. In a
symmetric field the winner is always luckier and the condition fails;
with a dominant leader the winner needs no luck while the two chasers
need luck to beat each other, and it holds. The static threshold is

    d* = 2.170 noise units.

In equilibrium (damped best response, kappa = 3) the optimal
second-prize share is s* = 0 at d = 0 and d = 1, s* = 0.3 at d = 2, and
s* >= 0.5 (grid edge) at d = 3; the equilibrium threshold lies between
1 and 2, below the static one, because the leader's own effort widens
the effective gap. Total effort at d = 3 rises from 0.299 (winner-
take-all) to 0.877 (equal split): a factor of three from adding a
second prize. This is the analytic content of "contestants ARE
unequal": the schedule should follow the luck coefficients of the
actual field, not of the symmetric idealisation.

## R7. Time to discouragement
Two players, winner-take-all, participation cost c, public belief from
a unit prior and unit-noise observations, no effort. After t rounds
the belief variance is `v_t = 1/(t+1)` and the expected rating gap is
shrunk to `d t/(t+1)`. The weak player's predictive win probability is
`Phi(-d t/(t+1) / sqrt(2 (1 + v_t)))` and they quit at the first t with

    (d^2 - 2 z_c^2) t^2 - 6 z_c^2 t - 4 z_c^2 > 0,   z_c = Phi^{-1}(1 - c),

never if `d <= sqrt(2) z_c`. At c = 0.005: nobody within 3.64 noise units
of the leader ever quits; predicted quitting rounds 16, 4, 3 at gaps 4,
5, 6 against simulated medians 16, 5, 3 over 200 seeds (the gap-5 case
sits within 0.006 of the boundary, so noise in the realised rating
pushes the median up by one). At d = 3, 97 percent of seeds never quit
within 400 rounds. The hope horizon is long near the threshold and
collapses fast beyond it: a contestant four units back stays sixteen
rounds, six units back stays three.

## R8. Linear-filter facts about handicaps (no simulation)
With steady-state Kalman gain g on the rating:
- a permanent improvement delta under an instantaneous handicap lam is
  paid `(1 - lam) delta` per round once absorbed, plus a transient
  `lam delta (1-g)/g` in total; under a handicap lagged k rounds it is
  paid in full for k rounds first. The lag is a patent life (exp2).
- the impulse response of the rating to a one-round shock integrates
  to one, so a one-time sandbag of size s costs about `inc * s` today
  and returns about `inc * s` in total future handicap: a wash before
  discounting, a loss after it, weaker still with a lag.

## What is proved and what is measured
R0-R4 and R8 are exact statements about the Gaussian model, verified
numerically. R5-R7 are exact computations on particular fields (the
threshold d* = 2.170 is exact for the three-player configuration; the
equilibrium rows are best-response fixed points to 1e-6). Nothing here
covers correlated noise, though R0 and R1 go through unchanged with
`Cov(prize_i, eps_i)` replaced by the covariance against the runner's
own noise component, and the engine's factor races price the
coefficients. Uniqueness of the heterogeneous equilibrium is not shown;
best response converged from zero effort in every case run.
