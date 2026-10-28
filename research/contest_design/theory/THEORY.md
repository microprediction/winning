# Analytic results for Gaussian rank-order contests
(2026-09-24, corrected 2026-10-01 after review. Every numbered result
is checked in `verify_theory.py`; numbers quoted are from its latest
line in the append-only `verify_runs.jsonl` (`verify_results.json` is
the first version's frozen output). Peter asked for
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
conjugate field. The identity is exact; the race engine evaluates the
left side NUMERICALLY, as a central difference (step H = 1e-4) of
lattice rank probabilities (points = 257). Convergence check in
`verify_theory.py`: over points 129..1025 and H <= 1e-4 the incentives
agree with (1025, 1e-5) to 5.2e-9. The right side is what Monte Carlo
would estimate. Checked in exp2's notes
(0.1585 vs 0.1577 +- 0.0005 for a favourite; totals 1.169 vs 1.167).

## R1. Total incentive is linear in the prize vector
Summing R0 over contestants with unit noise,

    sum_i inc_i = -E[ sum_i prize_i eps_i ] = sum_k w_k E[L_(k)].

The total incentive of ANY schedule is the prize vector dotted with the
expected luck of each finishing position. The coefficients `E[L_(k)]`
depend on the abilities but not on the prizes, so for a designer
maximising total STATIC incentive at a fixed field the problem is a
linear programme over the model's feasible set, the ORDERED unit-purse
simplex `w_1 >= ... >= w_N >= 0, sum w = 1`. Its vertices are the
equal top-k splits `(1/k, ..., 1/k, 0, ...)`, not single positions, so
the static optimum is

    pay the top k equally, k maximising the prefix average
    (E[L_(1)] + ... + E[L_(k)]) / k.

(Over the unordered simplex the vertex would be "pay only the luckiest
position", which is outside the model whenever that is not first.)

By Stein each coefficient is also a sum of rank-probability
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
  each of two players faces half of it, an INCENTIVE of
  `1/(2 sqrt(pi)) = 0.2821`, the density of the noise difference at
  zero.

## R3. Symmetric Nash equilibrium in closed form
In a symmetric field all incentives are equal, so equal efforts shift
every mean equally and leave the race unchanged. The symmetric Nash
effort is therefore given in closed form:

    e* = kappa * (w . E[Z]) / N,   total effort = kappa * (w . E[Z]).

Checked by damped best response at N = 24, kappa = 3: winner-take-all
e* = 0.2435, top-3 (.5/.3/.2) 0.2091, geometric .7 0.1663, all matching
to 1e-13 with zero spread across contestants. With belief variance v
added to every runner the noise scale is `sqrt(1 + v)` and every
coefficient, hence every effort, scales by `1/sqrt(1 + v)` (checked:
the 24-player winner-take-all total incentive is 1.94767408, 1.37721355,
0.97383704 at 1 + v = 1, 2, 4, each times sqrt(1 + v) = 1.94767408).
Effort RISES over rounds because v falls: the sharpening measured in
exp1.

Participation: each player expects `1/N` of the purse, so the symmetric
field stays whole iff `1/N >= c_part + e*^2 / (2 kappa)`.

## R4. Two unequal players: effort is equal and decays with the gap
N = 2, gap d in ability, winner-take-all. Both players' incentive is the
tie density of the difference at zero,

    inc_1 = inc_2 = phi(d / sqrt 2) / sqrt 2,

so the strong and the weak player exert IDENTICAL effort
`e_i = kappa inc_i` (the Lazear-Rosen result with additive noise), and
effort falls with the gap as a Gaussian. The incentive is 0.2821,
0.2197, 0.1038, 0.0297 at d = 0, 1, 2, 3 (engine and formula agree to
better than 1e-8); at kappa = 3 that is effort 0.8463, 0.6591, 0.3113,
0.0892 EACH and total effort `2 kappa inc` = 1.69257, 1.31817, 0.62266,
0.17840. Equal efforts leave the gap
unchanged, so this is also the equilibrium. Any handicap that closes
the gap raises effort; full handicapping is effort-optimal at N = 2.

## R5. Unequal field: the static rule and the equilibrium disagree
On exp1's 24-player field (abilities N(0,1), seed 0), the position-luck
coefficients at zero effort are `1.271, 1.130, 0.970, 0.829, ...` --
decreasing, so their prefix averages `1.271, 1.201, 1.124, 1.050, ...`
peak at k = 1 and the STATIC rule R1 says winner-take-all. But the
equilibrium total effort by damped best response (kappa = 3) over the
feasible vertices is

    top-1 (winner-take-all)  2.472
    top-2 equal              3.176
    top-3 equal              3.165
    top-4 equal              3.029
    top-5 equal              2.863
    top-3 .5/.3/.2           3.146

Splitting the purse equally between the first two places buys 28.5
percent more equilibrium effort than winner-take-all; top-3 schedules
are within 1 percent of it. (Best of the top-1..8 vertices and the
.5/.3/.2 schedule; the equilibrium objective is not linear in w, so a
non-vertex ordered schedule could do slightly better, and that search
is not done.) The static coefficients are evaluated at zero effort; in
equilibrium under winner-take-all the favourite's effort widens the
gap, which kills the tie densities at the top boundary, including its
own. A second prize puts a step where the mid-field is dense. That is
the mechanism of R6 in a large field.

Foil outside the model: "pay second only", `w = e_2`, violates
`w_1 >= w_2` and is not a schedule of this game. Reported only for
comparison (3.189, within 0.4 percent of top-2 equal); it would also
invite the favourite to sandbag for second if effort could be negative.

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
numerically. R5-R7 are numerical computations on particular fields:
lattice rank probabilities and finite-difference incentives (converged
to about 5e-9, see R0), a root-find for the threshold d* = 2.170 of the
three-player configuration, and best-response fixed points to 1e-6. Nothing here
covers correlated noise, though R0 and R1 go through unchanged with
`Cov(prize_i, eps_i)` replaced by the covariance against the runner's
own noise component, and the engine's factor races price the
coefficients. Uniqueness of the heterogeneous equilibrium is not shown;
best response converged from zero effort in every case run.
