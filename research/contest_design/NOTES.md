# Contest design on the ability axis
(Peter's idea, 2026-09-23, in conversation: "using winning to compute
the optimal incentive / prize-money structure ... in some kind of
iterative setting where you want people to keep competing but some
realize that they really have no chance ... where the natural 'work'
axis is the ability scale." Foil requested: the classic one-shot Nash
contest literature.)

## The identity
If effort moves a contestant along the ability axis, the marginal
incentive to work is the derivative of expected prize with respect to
own ability. With prizes w_k attached to finishing positions,

    d P(rank_i <= k) / d mu_i  =  density that i ties at the k-th boundary

so

    incentive_i = sum_k (w_k - w_{k+1}) * tie density of i at boundary k.

Effort responds to prize STEPS, and only where the contestant has
finishing density. Winner-take-all is one step at the top, felt only
by those with density at the front. This is the discouragement effect
as an object the engine prices: rank_probabilities gives every
P(rank_i = k) exactly, and the derivative is a central difference of
it (or tie_densities, which are the win-boundary case). Contestants
who are far from every paying boundary have zero incentive whatever
the purse.

## Why the engine is the right tool
- Heterogeneous, correlated fields of N contestants priced exactly at
  every position: the incentive VECTOR for the whole field in one
  pass, not a symmetric closed form.
- Handicaps are the inversion: offsets that hit a target share vector
  are abilities_from_race at those shares (the same operation as
  DeepSeek's aux-loss-free MoE bias, applications/ml-systems.md 5).
- The iterative part needs a belief, and AbilityTracker is one:
  predictive races run at posterior means with 1 + v as the per-runner
  variance, so early-round hope (large v) and its collapse (v -> 0)
  are the same object that drives quitting.
- Paying on surprise (rank the residual against rating) is the
  martingale diagnostic turned into a payout rule; it puts every
  contestant's density back at the boundary by construction.

## The literature as foil
Read at source this session unless marked [U].
- Lazear & Rosen (1981), Rank-Order Tournaments as Optimum Labor
  Contracts, JPE [U]. Two players, X_i = e_i + eps_i, prize spread
  dW. First-order condition: dW * g(0) = c'(e), g the density of the
  noise difference at zero -- i.e. PRIZE SPREAD TIMES TIE DENSITY. The
  identity above is its N-player, full-prize-vector, heterogeneous
  generalisation. Their heterogeneity discussion is the origin of the
  handicapping literature.
- Moldovanu & Sela (2001), The Optimal Allocation of Prizes in
  Contests, AER 91(3) 542-558 (abstract read; aeaweb.org/articles?id=
  10.1257/aer.91.3.542). All-pay auction, private cost types, designer
  maximises expected effort: with linear or concave effort cost the
  whole purse goes to ONE first prize; with convex cost several
  prizes can be optimal. Symmetric-ex-ante contestants; one shot.
- Szymanski & Valletti (2005), Incentive effects of second prizes,
  EJPE 21(2) 467-481 (abstract read; sciencedirect S0176268004000795).
  With heterogeneous abilities a second prize can RAISE aggregate
  effort: under winner-take-all the weak have negligible incentive;
  a second prize gives them one, which in turn makes the strong work
  harder. This is our identity in a three-player Tullock setting.
- Sisak (2009), Multiple-prize contests: the optimal allocation of
  prizes, J. Econ. Surveys 23(1) 82-114 [U]: the survey of the above.
- Rosen (1986), Prizes and Incentives in Elimination Tournaments, AER
  [U]: sequential elimination needs an escalating top prize to keep
  incentives in late rounds -- the only classic dynamic result, and
  still complete-information.
- Harris & Vickers (1987), Racing with Uncertainty, REStud [U]: the
  laggard slackens, the leader works harder -- discouragement as a
  dynamic equilibrium property.
- Brown (2011), Quitters Never Win: The (Adverse) Incentive Effects of
  Competing with Superstars, JPE 119(5) 982-1013 (abstract read).
  Golfers' first-round scores about 0.2 strokes worse when Tiger Woods
  plays, 0.8 over the tournament. Contested by Connolly & Rendleman
  (SSRN 2533537) [U]. The empirical face of "some realize they have no
  chance".

What the classics share: one shot, effort chosen simultaneously,
abilities (or cost types) known or drawn from a known symmetric
distribution, and a closed-form symmetric equilibrium. None has a
belief that sharpens over rounds, and none prices a heterogeneous
correlated field position by position. That is the gap the toy
occupies.

## exp1_prize_schedules: the toy
N = 24 contestants, true abilities a_i ~ N(0, 1) in units of the
per-round noise sd; X_i = a_i - e_i + eps_i; unit purse per round;
effort cost e^2 / (2 kappa), kappa = 3; participation cost 0.005 of
the purse per round; T = 50 rounds; public belief = AbilityTracker on
the results; quit for good when expected prize - costs < 0 under the
public belief. Handicap dial lam in {0, 0.5, 1}: positions decided on
X_i - lam (m_i - mean m). Schedules: winner-take-all, top-3
(.5/.3/.2), geometric with ratio .5/.7/.85/.95, linear, flat.
Foil rows: the one-shot complete-information Nash equilibrium of the
same fields by damped best response on the same incentive, forced
entry and free entry.

Myopia caveat, stated up front: contestants ignore that this round's
effort raises next round's rating and therefore next round's
handicap (the ratchet), and nobody sandbags. Both bite hardest at
lam = 1, so the surprise-payout rows are an upper bound on what a
naive surprise rule delivers.

## Results (2026-09-23; results.json; N = 24, T = 50, two fields)
Talent terciles below are (strong, mid, weak). "Effort" is the sum of
e_i over rounds and contestants; spend is 50 in every row (all
positions among the active are paid), so effort is also effort per
unit purse.

One-shot complete-information Nash, forced entry (per round):

    schedule        effort   by tercile           free entry: n, effort
    wta              3.03    2.58  0.38  0.07     11.0   2.88
    top3 .5/.3/.2    3.47    2.48  0.80  0.19     18.0   3.40
    geometric .5     3.37    2.33  0.82  0.22     19.0   3.31
    geometric .7     2.98    1.72  0.92  0.34     21.5   2.94
    geometric .85    2.05    0.93  0.72  0.40     24     2.05
    linear           1.24    0.39  0.45  0.40     24     1.24
    flat             0.00                          24     0.00

Both classical results reproduce on the toy. Winner-take-all leaves
the weak tercile at 0.07 total effort -- the discouragement effect --
and a second and third prize raise TOTAL effort by 14 percent
(Szymanski & Valletti); the cost is quadratic, so Moldovanu & Sela's
winner-take-all theorem does not apply and multiple prizes can win.
With free entry winner-take-all loses 13 of 24 contestants.

Iterated, public belief from the tracker, quitting allowed:

    schedule | lam      effort  final n   effort by tercile     paid by tercile   effort/round t=0 -> 45
    wta      | 0        167.3    12.5     132.0  30.2   5.1     47.5  2.5  0.0    4.13 -> 3.27
    wta      | 0.5      240.7    23.0     133.1  68.3  39.3     33.0 12.5  4.5    4.13 -> 4.92
    wta      | 1        261.4    24       87.1  87.1  87.1     16.0 20.0 14.0    4.13 -> 5.45
    top3     | 0        166.7    17.5     112.3  44.4  10.0     42.6  7.0  0.5    3.55 -> 3.31
    top3     | 0.5      210.6    24       104.2  64.4  42.0     31.0 13.8  5.2    3.55 -> 4.34
    top3     | 1        224.5    24        74.8  74.8  74.8     17.5 17.8 14.7    3.55 -> 4.68
    geom .5  | 0        162.1    18.5     104.1  44.6  13.4     41.1  7.6  1.3    3.43 -> 3.24
    geom .7  | 0        141.5    21.5      77.1  43.7  20.7     35.1 12.0  2.9    2.82 -> 2.88
    geom .85 | 0         97.1    24        42.6  32.7  21.8     28.3 14.5  7.2    1.86 -> 1.97
    linear   | 0         58.5    24        18.6  20.9  19.0     22.9 16.4 10.7    1.10 -> 1.19
    flat     | any        0.0    24
    (lam = 0.5 and 1 raise effort and hold participation at 24 for
    every schedule; full grid in results.json.)

What the iteration adds.
1. HOPE IS WORTH EFFORT, AND IT DECAYS. Round-0 effort under
   winner-take-all is 4.13 against a complete-information 3.03: with
   belief variance 1 every contestant has density at the top boundary.
   By round 10 half the field has learned it has no chance and the
   per-round effort has fallen to the one-shot level with half the
   entrants. The one-shot free-entry count (11) is the iterated
   endpoint (12.5): the classic equilibrium is where the dynamic
   lands, not where it starts.
2. TOP-3 STOPS BEATING WINNER-TAKE-ALL. Over 50 rounds the two tie on
   effort (167 vs 167); top-3 keeps five more contestants. The one-shot
   14 percent edge is what winner-take-all gives back in its early,
   uncertain rounds. A designer who only sees the one-shot theorem
   over-buys consolation prizes.
3. THE SURPRISE RULE DOMINATES. Ranking on the residual against rating
   (lam = 1) gives the most effort of anything in the grid, 261 against
   167 for the same purse, keeps all 24 in, equalises effort across
   terciles and pays them almost equally. The mechanism is the
   identity: with all predictive means equal the tie densities are
   maximal, and per-round effort is 4.13 at t = 0 and 5.45 at the end,
   the ratio being sqrt(1 + v) -- the density of the symmetric field
   sharpening as the belief does. This is the symmetric-contest
   optimum of the classics, manufactured from a heterogeneous field by
   the rating.
4. THE DIAL HAS A PRICE. lam = 1 pays the strong tercile 16 of 50,
   lam = 0 pays them 47.5. lam = 0.5 gets 240 of the 261 effort while
   still paying the strong 33: most of the effort gain at half the
   redistribution. Over a longer horizon the return to being good is
   what buys investment in ability, which the toy does not model; the
   dial is where that trade is made.

Caveats beyond the myopia already stated. The one-shot rows are
damped best response to a 1e-4 tolerance, not a proof of equilibrium.
Two fields, no error bars; the ordering of schedules is not close
enough for that to matter, the exact numbers are. Quitting is
irrevocable and myopic. The tracker runs with drift = 0, so belief
variance falls as 1/(t + 1); a positive drift would keep some hope
alive under every schedule and shrink the surprise rule's advantage.
Nobody sandbags, and under lam > 0 they would.

Next if pursued: (a) let contestants invest in ability across rounds
and find the lam that maximises long-run effort; (b) a
sandbagging-proof surprise rule (pay on the residual against a rating
frozen k rounds back, or against the market); (c) solve the designer's
per-round problem with the exact incentive Jacobian instead of a grid.

## exp2_investment: the micromanager's problem (2026-09-24)
Reframed by Peter: the designer is a micromanager who pays some number
of forecasters and consumes their forecasts. "You cannot just award a
sole winner as their absence will leave you exposed." And: "you need
incentive for people to pursue improvements to the models ... what
should that prize schedule be and what ends up being optimal under
long-running simulations or theory?"

So the objective is not total effort. It is the quality of the pool
the platform can draw on, -E[min_i X_i] over the active field (the
best forecast on offer), and its ROBUSTNESS: the same quantity with
the most valuable member deleted. The gap is exposure, the largest
deletion value, a cavity object. Both are computed exactly on a
lattice from the true performance means. Ability is now a stock:
each round a contestant also chooses a permanent improvement u_i,
valued by the steady-state rule that a handicap lam claws back lam of
any gain once the rating catches up, and a handicap LAGGED k rounds
leaves the gain fully rewarded for k rounds first. Tracker drift is
positive so beliefs can follow moving abilities.

### Theory 1: the incentive is a covariance (Stein), so it is
### Martin's fluctuation-dissipation identity
For X_i = mu_i + s_i eps_i with Gaussian eps, Stein's lemma gives

    d E[prize_i] / d mu_i = E[prize_i eps_i] / s_i = Cov(prize_i, eps_i) / s_i.

A contestant's marginal incentive is how much their payout co-moves
with their own luck, per unit of luck. Verified against the exact
lattice incentives by Monte Carlo on the exp1 field with top-3 prizes:
0.1585 vs 0.1577 +- 0.0005 for the favourite, 0.0014 vs 0.0014 for the
tail-ender, totals 1.169 vs 1.167; heteroscedastic totals 1.166 vs
1.165. (Min-wins flips the sign.) This is the response = covariance
identity of the REINFORCE chapter (IDEAS.md), with prize as the
observable and the contestant's own noise as the conjugate variable.
The engine computes the covariance exactly instead of sampling it.

Consequences.
- TOTAL incentive of a schedule = E[ sum_i prize_i eps_i ] = the
  expected prize-weighted surprise. A schedule motivates the field
  exactly to the extent it pays luck.
- SYMMETRIC FIELD (what lam = 1 manufactures): a unit prize step at
  boundary k buys c_k = sum of the top-k expected order statistics of
  N standard normals (Stein again: sum_i E[X_i 1(rank_i <= k)]).
  Engine values at N = 24: c_1 = 1.948 = E[max], c_2 = 3.451,
  c_12 = 9.269 (the peak: spacings are tightest mid-field), and
  c_k = c_{N-k}. But a top-k EQUAL split has total incentive c_k / k
  = the mean of the top-k order statistics, which falls
  monotonically: 1.948, 1.726, 1.563, ... 0.772 at k = 12, 0 at
  k = 24. Geometric r = .5/.7/.85 give 1.616/1.330/0.876; linear 0.519.
  So with a symmetric field and any cost that makes effort increase
  in the marginal incentive, WINNER-TAKE-ALL MAXIMISES TOTAL EFFORT,
  because the expected maximum exceeds every top-k average. Multiple
  prizes only ever win through heterogeneity (exp1's one-shot: the
  static incentive at zero effort has top-1 at 1.271 vs top-3 at
  1.124 even on the heterogeneous field; the 14 percent top-3 edge
  is an equilibrium effect -- winner-take-all makes the strong pull
  away, which kills the tie densities that pay).
- Per-round effort under lam = 1 is kappa c(w) / sqrt(1 + v), which is
  the sqrt(1 + v) sharpening measured in exp1.

### Theory 2: what pays improvement
A permanent gain delta earns inc_i delta per round while it is not
priced in. Under handicap lam it earns (1 - lam) inc_i delta once the
rating has caught up; under a lag k it earns the full amount for k
rounds first. So over a horizon H,

    value of improving = inc_i * [ k + (1 - lam)(H - k) ] * delta.

The lag is a PATENT LIFE: under the pure surprise rule (lam = 1) the
only return to getting better is the k rounds before your rating
absorbs it. Total investment volume is sum_i inc_i(lam) times the
bracket; the sum rises with lam (symmetry) and the bracket falls, and
exp1's numbers (sum at lam = 0 is 3.27, at lam = 1 is 5.45, ratio
1.67 < 2) say the product is maximised at lam = 0 for k = 0: winner-
take-all buys the most improvement VOLUME. But it buys it from one
or two players, and the micromanager does not consume volume.

### Predictions, written before the sweep finished
(i) Winner-take-all with no handicap produces a runaway superstar:
the leader invests, pulls away, the tie densities of everyone else
vanish, the field quits, and the pool is one player -- highest
"best", worst "robust". (ii) lam = 1 keeps everyone and maximises
effort but nobody invests except through the lag, so "best" grows
only by effort. (iii) The micromanager's optimum is interior: a steep
schedule with a partial handicap, or the full surprise rule with a
long lag, trading a little of the leader's improvement for a bench.
(iv) Among schedules, top-3 should beat winner-take-all on robust
quality at every lam, and the ordering on "best" should flip as lam
rises.

### Results (T = 80, N = 24, three fields, kappa_u = 0.01, H = 20,
### tracker drift 0.05; results.json)
Columns: cumulative over 80 rounds. "best" is the improvement in the
quality of the best forecast on offer, -E[min X] over the active
field, relative to the round-0 field at zero effort; "robust_m" the
same after greedily deleting the m most valuable members; exposure
is best - robust_1. Ability gain is the sum of permanent improvement
over the strong / mid / weak talent terciles.

    schedule | lam | lag   effort  improve   best  robust1  exposure  final n   ability gain (S / M / W)
    wta      | 0   | 0       138     460    138.2    60.4     77.7      3.0     7.4  0.9  0.3
    wta      | .25 | 0       216     621    132.4    54.0     78.5      7.0     7.8  1.7  0.4
    wta      | .5  | 0       327     748     95.0    45.5     49.4     19.3     5.8  2.8  1.2
    wta      | .75 | 0       378     622     41.1    30.6     10.4     24       2.4  1.8  1.3
    wta      | 1   | 0       386     386     16.1    16.1      0.0     24       0    0    0
    wta      | 1   | 5       316     515     38.0    31.4      6.5     13.7     2.6  1.6  0.9
    wta      | 1   | 15      261     713     95.1    68.2     26.9      8.0     8.7  2.7  0.5
    top3     | 0   | 0       204     711     86.4    65.7     20.8     10.3     9.0  2.9  0.4
    top3     | .25 | 0       258     746     75.6    52.1     23.5     16.0     7.0  3.3  1.2
    top3     | .5  | 0       302     693     54.8    38.9     15.9     23.0     4.5  2.8  1.6
    top3     | .75 | 0       326     535     31.0    25.5      5.5     24       2.0  1.6  1.3
    top3     | 1   | 0       331     331     13.8    13.8      0.0     24       0    0    0
    top3     | 1   | 5       288     469     25.7    23.4      2.3     18.3     1.9  1.8  1.0
    top3     | 1   | 15      257     722     57.4    49.7      7.7     14.0     6.7  4.3  0.7
    geom .7  | 0   | 0       193     688     52.2    44.5      7.7     18.0     6.4  3.6  1.5
    geom .7  | .25 | 0       218     639     45.4    37.0      8.4     21.0     4.9  3.2  1.6
    geom .7  | .5  | 0       245     563     35.0    28.5      6.4     24       3.2  2.4  1.7
    geom .7  | 1   | 0       264     264     11.0    11.0      0.0     24       0    0    0
    geom .7  | 1   | 15      226     653     35.3    33.0      2.3     20.7     4.2  3.8  2.3
    (deeper deletions and lag = 30 rows in the table after the reading.)

Reading it.
1. THE HANDICAP IS THE WRONG DIAL FOR THE MICROMANAGER. lam buys
   effort (138 -> 386 under winner-take-all) and participation, and
   destroys improvement: best falls 138 -> 16 and at lam = 1 nobody
   invests at all. exp1's headline, "the surprise rule dominates",
   was true for effort and is false for quality. Prediction (ii)
   confirmed; the exp1 recommendation is withdrawn for any designer
   who consumes the forecasts rather than the effort.
2. WINNER-TAKE-ALL BUYS THE BEST FORECAST AND NOTHING ELSE. best =
   138, the top of the grid, with the strong tercile gaining 7.4 sd
   of ability -- concentrated in the one to three players still
   competing at round 80. Exposure 78: more than half the quality
   evaporates with one departure, and the pool is three deep.
   Prediction (i) confirmed on the mechanism, wrong on robust_1: the
   survivors ALL invested, so deleting one of three still leaves a
   good forecaster. The single-deletion measure is too kind to a
   three-player pool; hence the m = 2, 3 rerun.
3. THREE PRIZES ARE THE ROBUST STEEP SCHEDULE. top-3 with no
   handicap: best 86 (62 percent of the superstar's), robust_1 66
   (the best of any lam = 0 row), exposure 21 (a quarter of winner-
   take-all's), ten players left, and the largest total improvement
   volume of any un-handicapped schedule (711). Heterogeneity is
   doing the work the theory said it would: the second and third
   steps keep the mid tercile's tie densities alive long enough for
   it to invest (2.9 sd against 0.9 under winner-take-all).
4. THE PATENT LAG IS THE RIGHT DIAL. Winner-take-all on surprise with
   a 15-round lag: best 95, robust_1 68 (the top of the grid),
   exposure 27, eight players. It pays improvement for exactly 15
   rounds and then takes it back, which is enough to make the strong
   invest 8.7 sd while the equalised race keeps the mid tercile
   working. Lag 5 is too short a patent: improvement 515 and the
   leader's transient advantage still discourages (13.7 players).
   Longer lags in the rerun.
5. FLAT SCHEDULES BUY BREADTH, NOT QUALITY. geometric .7 spreads the
   ability gain across all terciles (6.4 / 3.6 / 1.5, and 4.2 / 3.8 /
   2.3 with lag 15) with 18-21 players, at a third of the best
   quality. If the micromanager combines forecasts rather than picks
   one, this is the row to look at; that objective is not computed
   here.

### Deeper deletions and longer patents (same runs; rerun with M_DEL = 3, lag 30 added)

    schedule | lam | lag    best  robust1 robust2 robust3  improve  final n   ability gain (S / M / W)
    wta      | 0   | 0     138.2   60.4    11.0   (empty)    460      3.0     7.4  0.9  0.3
    wta      | .25 | 0     132.4   54.0    39.6    28.7      621      7.0     7.8  1.7  0.4
    wta      | .5  | 0      95.0   45.5    41.2    37.7      748     19.3     5.8  2.8  1.2
    wta      | 1   | 15     95.1   68.2    59.3    52.2      713      8.0     8.7  2.7  0.5
    wta      | 1   | 30    156.7   77.3    58.4    41.8      679      6.3    10.8  1.5  0.3
    top3     | 0   | 0      86.4   65.7    54.1    45.8      711     10.3     9.0  2.9  0.4
    top3     | .25 | 0      75.6   52.1    46.2    42.3      746     16.0     7.0  3.3  1.2
    top3     | 1   | 15     57.4   49.7    46.9    44.7      722     14.0     6.7  4.3  0.7
    top3     | 1   | 30     80.7   60.9    54.8    50.4      806     12.7     8.9  4.9  0.5
    geom .7  | 0   | 0      52.2   44.5    41.5    38.6      688     18.0     6.4  3.6  1.5
    geom .7  | 1   | 30     46.9   42.4    40.5    38.8      743     20.0     5.8  4.4  2.3
    (every lam = 1, lag = 0 row: robust_m = best, nobody invests;
    "(empty)" = the pool had three members and deleting three leaves
    no forecaster at all -- the floor value is arbitrary, the
    emptiness is the point.)

6. THE SUPERSTAR POOL IS THREE DEEP AND THEN IT IS NOTHING. Winner-
   take-all with no handicap: robust_2 = 11, robust_3 = empty. Two
   departures and the micromanager has one forecaster; three and
   none. Prediction (i) fully confirmed once the deletion goes past
   one.
7. THE LAG DOMINATES THE HANDICAP DIAL EVERYWHERE. As lag -> infinity
   the handicap freezes at the round-0 rating (zero for all), so
   lag = infinity IS lam = 0: the lag interpolates between the pure
   surprise rule and no handicap along a DIFFERENT path from lam, and
   a better one. Winner-take-all with lag 30 beats winner-take-all
   with no handicap on best (157 vs 138) AND robust_1 (77 vs 60) AND
   players (6.3 vs 3): thirty rounds of near-equality keep more
   players investing for longer, which produces a stronger eventual
   leader (10.8 sd) with a deeper bench behind. Every lam > 0 row is
   dominated by some lag row on quality at equal or better robustness.
8. THE PATENT LENGTH SETS THE DEPTH OF THE BENCH. For winner-take-all,
   lag 15 maximises robust_2 and robust_3 (59, 52); lag 30 maximises
   best and robust_1 (157, 77) at the cost of robust_3 (42). For
   top-3, lag 30 is the all-rounder: the most total improvement in
   the grid (806), robust_3 = 50, 12.7 players, the mid tercile
   gaining 4.9 sd -- the largest mid-tercile gain anywhere.
9. THE ANSWER DEPENDS ONLY ON HOW DEEP A BENCH YOU NEED.
   - You consume one forecast and can replace a superstar: winner-
     take-all on surprise, lag 30. best 157.
   - You need two or three you can trust: winner-take-all on surprise,
     lag 15 (robust_3 = 52) or top-3 on surprise, lag 30 (robust_3 =
     50 with 60 percent more players and the most total improvement).
   - You combine many forecasts: geometric with a long lag; a third
     of the peak quality, twenty players all improving.
   - Never the equalising handicap, at any strength, for any of these.

Answer to "what should the schedule be". Pay a small number
of places -- one if you can replace a superstar, three if you need a
bench -- and judge performance against each contestant's OWN rating
from 15-30 rounds back rather than against the field. The lag is a
patent life: an improvement is paid for that long and then absorbed
into the handicap. It keeps the unequal field near enough to equal
that the mid-field keeps working and investing, without destroying
the return to getting better that the instantaneous handicap
destroys. Between the two robust options, top-3 with lag 30 buys a
deeper and broader bench (12.7 players, the most total improvement)
and winner-take-all with lag 15 buys a slightly better top three from
eight players. Contestants are unequal, so the classic symmetric
winner-take-all theorem is the wrong guide; Szymanski-Valletti's
heterogeneity result is the right one, and the lagged rating is how
to have both its multiple-prize breadth and a strong leader.

Caveats. kappa_u, H, the drift and the participation cost are set,
not calibrated; the ORDERINGS above are the claim, the numbers are
not. Contestants value improvement by a steady-state rule, not by
solving the game; effort is myopic; nobody sandbags; quitting is
irrevocable. Three fields; standard deviations are in the log lines
of results.json and are small relative to the gaps discussed except
for winner-take-all at lam <= .25, where the identity of the
survivors moves the total by a third.
