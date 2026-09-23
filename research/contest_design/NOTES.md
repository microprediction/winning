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
