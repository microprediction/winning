# Fast qPO: can a low-rank correlated probit replace the Monte Carlo?

qPO (multipoint probability of optimality) scores a candidate molecule by the
probability that it is the best in the library,

    p_i = P(Y_i = max_j Y_j),

under the joint Gaussian-process posterior, and buys the b candidates with the
largest p_i. Because the posterior is correlated, two near-duplicate molecules
split one probability between them, so a qPO batch disperses itself without a
diversity penalty bolted on. That is the idea worth preserving.

The released implementation (github.com/jenna-fromer/qPO) cannot compute p_i.
It draws 10,000 samples from the joint posterior and counts winners, and when
the candidate pool exceeds 10,000 molecules it first discards everything
outside the top 10,000 by upper confidence bound, because sampling a dense
N-dimensional Gaussian costs O(N^2) per draw. QM9 has 133,802 molecules.

This directory tests one hypothesis: that the posterior is well enough
approximated by

    Sigma ~= V V' + D,     V of rank r << N,

for a deterministic factor probit to compute the same acquisition decision far
faster. Conditional on the r factors the candidates are independent, so the N
idiosyncratic dimensions integrate exactly and only the r factor dimensions
need quadrature.

The experiment was designed to be able to fail. If qPO needed rank in the
hundreds before the batch stabilised, the method would stop being fast and the
application would be dead. Phase VII below is that kill test.

## Where things live

    snapshot.py          freeze one GP posterior; every method then reads the same (mu, Sigma)
    pom.py               the six estimators, and the lattice kernel
    factorize.py         rank-r factor models of a covariance (the oracle)
    factorgp.py          rank-r factor models straight from the GP, with no N x N matrix
    metrics.py           agreement, batch overlap, qPO efficiency, diversity
    run_phase3.py        is the fast calculation right, and how many nodes does it need
    run_sweep.py         the rank ladder (phases II, IV, V, VII)
    run_budget.py        accuracy against wall time on the real posterior
    run_phase6.py        runtime scaling, and the matched-accuracy frontier
    run_full_library.py  all 133,702 candidates against the 10,000 the prefilter keeps
    run_closed_loop.py   the twenty-round Bayesian optimization loop (phases IX, X)
    collect_results.py   assembles results/all_results.csv
    make_figures.py      the six figures
    envelope.py          one-factor conditional envelopes for arbitrary covariance
    covfamilies.py       covariance families for the general-covariance experiments
    theory.py            why a badly-approximated covariance gives a good batch
    run_theory.py        the sensitivity calculation against measured error
    run_envelope.py      variance reduction of envelopes against winner counting
    tests/               62 anchors; nothing below was believed before these passed

Related: `../shifted_cv/` attacks general-covariance share inversion on 92
problems using this directory's envelope kernel (the two implementations agree
to 2.2e-16). See "Cross-check against research/shifted_cv" below.

Reproduce with:

    conda create -n fastqpo python=3.12 && conda activate fastqpo
    pip install torch rdkit pandas gpytorch botorch scipy numpy tqdm pathos matplotlib seaborn
    pip install -e <winning repo root>
    QPO_DIR=~/github/qPO python snapshot.py --dataset qm9 --objective gap --seed 7
    python run_sweep.py --snapshot qm9_gap_seed7 --N 500 1000 2000

## What the posterior actually looks like

Reproducing the authors' initialization exactly -- their shuffle, their seeded
draw of 100 molecules, their Tanimoto GP, their marginal-likelihood fit, their
UCB prefilter -- gives a posterior that is **not** low rank in the ordinary
sense. At N = 1000, rank 64 captures 41% of the trace and the off-diagonal
residual flattens near 10%.

That turns out to be the wrong yardstick. Most of the trace is idiosyncratic
variance, which D reproduces exactly by construction; what rank r has to buy is
the off-diagonal. And what the off-diagonal has to buy, in turn, is not itself
but the batch.

Two further features shape every number below.

The probabilities are nearly uniform. At N = 500 the largest is about three
times the smallest, so a randomly chosen batch already scores 0.68 on raw qPO
efficiency. Every efficiency is therefore also reported on a normalised scale
that puts random at 0 and optimal at 1.

The reference is itself Monte Carlo. Every agreement number is accompanied by
the number two independent runs of the reference achieve against each other.
That is the ceiling, and a rank that reaches it has nothing left to prove.

## Phase III: is the fast calculation right?

Fast probit and factor Monte Carlo run on the *same* factor model, so the
covariance error is identically zero and any disagreement is a bug. Because
these probabilities sit near 1/N, the test statistic is the per-candidate
z-score rather than total variation, which Monte Carlo noise would swamp.

At N = 1000 with 4,000,000 factor samples:

| rank | max abs z | mean z | sd of z | verdict |
|---:|---:|---:|---:|---|
| 2 | 2.94 | +0.007 | 0.960 | Monte Carlo noise |
| 8 | 3.01 | +0.004 | 1.034 | Monte Carlo noise |
| 32 | 7.94 | +0.007 | 2.042 | quadrature error, not a bug |

The rank-32 excess is the 32-dimensional quadrature at a fixed 2048 nodes; the
independent-scramble self-error at that budget is 1.4e-2, which is exactly the
observed disagreement. The probability calculation is correct.

## Phase VII: the kill test

QM9, seed 7, N = 500, batch 100, reference 20,000,000 samples. The reference
recovers its own top 100 at 1.00, so the ceiling is 1.00.

| rank | off-diag cov error | eta | normalised eta | top-100 recall | mean batch Tanimoto |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 0.898 | 0.685 | 0.56 | 0.381 |
| **2** | 0.395 | **0.9969** | 0.990 | **0.94** | 0.310 |
| 4 | 0.324 | 0.9982 | 0.994 | 0.95 | 0.309 |
| 8 | 0.259 | 0.9977 | 0.993 | 0.94 | 0.310 |
| 16 | 0.190 | 0.9987 | 0.996 | 0.94 | 0.311 |
| 32 | 0.141 | 0.9978 | 0.993 | 0.92 | 0.316 |
| 64 | 0.112 | 0.9877 | 0.962 | 0.86 | 0.313 |
| 128 | 0.100 | 0.9729 | 0.916 | 0.77 | 0.325 |
| 256 | 0.084 | 0.9090 | 0.718 | 0.57 | 0.312 |

Rank 2 is enough. The decline past rank 32 is the fixed 1024-node quadrature
failing in high dimension, not the model: factor Monte Carlo on the same rank-256
model reaches eta = 0.9989 and recall 0.96. A rank-2 model whose off-diagonal
covariance error is still 40% delivers essentially the whole qPO objective.

The shape replicates. Rank 2 and rank 16, across the three universe sizes:

| N | reference vs itself | rank 2 | rank 16 | rank 16, quotient fit |
|---:|---:|---:|---:|---:|
| 500 | 1.000 / 1.00 | 0.9969 / 0.94 | 0.9987 / 0.94 | 0.9996 / 0.98 |
| 1000 | 1.000 / 0.99 | 0.9956 / 0.92 | 0.9971 / 0.95 | 0.9990 / 0.97 |
| 2000 | 0.9998 / 0.98 | 0.9981 / 0.93 | 0.9992 / 0.95 | 0.9987 / 0.95 |

(eta / top-100 recall.)

## The comparison the paper cares about

The shipped estimator uses 10,000 samples. Spread over N candidates whose
probabilities sit near 1/N, that is about 10,000/N winner counts each, so its
accuracy has to fall as the library grows. It does, and the deterministic
answer does not:

| N | qPO Monte Carlo, 10,000 samples | | fast factor probit, rank 4 | |
|---:|---:|---:|---:|---:|
| | eta | top-100 recall | eta | top-100 recall |
| 500 | 0.935 - 0.957 | 0.66 - 0.72 | 0.998 | 0.95 |
| 1000 | 0.890 - 0.925 | 0.53 - 0.65 | 0.997 | 0.94 |
| 2000 | 0.846 - 0.871 | 0.46 - 0.53 | 0.997 | 0.94 |

Five independent Monte Carlo seeds per row; ties broken by posterior mean
exactly as `acquire_qPO` does, so the comparison is not a straw man.

This is a statement about the shipped budget, not about sampling in general.
Given more samples dense Monte Carlo does reach and then pass the rank-4
answer -- at N = 1000 it matches recall 0.94 at roughly 500,000 samples and
reaches 0.99 at ten million. What the table shows is that the default is far
from that, and moving away from it in the wrong direction as N grows: the same
10,000 draws are spread over more candidates.

The trend is the point. The released implementation prefilters to 10,000
candidates, five times the largest N here, so the shipped method's actual
operating point sits further along this curve than anything measured directly.

Timings in this section come from the sweep, which ran with other jobs on the
machine; the wall-clock comparisons that matter are in the frontier table
below, measured on a quiet machine.

## Does correlation earn its place?

| method | eta | top-100 recall | mean Tanimoto | fraction of pairs > 0.4 |
|---|---:|---:|---:|---:|
| reference (second seed) | 1.000 | 0.99 | 0.315 | 0.123 |
| fast factor probit, rank 4 | 0.997 | 0.94 | 0.311 | 0.120 |
| independence (rank 0, exact) | 0.918 | 0.60 | 0.379 | 0.356 |
| F-LITE | 0.920 | 0.61 | 0.377 | 0.351 |
| A-LITE | 0.915 | 0.60 | 0.380 | 0.362 |
| UCB | 0.865 | 0.49 | 0.392 | 0.435 |
| Greedy | 0.758 | 0.33 | 0.399 | 0.485 |

(N = 1000. F-LITE and A-LITE are Menet, Huebotter, Kassraie and Krause,
AISTATS 2025; the numpy port here matches their released JAX implementation to
1e-6 in float64.)

Correlation is doing real work, and rank 2 keeps essentially all of it. The
independence methods triple the fraction of redundant pairs in the batch, and
admit a 0.949-Tanimoto near-duplicate pair that the correlated batch does not.

F-LITE and A-LITE are approximations to the *independence* answer, which is the
r = 0 row, and they are good ones. Against exact independence they score
total variation 0.010 and 0.006 respectively, with 0.99 top-100 overlap;
A-LITE is the more accurate of the two, as their paper says. The independence
assumption itself costs total variation 0.086 against full qPO, nearly ten
times more. So nothing here is a criticism of LITE: it estimates its target
well, and its target is the thing the factor model replaces. Slepian's
inequality says an independence approximation must be conservative about the
leaders, and that is visible as the redundant-pair fraction.

## Why rank 2 is enough

Posterior correlation is essentially chemical similarity: across candidate
pairs, corr(Tanimoto similarity, posterior correlation) = 0.878.

The first factor is nearly the common mode -- its alignment with the all-ones
direction is 0.959 -- and a factor loading equally on every candidate cannot
change an argmax. That is why fitting in the quotient space (`contrast_factor`,
which truncates P Sigma P with P = I - 11'/N) beats the raw truncation at equal
rank: it does not spend a factor on nothing.

The similarity structure in a prefiltered pool is diffuse rather than a
scattering of isolated duplicate pairs -- only 0.07% of pairs exceed Tanimoto
0.7 -- and diffuse global structure is exactly what a low-rank factor
represents well. A library of tight analogue series would be the hard case, and
is not tested here beyond the random-subset check.

## Does the deterministic calculation earn its place?

Every method turned up until it stops improving, at N = 1000, scored against a
20,000,000-sample reference that recovers its own top 100 at 1.00. Three
comparisons come out of one table, and they are not equally impressive, so they
are worth separating.

| method | budget | seconds | eta | top-100 recall |
|---|---:|---:|---:|---:|
| dense MC-qPO (shipped) | 10,000 | 0.11 | 0.901 | 0.58 |
| dense MC-qPO | 100,000 | 1.11 | 0.985 | 0.84 |
| dense MC-qPO | 1,000,000 | 19.7 | 0.999 | 0.97 |
| dense MC-qPO | 10,000,000 | 148 | 1.000 | 0.99 |
| factor MC, rank 2 | 10,000 | 0.07 | 0.917 | 0.62 |
| factor MC, rank 2 | 1,000,000 | 7.8 | 0.993 | 0.89 |
| factor MC, rank 2 | 10,000,000 | 88 | 0.996 | 0.92 |
| **fast probit, rank 2** | **32 nodes** | **0.10** | **0.997** | **0.93** |
| fast probit, rank 4 | 64 nodes | 0.22 | 0.996 | 0.93 |
| fast probit, rank 4 | 1024 nodes | 3.3 | 0.997 | 0.94 |

**Against factor Monte Carlo on the same model, the margin is enormous.** Both
carry the identical rank-r covariance error, so this isolates the probability
calculation: 32 nodes at 0.10 s match what 10,000,000 samples need 88 s to
reach. That is 880x at rank 2 and 620x at rank 4.

**Against the shipped default at equal wall time, the margin is decisive.**
0.10 s of deterministic rank-2 buys top-100 recall 0.93; 0.11 s of the shipped
estimator buys 0.58.

**Against dense Monte Carlo given unlimited budget, the margin is modest, and
in one direction it runs the other way.** The fast method plateaus at the
accuracy its rank affords -- rank 2 stops at recall 0.92, rank 4 at 0.94 --
while dense sampling keeps climbing to 0.99 if you pay 148 s. Reaching recall
0.93 costs dense Monte Carlo about 500,000 samples, roughly 8 s, against 0.10 s;
past that, higher rank rather than more nodes is what buys accuracy. At
N = 1000 dense Monte Carlo is therefore beaten but not embarrassed. Its
difficulty is not this N. It is that its cost per sample is O(N^2) and the
covariance must exist at all.

### Where the deterministic advantage comes from

Same rank-4 model at N = 1000, relative error by probability band:

| method | time | bottom decile | middle | top decile | exact zeros |
|---|---:|---:|---:|---:|---:|
| factor MC, M = 10^4 | 0.2 s | 0.267 | 0.221 | 0.135 | 2 |
| factor MC, M = 10^6 | 16 s | 0.028 | 0.021 | 0.019 | 0 |
| factor MC, M = 10^7 | 112 s | 0.010 | 0.007 | 0.005 | 0 |
| **fast probit, Q = 64** | **0.5 s** | **0.008** | **0.005** | **0.005** | 0 |
| fast probit, Q = 1024 | 7.5 s | 0.001 | 0.001 | 0.000 | 0 |

The margin is widest in the low-probability band, which is the structural
point. Counting winners gives a
candidate a 0 or a 1, so its relative error scales as 1/sqrt(M p_i); the
deterministic conditional integral gives every candidate a smooth positive
number at every node, and its error does not depend on p_i at all.

The same reasoning says the advantage grows with N: holding total variation
fixed costs the sampler M proportional to N, while the quadrature error is
governed by the factor dimension, not by N.

## Two changes to the lattice kernel

Both measured rather than assumed, and both checked against
`winning.factor.core.win_probabilities_factor`, which this kernel is a retuned
copy of.

**An exact adaptive window.** Summing the integrand over candidates telescopes:
sum_i phi_i(x) prod_{j != i} Phi_j(x) is the derivative of prod_j Phi_j(x), the
density of the maximum. So the integration window is that density's bulk, and
the omitted mass is exactly G(x_lo) + (1 - G(x_hi)) = 2 delta -- a bound, not an
estimate. Both endpoints come from vectorised bisection on monotone functions.
On the molecular posteriors this is 2.4x narrower than a conservative window.

**A much smaller lattice.** With that window the integrand vanishes at both
endpoints, so the rectangle rule has no boundary term and converges
spectrally. Measured on all three real posteriors: 65 points give total
variation 1.6e-14 against 4097 points. A stress test holds this up to a
1000-fold spread in marginal standard deviations. The default is 129.

## Phase VIII: the factor model without Sigma

The eigen-oracle above is deliberately not an algorithm -- it needs the N x N
covariance, which is the thing being avoided. `factorgp.py` produces the same
object from the GP directly. The Tanimoto posterior

    Sigma_* = s^2 K_** - s^4 K_*n A^{-1} K_n* + sigma^2 I

has a rank-n data correction and a diagonal noise term already; only the prior
block is genuinely N x N, and a Nystrom approximation on r_z inducing molecules
handles it. The whole thing becomes B S B' + sigma^2 I with B of width r_z + n,
whose top eigenpairs come from a small Gram matrix in two streaming passes --
nothing of size N x N or even N x (r_z + n) is ever stored. D is then set from
the **exact** marginal variances, which cost O(N n^2), so the model reproduces
every candidate's posterior variance exactly and approximates only the
off-diagonal dependence, the same contract the oracle offers.

Anchors: the numpy posterior matches gpytorch's `mean_cov_from_gp` to 1e-11 on
Sigma and 1e-9 on mu; the streaming route matches the dense QR route to 1e-14;
the Nystrom factor model converges to the oracle as inducing points are added.

## The whole library

QM9 seed 7, all 133,702 unacquired candidates. The dense covariance the
released pipeline would need is 143 GB, so this comparison does not exist in
the current implementation at any budget.

| method | candidates scored | seconds | oracle mean of batch | oracle top-10 | in true top 1% | batch Tanimoto |
|---|---:|---:|---:|---:|---:|---:|
| prefiltered MC-qPO, 10,000 samples | 10,000 | 30 - 35 | 0.3218 - 0.3244 | 0.3550 - 0.3608 | 8 - 11 | 0.297 - 0.316 |
| prefiltered fast qPO, rank 2 | 10,000 | 29 | 0.3293 | 0.3653 | 15 | 0.336 |
| prefiltered fast qPO, rank 4 | 10,000 | 24 | 0.3292 | 0.3657 | 15 | 0.345 |
| full-library fast qPO, rank 2 | 133,702 | 232 | 0.3337 | 0.3722 | 23 | 0.375 |
| full-library fast qPO, rank 4 | 133,702 | 161 | 0.3337 | 0.3722 | 23 | 0.380 |
| full-library F-LITE | 133,702 | 0.1 | 0.3339 | 0.3722 | 24 | 0.375 |
| full-library UCB | 133,702 | 0.0 | 0.3336 | 0.3670 | 22 | 0.392 |
| full-library Greedy | 133,702 | 0.0 | 0.3339 | 0.3659 | 24 | 0.399 |

Three readings, of decreasing strength. The third is the one that stops this
table being oversold.

**The capability is real.** Correlation-aware qPO over 133,702 candidates runs
in about three minutes and 2 GB, against a covariance that would need 143 GB.
That is a new operating point, not a speed-up of an existing one.

**Holding the pool fixed, the deterministic estimator picks better batches.**
On the same 10,000 candidates it finds 15 members of the true top 1% against
8 - 11 for the sampler, and a better oracle top-10. This is the accuracy result
of the previous sections showing up in molecules.

**But a single batch cannot separate the acquisition functions, and this one
does not.** Greedy and F-LITE score 24 top-1% hits, full-library qPO 23, UCB 22.
That is the expected outcome, not an anomaly: one round of immediate batch
value is precisely what Greedy maximises, and the case for dispersion is that
it pays over later rounds. A single-round oracle comparison is the wrong
instrument for comparing acquisition functions, and the closed loop below is
the right one. Reporting this row as a win for qPO would be wrong.

One more thing worth recording: every method's batch lay entirely inside the
top 10,000 by upper confidence bound, so at this iteration the prefilter
discards nothing that anyone wanted. The full-library gain over the
prefiltered run of the *same* method therefore comes from computing the
probability against the true field of 133,702 competitors rather than a
truncated field of 10,000 -- not from access to molecules the prefilter hid.

## Five posteriors, two datasets

The rank result was measured on one posterior from one seed on one dataset,
with a candidate pool chosen by the prefilter it is meant to replace. Four
more posteriors test whether any of that mattered. Top-100 recall, N = 1000:

| posterior | reference vs itself | shipped MC, 10k | r = 0 | r = 2 | r = 4 | r = 16 |
|---|---:|---:|---:|---:|---:|---:|
| QM9 seed 7, UCB pool | 0.99 | 0.60 | 0.60 | 0.92 | 0.94 | 0.95 |
| QM9 seed 11, UCB pool | 0.98 | 0.66 | 0.71 | 0.85 | 0.90 | 0.95 |
| QM9 seed 7, **random** pool | 1.00 | 0.95 | 0.98 | 0.98 | 0.99 | 0.99 |
| QM9, posterior at **round 10** | 0.99 | 0.78 | 0.94 | 0.98 | 0.99 | 0.99 |
| antibiotic screen, UCB pool | 0.97 | 0.44 | 0.55 | 0.90 | 0.92 | 0.94 |

Three things come out of this.

**Low rank is not an artefact of a barely-trained model.** The round-10
posterior sits on 1,100 observations rather than 100, and it is *easier*, not
harder: rank 2 reaches 0.98.

**Correlation matters precisely where the method operates.** In a randomly
drawn candidate pool the molecules are chemically unlike each other,
independence is nearly right, and r = 0 already scores 0.98 -- the fraction of
redundant pairs in its batch is 0.03, so there is no redundancy to remove. In
a UCB-prefiltered pool the candidates are similar by construction, r = 0 falls
to 0.60, and its redundant-pair fraction is 0.36 against 0.12 for rank 2. The
prefilter creates the very homogeneity that makes correlation load-bearing.

**The shipped sampler is worst on the harder dataset.** On the antibiotic
screen, where fitted noise (8.2e-3) is most of the outputscale (1.0e-2), its
top-100 recall falls to 0.44 at N = 1000 and 0.29 at N = 2000, while rank 4
holds at 0.92 and 0.84.

## The closed loop, and what it does not show

The authors' setup unchanged: QM9, gap, 100 random molecules to start, batches
of 100, twenty rounds, the GP refitted by marginal likelihood every round, ten
seeds. The library optimum is a top-10 average of 0.4655 and a top-100 average
of 0.3969, so both are ceilings, and the discriminating question is how many
rounds it takes to reach them.

| method | rounds to 99% of optimal top-10 | final top-10 ave | % of optimal | final top-1% | seconds per acquisition |
|---|---:|---:|---:|---:|---:|
| F-LITE, full library | 12.7 +- 2.5 | 0.4654 | 99.99 | 0.630 | 0.09 |
| qPO-fast rank 4, 10k | 13.0 +- 2.4 | 0.4654 | 99.99 | 0.631 | 6.6 |
| qPO-fast rank 4, full library | 13.3 +- 2.3 | 0.4654 | 99.99 | 0.629 | 58.2 |
| qPO-MC-10k (shipped) | 14.3 +- 2.5 | 0.4654 | 99.99 | 0.620 | 17.8 |
| UCB | never within 20 | 0.4567 | 98.12 | 0.664 | 0.009 |
| Greedy | never within 20 | 0.4456 | 95.71 | 0.650 | 0.009 |

**The qPO idea is vindicated and the paper's claim reproduces, clearly.** Every
method that computes something like a probability of maximality reaches 99.99%
of the library-optimal top-10. UCB reaches 98.1% and Greedy 95.7%, and neither
gets to 99% inside twenty rounds, on any seed.

**The shipped sampler costs about a round and a half, at the edge of
detectability.** Paired across the same ten seeds, the shipped estimator takes
1.3 rounds longer than fast rank 4 on the same 10,000 candidates (p = 0.15),
1.0 rounds longer than the full-library version (p = 0.15), and 1.6 rounds
longer than F-LITE (p = 0.03). So the snapshot-level accuracy does show up,
faintly, in the direction predicted -- and nothing here would survive a
demand for a large effect.

**Correlation does not pay off at all on this benchmark.** F-LITE assumes
independence, costs 0.09 s against 58 s, and is nominally the *fastest*
converging method of the six. Whatever the correlated model is buying in batch
composition at the snapshot level, twenty rounds of QM9 gap cannot see it.

The benchmark saturates, which is the honest reason. Every method recovers all
thirteen members of the true top 0.01% by round 15, and twenty rounds acquires
1.6% of the library. A task that everything solves cannot separate the things
that solve it.

Note also that UCB recovers the largest fraction of the true top 1% (0.664)
while finishing well short on top-10. Those are different objectives: qPO
concentrates on the very best molecules, UCB spreads across the good ones. The
paper's metrics are the top-k averages, and on those qPO wins.

### The antibiotic screen reverses it

Ten rounds of 50 on the antibiotic screen, five seeds, minimising growth. Here
the **shipped Monte Carlo estimator is the best method**, and the deterministic
ones are significantly worse (paired p = 0.05 and 0.02 for fast rank 4 on 10k
and on the full library). Its top-10 average is -0.1004 against a library
optimum of -0.0931; the deterministic versions reach -0.1099 to -0.1180, worse
than Greedy.

That is the opposite of what accuracy predicts, and it should not be explained
away. The plausible mechanism is that on a posterior where the fitted
likelihood noise is 82% of the outputscale, exact optimality probabilities
concentrate on high-variance candidates, and the sampler's noise acts as
useful extra exploration -- being wrong helps. Five seeds, ten rounds and one
dataset is thin evidence either way; ten more seeds are running.

Either way it makes the same point as the QM9 result from the other side:
**estimator accuracy and closed-loop discovery are not the same axis**, and
this experiment has established the first without establishing the second.

## Why a 40%-wrong covariance is 99% right about the batch

The rank ladder leaves something unexplained: at rank 2 the off-diagonal
residual is 35-40% of the off-diagonal Frobenius norm, and the decision is
99.6% right. Frobenius norm is evidently not the quantity that governs the
error. What is?

**Plackett's relation.** The Gaussian density satisfies a diffusion equation in
the covariance, so for any probability under N(mu, Sigma),

    d p / d Sigma_ij = d^2 p / d mu_i d mu_j   (i != j),
    d p / d Sigma_ii = (1/2) d^2 p / d mu_i^2.

Both cases collapse into one statement: for any symmetric perturbation,

    d/dt p(Sigma + t Delta) = (1/2) <Delta, H>,   H = Hessian of p in the means.

Verified here to a few percent against finite differences, including that the
one-half on the diagonal is really there (without it the prediction doubles),
and that the error is genuinely first order in the size of the perturbation.
Getting that check honest needed perturbations that stay exactly inside the
factor family; perturbing one covariance entry and evaluating with a
rank-(r+1) model measures the truncation, not the derivative.

**What it says about the factor construction.** Because D_r reproduces
diag(Sigma) exactly, the residual has zero diagonal, and the error is the
residual contracted against the Hessian of the win probabilities. H is
supported on pairs of candidates that are jointly in contention for the lead:
move the covariance of two candidates that never threaten the maximum and
nothing happens. Frobenius norm weights all N(N-1)/2 pairs equally; the argmax
weights the few that compete. **Argmax effective rank and covariance effective
rank are different quantities, and the first is governed by an H-weighted norm.**

Measured on the real posterior (QM9 seed 7, top 200 by UCB, all 200
eigendirections of the residual used, reference 2e7 samples with a seed-to-seed
total variation of 0.0018):

| rank | Frobenius ratio | predicted TV | measured TV | correlation |
|---:|---:|---:|---:|---:|
| 2 | 0.347 | 0.0122 | 0.0160 | +0.71 |
| 4 | 0.279 | 0.0122 | 0.0145 | +0.82 |
| 8 | 0.226 | 0.0080 | 0.0107 | +0.77 |
| 16 | 0.176 | 0.0059 | 0.0138 | +0.59 |

First order gets the size to within about 25% and correlates 0.6-0.8 with the
actual error, for a perturbation that is nowhere near small. The number to
quote is that a residual worth 35% of the covariance is worth 1.2% of the
decision.

Scaling the same residual down confirms it is genuinely the derivative. Along
the path Sigma_2 + t Delta, with both ends of the difference drawn from the
same random stream so the sampling error cancels:

| t | measured TV | predicted TV | correlation |
|---:|---:|---:|---:|
| 0.05 | 0.00068 | 0.00061 | +0.92 |
| 0.10 | 0.00127 | 0.00122 | +0.97 |
| 0.25 | 0.00306 | 0.00304 | +0.97 |
| 0.50 | 0.00632 | 0.00608 | +0.92 |

The predicted and measured sizes agree to 4% at t = 0.1 and to under 1% at
t = 0.25. The relative error on the full vector is U-shaped in t -- Monte Carlo
noise dominates below t = 0.1, curvature above t = 0.5 -- which is exactly the
signature of a correct first-order term. Common random numbers matter here: an
earlier version differenced independent runs, and at small t the true change
was smaller than the sampling noise, so the check reported nothing where it
should have looked best.

A warning for anyone repeating this: the residual's spectrum is nearly flat, so
truncating it to a few dozen eigendirections discards most of it and produces
a prediction that is five times too large and *negatively* correlated with the
truth. The first version of this experiment did exactly that.

## A negative result: one-factor conditional envelopes

The construction above conditions on r factors and integrates the N
idiosyncratic dimensions exactly, which needs Sigma to be low rank plus
diagonal. The natural complement conditions on the residual and integrates ONE
direction exactly, which needs no low-rank assumption at all: with
Sigma_c = b b' + R, the race conditional on eta ~ N(0, R) is the upper envelope
of N straight lines in z, so the conditional shares are exact segment
probabilities Phi(tau_k) - Phi(tau_{k-1}), and the conditional Jacobian is a
Laplacian on a **path** -- only adjacent envelope segments touch.

That is all implemented (`envelope.py`, 16 anchors: hull against brute force,
conditional shares against direct simulation of Z, the Laplacian against finite
differences with common random numbers, J1 = 0, positive semidefiniteness, and
the path structure). It works. It is not useful.

| covariance | leading eigenvalue's share of the trace | variance reduction per draw |
|---|---:|---:|
| spectrum k^-0.25 | 0.014 | 1.09 |
| random dense | 0.015 | 1.23 |
| spectrum k^-0.5 | 0.037 | 1.19 |
| condition number 1e4 | 0.045 | 1.57 |
| factor, r = 20 | 0.054 | 1.38 |
| clustered | 0.122 | 1.25 |
| factor, r = 5 | 0.156 | 1.67 |
| spectrum k^-1 | 0.171 | 2.00 |
| spectrum k^-2 | 0.611 | 3.39 |

Correlation between the two columns: **0.968**. The reason is not subtle once
stated. Conditioning on eta leaves only the variance along b unresolved, so if
b carries a fraction f of the trace the conditional winner is nearly determined
and q(eta) is nearly an indicator, whose variance is what counting winners
already had. For a dense covariance with a flat spectrum f is O(1/n) and there
is nothing to reclaim. Meanwhile the envelope costs 30-45x more per draw than
counting. It fails falsification criterion 1 of its own brief.

**The general principle, which is the useful part.** Rao-Blackwellising a
subspace pays in proportion to the share of the winner-relevant variance that
subspace carries. So condition on the *small* subspace and integrate the *big*
one. The factor probit integrates N of N+r dimensions exactly and quadratures
r; the envelope integrates 1 of N and samples the rest. That asymmetry, not
any implementation detail, is why one works and the other does not -- and it
is the same reason the Rao-Blackwellised dense reference earlier in this
directory won 4.3x per draw and lost 400x per second.

### Cross-check against research/shifted_cv

`../shifted_cv/` attacks the same general-covariance inversion problem far
more thoroughly, on 92 problems across ten covariance families, using this
directory's envelope algorithm as its kernel. The two implementations -- numpy
here, numba there -- agree on conditional shares to 2.2e-16, which makes each
an independent check on the other.

Where they overlap they agree. Rao-Blackwellisation alone gives median
variance-reduction factors of 1.59 / 1.38 / 1.75 at n = 50 / 250 / 1000 there,
against 1.1-3.4 here, and dense and near-singular covariances sit at ~1 at
every rank tried in both.

What that work adds is a **shifted control variate**: couple each draw to a
surrogate race whose location is calibrated so its winner shares equal the
target exactly, and Monte Carlo only the discrepancy. On factor-structured
covariance that reaches variance-reduction factors of 10^3, and it obeys an
exact and very useful identity,

    VRF = (1 - sum_i p*_i^2) / (2 (1 - P(W = V))),

confirmed here against their `agreement.csv`: correlation between predicted and
measured log-VRF is 0.9967 with a median ratio of 1.002 over 2,976 rows. One
scalar -- how often the two races pick the same winner -- determines the whole
thing, and a pilot measures it cheaply.

**The synthesis, which is visible only by putting the two together.** Read
their `inversion.csv` against the deterministic surrogate each run starts from
(`rmse_mu0 / rmse_mu`, above one meaning Monte Carlo helped):

| family | VRF of the best coupled estimator | surrogate's own share-L1 | best MC gain at M = 4096 |
|---|---:|---:|---:|
| factor | 1199 | 0.018 | 1.24 |
| clustered5 | 161 | 0.021 | 1.01 |
| spectral1 | 7.3 | 0.029 | 1.22 |
| clustered | 3.3 | 0.024 | 1.54 |
| dense | 1.6 | 0.031 | 1.17 |

The variance reduction is **anti-located with respect to where it is needed**.
It is spectacular on factor covariance, where the deterministic lattice
inversion is already at the reference's own noise floor and there is nothing
left to win; it is absent on dense covariance, where there would be something
to win. Across all families the deterministic surrogate alone reaches share-L1
0.018-0.031 with zero Gaussian draws and no failures in 1,164 inversions, and
the best Monte Carlo refinement anywhere improves the ability estimate by 1.5x
at most. Raw Monte Carlo has a median gain of 0.93-1.00 -- it never helps, and
at M = 64 on dense covariance it makes the answer four times worse.

So two independent attacks on general dense Sigma, one-factor
Rao-Blackwellisation here and calibrated shifted controls there, both land at
~1x. That is not a null result about two methods; it is evidence for the
dichotomy the whole program rests on. Arbitrary covariance carries N^2 of
information and behaves accordingly. Factor covariance is a genuinely
different regime, and in that regime the deterministic calculation is not a
faster route to the answer -- it is the answer, and sampling adds nothing.

## What qPO is actually optimising, and the selection rule it should use

The antibiotic reversal above turned out not to be about estimator noise at
all. Chasing it down led to the sharpest result in this directory.

### The paper's objective is additive, so the SELECTION cannot see the batch

qPO states its objective explicitly (Fromer et al., arXiv:2410.06333, Eqs 3
and 5): choose the batch maximising `Pr(x* in X_acq)`, and note that
`Pr(x* in X_acq) = sum_{x_i in X_acq} Pr(x* = x_i)` because those events are
disjoint. That identity is what makes the batch problem tractable -- and it
is also the whole story about diversity. **An additive objective has no
interaction between batch members**, so the argmax over batches is exactly
the top b by `p_i`, and the selection step is blind to redundancy. The
diversity qPO exhibits is entirely a property of the MARGINALS (correlated
candidates split their optimality mass), never of the choice. The paper's
claim that it "naturally captures diversity through model covariance" is
true in that restricted sense, but the covariance enters only through `p_i`,
and cannot be traded against value at selection time.

The paper is also explicit that it declines an iterative objective:

> "We pursue a batch acquisition strategy that is purely exploitative,
> optimizing expected performance in the immediate iteration as if the
> optimization could be stopped at any time."

So the iterative procedure has no stated objective: a one-shot criterion is
iterated, and everything sequential (exploration, information, the value of
the posterior update) happens by accident. That is the right thing to know
before reading any closed-loop result, including the ones above.

### The shipped sampler is accidentally doing Thompson sampling

The paper distinguishes itself from parallel Thompson sampling *by
determinism*: "qPO aims to choose candidates deterministically that maximize
Pr(x*=x_i)". Its own default implementation does not deliver that. Each Monte
Carlo draw's argmax IS a draw from `p(x*)`, and with M = 10,000 samples over
N = 10,000 candidates the counts are Poisson with mean ~1, so "top b by
count" is a tempered draw from `p(x*)` rather than its top b modes.

On the antibiotic posterior `p(x*)` is nearly uniform (max/mean = 3.25,
effective support 7,437 of 10,000), which makes the two rules maximally
different. The 2x2 that separates probability accuracy from selection
stochasticity (10-15 seeds, paired against the shipped sampler):

| arm | probabilities | selection | final top-10 | paired p |
|---|---|---|---|---:|
| qPO-MC-10k (shipped) | noisy | top-b of noisy counts | **-0.1025** | -- |
| qPO-fast-r4-thompson | **exact** | sample proportional to p | -0.1039 | 0.49 |
| qPO-fast-r4-10k | **exact** | top-b (as specified) | -0.1098 | **0.028** |
| Random-plausible | none | uniform over the pool | -0.1348 | **0.0004** |

**Stochastic selection is the whole of the shipped sampler's advantage.**
Thompson on exact probabilities matches it (p = 0.49) at a fraction of the
cost; the deterministic rule the paper specifies is significantly worse
(p = 0.028); and a random batch is catastrophic (p = 0.0004), which rules out
"noise as exploration" in the loose sense. The estimator was never the
problem -- the selection rule was, and the released implementation's
inaccuracy was quietly protecting it from its own specification.

### Explicit diversity beats accidental diversity

If the batch is worth choosing jointly, the objective should say so.
`E[max_{i in B} Y_i]` is submodular, so greedy is within 1 - 1/e, and a
candidate correlated with one already chosen adds little because it raises
the max only where the batch is already high. It also costs nothing new
structurally: conditional on the factor the candidates are independent, so
the batch's max has CDF `prod_{i in B} F_i(x|f)` -- the same field object the
ability transform builds. **The exotics cavity DIVIDES that field to remove a
competitor; batch selection MULTIPLIES one more CDF in to add a member**, and
one greedy step costs the same O(N L Q) as one qPO board
(`pom.greedy_expected_max`).

Antibiotic snapshot, N = 10,000, batch 50 (higher is better):

| selection rule | best molecule in batch | batch top-10 | Tanimoto |
|---|---:|---:|---:|
| **greedy E[max] (explicit)** | **-0.104** | **-0.643** | 0.099 |
| shipped MC (accidental) | -0.129 | -0.658 | 0.075 |
| top-b by qPO | -0.186 | -0.759 | 0.085 |
| Thompson on exact p | -0.275 | -0.735 | 0.076 |
| random from plausible | -0.268 | -0.711 | 0.089 |

Greedy E[max] beats every rule on the quantity a batch is for, including the
shipped sampler, using exact probabilities and 31/50 different molecules from
top-b. The closed-loop run of this arm is in
`results/closed_loop_abx_greedymax.csv`.

The synthesis: this directory made qPO computable exactly, and the exact
answer then exposed that `p(x*)`'s top-b modes are the one use of `p(x*)`
with no decision-theoretic backing. Sampling from it is Thompson; reducing
its entropy is entropy search; valuing the batch directly is E[max]. All
three are principled and all three beat top-b where the field is flat.


### Closed-loop postscript: neither corrected rule survives the benchmark

Reported for completeness, and explicitly not a claim of this work. Ten more
seeds of the antibiotic closed loop, final top-10 average (library optimum
-0.0931):

| method | final top-10 |
|---|---:|
| qPO-MC-10k (shipped) | -0.1025 |
| qPO-fast-r4-thompson (exact p, stochastic selection) | -0.1039 |
| F-LITE-full | -0.1058 |
| Greedy | -0.1073 |
| UCB | -0.1075 |
| qPO-fast-r4-full | -0.1086 |
| **qPO-fast-r4-greedymax (E[max], explicit diversity)** | **-0.1094** |
| qPO-fast-r4-10k (exact p, top-b) | -0.1098 |
| Random-plausible | -0.1348 |

Greedy E[max] wins the SNAPSHOT decisively (best molecule in batch -0.104
against the sampler's -0.129) and then does not convert it, landing at
-0.1094 over ten seeds. That is the third time in this directory that
snapshot accuracy has failed to become closed-loop discovery, and it is the
reason the paper claims arithmetic rather than acquisition: on ten rounds of
a noisy screen, seed-to-seed spread swamps every method difference except
the collapse of the random arm.

## Decision rule

The brief set three conditions, all of them about the snapshot experiment. All
three pass.

1. **Low rank is genuine.** Rank 2 preserves eta > 0.995 and rank 4 reaches
   0.998, far inside the r <= 64 that would have made the application
   compelling. This is the kill test, and it did not kill.
2. **Correlation is useful.** The correlated model beats exact independence and
   both LITE variants by roughly 8 points of eta and 34 points of top-100
   recall, and cuts redundant pairs in the batch by a factor of three.
3. **The deterministic calculation matters.** On the same factor model it is
   620-880x faster than well-implemented low-rank Monte Carlo at matched batch
   accuracy, and at equal wall time it beats the shipped estimator's top-100
   recall by 0.93 against 0.58. Against dense Monte Carlo with an unlimited
   budget the advantage is smaller and bounded by the rank; dense sampling's
   real problem is O(N^2) per draw and needing the covariance to exist at all.

A fourth condition, which the brief did not set but which decides whether any
of this matters to a chemist, currently **fails to be demonstrated**: none of
the accuracy translates into better closed-loop discovery on QM9 gap. Every
probability-of-maximality method, including the shipped sampler and including
independence-based F-LITE, reaches 99.99% of the library-optimal top-10 by
round 15. The benchmark saturates before the estimators separate.

So the honest summary is a computational result with an unproven application.
What is established: qPO is computable exactly and deterministically, at rank
2 to 4, in linear time and memory, over a whole molecular library, and more
accurately than the released sampler at equal cost. What is not established:
that anyone gets better molecules because of it. Whether the second follows
from the first needs a benchmark that does not saturate.

## Status

In `results/`, assembled by `collect_results.py` into `all_results.csv`:
rank ladders at N = 500/1000/2000 for QM9 seed 7, the frontier at N = 1000,
the full-library run, and the twenty-round closed loop at three seeds.
Running: runtime scaling to N = 250,000, the frontier at N = 10,000, seven more
closed-loop seeds, the antibiotic-screen closed loop, a random (unprefiltered)
candidate subset, a mid-run posterior from round 10, the second QM9 seed, and
the antibiotic screen.


## Note (2026-08-26): pom_fast deviates 2.2e-3 from the reference

Arbitrated while porting the Schur kernels to Rust: `fastrace`'s factor
kernel matches `winning.factor.core.win_probabilities_factor` to 1.3e-16;
this directory's numpy `pom_fast` sits TV 2.2e-3 from both, constant in
lattice points (so a windowing/convention difference, not resolution). Every
effect reported here is 5x-500x larger than 2.2e-3, so no conclusion
changes; but for reference-grade numbers use fastrace or winning.factor,
and the difference deserves a diagnosis before pom_fast is reused elsewhere.
