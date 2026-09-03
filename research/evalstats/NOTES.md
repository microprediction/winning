# The statistics of evals: field map and where we fit
(Synthesis of three reconnaissance agents, 2026-09-03. Full reports
in the session transcripts; locators below are agent-verified at
abstract level unless noted. Prompted by the pass@k note being
scooped three times in one day -- the estimation layer is crowded,
and this map says where it is not.)

## What is going on
"Statistics of evals" is crystallizing into a subfield with a venue
(CTB@ICML 2026, Seoul, white-paper ambitions) and three fronts:
- PREDICTION (Stanford/Koyejo arc): Brown 2407.21787 -> Schaeffer
  2502.17578 -> Kazdan 2510.05197. Observe the coverage law, explain
  it by difficulty distributions, estimate it cheaply.
- DECISION RULES (Case Western 2510.04265; Paris/Verine CTB'26; UK
  AISI 2608.14425): Bayesian posteriors and stopping heuristics.
- COVERAGE AS TRAINING SIGNAL (MSR/Princeton: TailSFT 2608.25756;
  also Where-to-Spend-Rollouts 2605.07114).
Adjacent 2026 arrivals: winner's-curse correction for adaptive
benchmarking (2605.05973), correlation/modal ceilings for test-time
scaling (2606.28661), repeated-inference safety statistics
(2602.11786), instance-optimal judge budgeting (CTB 68333).

## The three converging gaps
1. A-VS-B EVAL STOPPING UNDER SHARED PROMPTS: GENUINELY EMPTY. The
   AISI stopping paper (2608.14425) stops single-model evals on CI
   width, is not optimal stopping despite its title, and explicitly
   has no comparison protocol; industrial group-sequential tests are
   the only incumbent. Shared prompts are common random numbers;
   prompt difficulty is the shared factor; stopping on posterior
   probability-of-best under a factor-Gaussian posterior is exactly
   the exp1_stopping machinery of papers/exact_pom, measured
   synthetically already. Data to replay: HELM per-instance logs,
   Open LLM Leaderboard details, Kazdan's per-problem counts.
   Experiment: two adjacent checkpoints, samples-to-decision at
   matched error vs group-sequential and vs full-run.
2. CORRELATED ATTEMPTS AND FACTOR-COUPLED COVERAGE: EFFECTIVELY
   OPEN. Bay-Yearick (2606.28661) name a correlation ceiling but
   their rho is the exchangeable mixture's intraclass correlation --
   the same conditionally-iid class as everyone else; no
   non-exchangeable within-prompt dependence model exists (caching/
   prefix/decoding artifacts are documented as violations, 2511.22118,
   never modeled); Gaussian copulas appear only ACROSS models
   (2602.08003, ensemble error floors). No factor-coupled predictor
   of coverage curves across prompts exists (IRT estimates abilities,
   2510.05709 clusters prompts for uncertainty, neither predicts
   pass@k). The sharpest experiment (agent-specified): batch the
   generation with shared artifacts, test within-problem batch
   overdispersion (falsifies exchangeability), then fit exchangeable
   vs factor-correlated probit on k<=16 and predict the k=100-1000
   curve, where the exchangeable model over-predicts coverage. This
   is also precisely the "correlated, heterogeneous attempt
   portfolios" bar the pass@k note's reviewer set for a distinct
   contribution.
3. NON-MYOPIC EVALUATION-BUDGET ALLOCATION: NAMED INCUMBENT.
   Kazdan's Algorithm 1 is greedy min-success-count with no stopping
   rule, and their own words: "it is generally difficult to analyze
   the effect of such adaptive schemes in a Bayesian context."
   Knowledge-gradient on their own beta-binomial posterior is the
   obvious upgrade, their Theorem 1 the oracle benchmark, their
   released per-problem records the data.
   Crowded lane to AVOID: per-prompt inference stopping for
   self-consistency (2602.05395, 2305.11860 and successors).

## Where we fit, in one sentence each
(1) is exact_pom's stopping experiment replayed on real logs -- the
empty lane, fastest to a result. (2) is the substantive paper: the
extremal-portfolio/factor machinery applied where every incumbent
assumes exchangeability and two papers name the violation without
modeling it. (3) is a clean head-to-head with a named incumbent and
doubles as the VOC/knowledge-gradient demonstration Tuisov's
framework prices. Priority in that order; (1) and (3) share data and
posterior code with the pass@k note, and (2) is where the November
window matters least because the machinery barrier is highest.
