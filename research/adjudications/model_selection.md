# Adjudication: model/prompt/agent selection (Track E)
(Agent report, 2026-09-01. Verdict: PURSUE -- flagship demo.)

## Incumbent practice
- Leaderboard uncertainty is BT + bootstrap: Chatbot Arena/LMArena
  fits Bradley-Terry by MLE, 95% CIs by ~1000 bootstrap resamples of
  votes; nobody reports P(best) -- practitioners eyeball overlapping
  intervals.
- MODEL SELECTOR (Okanovic et al., AISTATS 2025, arXiv:2410.13609,
  github.com/RobustML-Lab/model-selector) is EXACTLY our object: a
  Bayesian posterior P(model j best | labels) with greedy
  information-gain labeling, up to 94% label savings, 1500+ models /
  18 collections. CRITICAL WEAKNESS: conditional independence of
  model correctnesses given the best (naive Bayes, single epsilon)
  -- no error correlation between models. That is the hole a
  factor-covariance race fills.
- LLM-judge uncertainty: arXiv:2505.15240 (uncertainty-guided
  comparison selection, ~half the comparisons -- directionally
  confirmed, figure not re-derived); PAIRS (arXiv:2403.16950); SCOPE
  (arXiv:2602.13110). All pairwise/BT; none races N models jointly.
- Closest academic competitors: arXiv:2606.08679 (hierarchical
  conformal rank intervals, models correlations) and arXiv:2607.16259
  (pairwise-test rank CIs on MMLU). Both produce INTERVALS, not
  P(best) vectors; neither does removal counterfactuals.

## Correspondence
Exact. S_ir = mu_i + v_i' F_r + eps_ir with shared questions/judges
as factors; "which model is truly best" is factor-race PoM; router
failover is the removal counterfactual; softmax-renormalize-on-
failover is the Luce move. MODEL SELECTOR's posterior is the
independent-race special case.

## Leverage (honest)
Bootstrap over QUESTIONS does capture shared-question correlation --
resampling rows of the N x R matrix preserves cross-model
dependence. "We model correlation, bootstrap doesn't" is FALSE and
must not be the pitch. Real leverage:
(a) REMOVAL COUNTERFACTUALS (strongest): bootstrap re-prices a
    survivor field only by rerunning the whole bootstrap per
    deletion; winning gives all N re-pricings exactly, and shows
    Luce renormalization misprices failover when the removed model's
    errors correlate with a survivor's. No incumbent does this.
(b) SMALL-P(BEST) RESOLUTION: at B=1000, P(best)=0.003 is 3 counts
    -- noise. Exact computation resolves tail candidates (sequential
    elimination, routing tails).
(c) SEQUENTIAL DESIGN (strong but contested): correlation-aware
    information gain vs MODEL SELECTOR's independence, on their own
    public data. Contested: active-selection papers now numerous
    (e.g. arXiv:2510.09418).

## Kill risks / data
- Bootstrap is standard, simple, assumption-free; the Gaussian
  factor model is parametric on binary/bounded per-item scores
  (probit-style link needed; reviewers will poke).
- Data fine: MODEL SELECTOR repo ships prediction matrices in
  resources/datasets (on-disk format unverified -- clone to check);
  HELM per-instance stats public on GCS bucket crfm-helm-public
  (per_instance_stats.json per run); HF Open LLM Leaderboard
  details_* datasets (~87.9M responses) downloadable though
  archived. Chatbot Arena is pairwise battles only -- skip.
- Router realism: production routing weighs latency/cost/context;
  frame as the quality term inside any router objective.

## Decisive demo
MODEL SELECTOR's own matrices (fallback HELM). One figure, two
panels: (left) exact P(best) under factor covariance vs their
independence posterior vs question-bootstrap at matched budget --
bootstrap's zero-count tail candidates vs exact small probabilities;
(right) failover: delete the top model, survivor P(best) under Luce
renormalization vs exact re-pricing, highlighting the survivor whose
correlated errors Luce overprices.

## Positioning
"Model selection on a shared eval set is a correlated race; winning
turns the leaderboard into exact P(best) vectors and prices router
failover by correlation-aware removal, where the field renormalizes
softmax."

## Peter's addendum (same date)
The router argument stated crisply: if agent A is down, softmax
renormalization assumes its probability flows by current score
levels; Gaussian maximality asks which surviving agent was likely to
beat A UNDER THE SAME LATENT TASK REALIZATION. Factor logit is
disqualified whenever the stated quantity is "probability this model
is best"; it remains permissible only when the designer explicitly
wants a Boltzmann router. Verdict concurs: best flagship demo (data
easy, factors interpretable, distinction immediately comprehensible).
