# Applications scouting: machine learning systems

Front: ML systems applications of the exact multinomial-probit race engine
(low-rank-plus-diagonal covariance; all N argmax probabilities exact in O(QNL);
win-share -> utility inversion; exact O(QNL) Jacobian-vector products; removal
counterfactuals in one shared pass). Horse racing / betting excluded by charter.

Scouted 2026-08-25. Ranked list at the bottom; sources and verbatim quotes inline.

---

## 1. Heteroskedastic classifier heads (Collier et al. line) — RANK 1

### (i) Race mapping
Exact, not analogical. The HET model (Collier et al., CVPR 2021, arXiv:2105.10305)
*is* our model: latent utility u(x) = mu(x) + eps, eps ~ N(0, Sigma(x)), label =
argmax_j u_j(x), and Sigma(x) is explicitly **low-rank-plus-diagonal**:

> "We make a low-rank approximation to Sigma(x) = V(x)V(x)^T where V(x) is a
> K x R matrix, R << K. To ensure the positive semi-definiteness of the
> covariance matrix, we compute a K dimensional vector d^2(x) which we add to
> the diagonal of V(x)V(x)^T."

They use R = 50 with K up to 21,843 classes (ImageNet-21k) and JFT — squarely
inside our N=10^4-under-a-minute envelope, and their L=50 matches our L.

### (ii) Public data / benchmarks
ImageNet ILSVRC-2012, ImageNet-21k, JFT (private), CIFAR; reference code in
Google's `uncertainty_baselines` repo. Follow-up HET-XL ("Massively Scaling
Heteroscedastic Classifiers", ICLR 2023, openreview sIoED-yPK9l) scales the
same head further, so the line is alive inside Google DeepMind.

### (iii) Incumbent + documented limitation (verbatim)
Incumbent: S-sample Monte Carlo over the noise plus a temperature softmax in
place of the argmax. From the CVPR 2021 paper:

> "This generative process leads to two main challenges while computing its
> resulting likelihood: (a) the intractable marginalization over the latent
> variables which we estimate via Monte Carlo integration and (b) an arg max
> in the generative process which we approximate with a temperature
> parameterized softmax."

> "First there is now *no closed form solution* for the predictive
> probabilities."

> "In the zero temperature limit this approximation is exact, but for non-zero
> temperatures tau controls a bias-variance trade-off. At lower temperatures
> the approximation is closer to the assumed generative process but the
> gradient variance is higher and vice versa. In practice tau is a
> hyperparameter that must be selected on a validation set."

So the incumbent is *biased* (tau > 0), *noisy* (finite S), and carries an
extra hyperparameter whose sole job is to trade those two off.

### (iv) Unique advantage
We delete the approximation: p_c = P(argmax u = c) computed exactly in O(QNL),
and the exact JVP gives the exact gradient of log p_y w.r.t. (mu, V, d) —
no tau, no MC variance, no bias-variance knob. Their own table shows the full
covariance beat the diagonal by +0.6% top-1 on ImageNet with the *approximate*
likelihood; exact likelihood is the natural next rung. Also fixes evaluation:
their predictive probabilities at test time are MC estimates; ours are exact,
which matters for calibration metrics (ECE/NLL) that the same literature reports.

### (v) Minimal demo
Reimplement the HET head on CIFAR-100 / ImageNet-1k in JAX or PyTorch with our
exact likelihood + exact gradient as a drop-in replacement for softmax_tau + MC.
Report: (a) gradient variance vs their estimator at matched wall-clock,
(b) NLL / ECE / top-1 vs tau-tuned baseline, (c) no tau sweep needed. A second
notebook: at a *frozen* trained head, compare exact p_c against their S-sample
estimate to show the residual bias at their published tau* = 0.9.

### (vi) Who would care
Google DeepMind uncertainty group (Collier, Mustafa, Jenatton, Kokiopoulou);
`uncertainty_baselines` maintainers; NeurIPS/ICML/AISTATS; the Bayesian deep
learning / calibration workshops. The paper explicitly connects itself to
"the existing discrete choice modelling econometrics literature" — our home turf.

---

## 2. LLM leaderboards / arenas — RANK 2

### (i) Race mapping
Each battle is a 2-horse race today, but the *leaderboard* question is N-way:
P(model c is best) across N ~ 200-400 models whose abilities are correlated
through family/provider/style factors — a factor-Thurstone race. Two exact
race operations map to documented arena pathologies:
- **Removal counterfactuals** = model deprecation. BT/Elo with IIA predicts
  removing a model changes nothing about the others' relative shares;
  a correlated model says shares redistribute toward siblings.
- **Best-of-N private-variant gaming** = the max of N correlated variants.
  Pricing E[max] and the winner distribution of a correlated family is exactly
  our shared-pass computation; the Leaderboard Illusion authors had to simulate it.

### (ii) Public data / benchmarks
Public battle-level data on Hugging Face: `lmarena-ai/arena-human-preference-55k`
and `-140k`, `lmsys/chatbot_arena_conversations` (33k), `lmarena-ai/VisionArena-Battle`,
`search-arena-24k`. Model-family metadata is public. Enough to fit a rank-L
factor model over hundreds of models.

### (iii) Incumbent + documented limitation (verbatim)
Incumbent: Bradley-Terry logistic regression (LMSYS moved from online Elo to BT
in Dec 2023). Style confounding, from LMSYS's own blog
(lmsys.org/blog/2024-08-28-style-control):

> "Style indeed has a strong effect on models' performance in the leaderboard."
> "We explicitly model style as an independent variable in our Bradley-Terry
> regression." ... "Although controlling for style is a big step forward, our
> analysis is still observational." ... "There are possible unobserved
> confounders such as positive correlation between length and substantive
> quality that are not accounted for by our study."

Assumption violations, from The Leaderboard Illusion (arXiv:2504.20879):

> "We show that deprecation can violate key assumptions of the Bradley-Terry
> model (Bradley & Terry, 1952), which underpins Arena scoring, thereby
> reducing the reliability of the leaderboard rankings."

> "out of 243 public models, 205 have been silently deprecated."

> "We show with real-world experiments and simulations that the ability to
> select the best-scoring variant from N models enables systematic gaming of
> the Arena rating." ... "In a single month, we observe as many as 27 models
> from Meta being tested privately on Chatbot Arena in the lead up to llama 4
> release."

Prior art to position against: "A Statistical Framework for Ranking LLM-Based
Chatbots" (arXiv:2412.18407, Berkeley) already adds Thurstonian covariance to
*pairwise* arena modeling and finds "the inclusion or exclusion of the
Thurstonian covariance factor emerges as the primary driver of ranking
clusters." It stays pairwise and does not do N-way winner probabilities,
share inversion, or removal counterfactuals — that is our open lane.

### (iv) Unique advantage
Factor-Thurstone ratings where a family factor (loading column) is shared by
sibling models: identifies "Gemini-ness" / "Llama-ness" as latent dimensions,
gives P(best overall) with correlation-aware uncertainty, and prices the two
documented abuses exactly: (a) what deprecating 205 models did to the survivors'
implied abilities, (b) the expected rating inflation from launching k correlated
private variants and keeping the max — as a closed computation, not a simulation.

### (v) Minimal demo
Fit rank-3 factor probit to the 140k public battles (pairwise likelihood — the
correlated-probit preference likelihood with exact gradient already exists in
the codebase), then use the race engine for the N-way readout: P(each model is
best), family factors, and a deprecation counterfactual table "if these 20
models had never been in the arena." Compare IIA (BT) vs correlated predictions
on held-out battles involving same-family pairs.

### (vi) Who would care
LMArena / LMSYS maintainers; Cohere Labs (Leaderboard Illusion authors, actively
recommending fixes); Mahoney group at Berkeley (2412.18407 / leaderbot);
evaluation-methodology tracks at NeurIPS Datasets & Benchmarks, COLM.

---

## 3. RLHF / preference and reward modeling — RANK 3

### (i) Race mapping
Pairwise preference = 2-horse race (probit instead of logit; the codebase
already has the correlated-probit preference likelihood with exact gradient).
The N-way engine earns its keep on **K-wise ranking data**: InstructGPT-style
labeling ranks K = 4-9 responses per prompt, and datasets like Nectar carry
7-wise rankings; current practice shreds these into (K choose 2) BT pairs,
which is exactly the IIA reduction. Winner-of-K (best-of-N sampling, rejection
sampling on reward) is a correlated race because responses share a prompt and
a policy.

### (ii) Public data / benchmarks
UltraFeedback, HelpSteer2, Nectar (7-wise ranked responses), the Arena
preference sets above; RewardBench for evaluating reward models.

### (iii) Incumbent + documented limitation (verbatim)
Incumbent: Bradley-Terry log-loss on pairs, near-universally. From "Beyond
Bradley-Terry Models: A General Preference Model for Language Model Alignment"
(arXiv:2410.02197):

> "Traditional reward modeling methods, such as the Bradley-Terry (BT) reward
> model, fall short in expressiveness, particularly in addressing intransitive
> preferences."

They demonstrate "modeling cyclic preferences where any BT reward model behaves
like a random guess." Related lines: Nash-of-preferences methods (Munos et al.),
IPO (Azar et al.), and Gaussian/Thurstonian reward models that "generalize BT
from a scalar reward with fixed noise to a heteroscedastic reward distribution"
(variance-aware reward modeling line, e.g. arXiv:2605.11865). Non-BT likelihoods
are an active, publishable complaint — but note most intransitivity is
*population-level* mixing, which correlated probit only partially captures.

### (iv) Unique advantage
A listwise correlated-probit likelihood for K-wise ranked data with exact
gradients: train the reward/preference model on P(response c ranked first
among K | shared prompt factor) instead of independent BT pairs. Correlation
via shared prompt/policy factors is the statistically honest treatment of
"all K responses came from similar models," and the exact JVP makes it a
first-class training objective, not an approximation.

### (v) Minimal demo
Take Nectar's 7-wise rankings; fit (a) BT-on-pairs, (b) our winner-of-7
correlated probit (rank-1 or 2 prompt factor) using the existing likelihood
code; compare held-out first-place prediction and RewardBench accuracy.

### (vi) Who would care
Reward-model teams (Nemotron/HelpSteer at NVIDIA, Llama alignment, Cohere);
academic RLHF theory groups (Azar/Munos lineage, ETH "why does BT work" line);
ICLR/NeurIPS alignment tracks.

---

## 4. Differentiable top-k / perturbed optimizers — RANK 4

### (i) Race mapping
Perturbed argmax with Gaussian noise *is* the race: y*(theta) =
argmax_i (theta_i + eps_i). Berthet et al.'s perturbed maximizer expectation
E[argmax one-hot] is our vector of winner probabilities, and F(theta) = E[max]
is our G with p = grad G. We compute both *deterministically and exactly* for
correlated Gaussian noise; they Monte-Carlo it.

### (ii) Public data / benchmarks
Method literature more than benchmark literature: NeurIPS 2020 (Berthet et al.,
arXiv:2002.08676), SoftSort, Differentiable Patch Selection (arXiv:2104.03059),
perturbed beam search, PyEPO for predict-then-optimize.

### (iii) Incumbent + documented limitation (verbatim)
From Berthet et al. abstract:

> "Our approach relies on stochastically perturbed optimizers, and can be used
> readily together with existing solvers. Their derivatives can be evaluated
> efficiently, and smoothness tuned via the chosen noise amplitude."

Derivatives are "expressed as simple expectations, which are easy to
approximate with Monte-Carlo methods" — i.e., every gradient is a noisy
estimate, and downstream work documents the consequences: in Differentiable
Patch Selection, using hard top-K at inference creates a train-test gap,
sigma must be annealed, and gradients vanish as sigma -> 0.

### (iv) Unique advantage
For the top-1 / argmax case with (low-rank) Gaussian perturbation we replace
their MC estimator with the exact value and exact JVP: zero-variance gradients
at any noise scale, no sample-count hyperparameter, correlated noise for free
(their framework assumes exchangeable noise; correlation across alternatives
is new expressive power). Honest caveat: they cover general polytopes
(shortest paths, matchings, top-k for k > 1); we cover the simplex vertex case
exactly, so we are a strictly better kernel for argmax/categorical layers, not
a full replacement. Order/place extensions in the codebase could reach top-k.

### (v) Minimal demo
A "perturbed argmax, exactly" note: reproduce a Berthet et al. toy (e.g.,
learning-to-rank first place or discrete attention over N=10^3 items) and plot
gradient variance and convergence: MC perturbed optimizer (S = 10, 100, 1000)
vs exact race gradient at matched wall-clock.

### (vi) Who would care
Google Brain/DeepMind structured-prediction people (Berthet, Blondel, Cuturi
lineage); differentiable-programming workshops; JAX/PyTorch ecosystem (an
`exact_perturbed_argmax` op is a natural OSS artifact).

---

## 5. Mixture-of-experts routing — RANK 5

### (i) Race mapping
Noisy top-k gating (Shazeer 2017) is literally a race: route to
argmax_i (h_i + noise_i). Expected expert load = winner probabilities; experts
are correlated (siblings learn overlapping functions). Most striking mapping:
DeepSeek's aux-loss-free balancing adjusts a per-expert bias b_i until loads
balance — that is our **inversion** (given target shares, solve for utilities)
done by online heuristic instead of exactly.

### (ii) Public data / benchmarks
OLMoE, Mixtral, DeepSeek-V3 open weights; routing statistics reproducible from
open checkpoints.

### (iii) Incumbent + documented limitation (verbatim)
Auxiliary load-balancing losses; from "Auxiliary-Loss-Free Load Balancing
Strategy for Mixture-of-Experts" (arXiv:2408.15664):

> "a large auxiliary loss will introduce non-negligible interference gradients
> into training and thus impair the model performance"

Routing collapse (a few experts win everything) is the standard documented
failure; noise and losses are the patches.

### (iv) Unique advantage
Exact expected-load computation under noisy gating (no sampling), and exact
bias inversion: solve for the b_i that hit uniform load in one deterministic
pass instead of an online update rule. Correlation-aware routing could
penalize sending a token to an expert correlated with one already chosen.

### (v) Minimal demo
Offline: take OLMoE router logits over a corpus shard, compute exact loads
under Gaussian gate noise, invert for balancing biases, compare against
DeepSeek's incremental rule in convergence speed and final balance.

### (vi) Who would care
MoE systems groups (DeepSeek, Databricks/Mosaic, AI2 OLMoE). Realistically the
weakest front: per-token routing is latency-critical and N (experts) is small
(8-256), so O(QNL) exactness buys little; heuristics are entrenched. Keep as a
footnote application, not a campaign.

---

## Ranking and rationale

1. **Heteroskedastic heads (Collier line)** — the incumbent's two documented
   approximations (MC marginalization, temperature-softmax argmax with an
   explicit bias-variance hyperparameter) are exactly the two things we do in
   closed form, at their exact covariance structure (low-rank R=50 + diagonal,
   K up to 21k). Cleanest "delete the approximation" paper; public code and
   benchmarks; differentiability is the whole point there.
2. **Arena/leaderboard rating** — best story and public data; deprecation
   (205/243 models silently removed, documented BT-assumption violation) and
   best-of-N variant gaming are *removal counterfactual* and *max of
   correlated family*, our two signature operations. Must position against
   arXiv:2412.18407 (pairwise Thurstone-with-covariance already exists).
3. **RLHF K-wise preference modeling** — leverages the existing codebase
   likelihood; non-BT models are an active complaint line; the novel bit is
   listwise winner-of-K with exact gradients rather than pair-shredding.
4. **Differentiable perturbed argmax** — exact kernel beats MC on variance,
   but only covers the argmax vertex case of their general framework.
5. **MoE routing** — elegant inversion mapping (aux-loss-free bias = our
   inversion) but small N, latency-bound, heuristic-entrenched.

## Sources

- https://arxiv.org/abs/2105.10305 (Collier et al., CVPR 2021; quotes read from PDF pp. 2-4)
- https://openreview.net/pdf?id=sIoED-yPK9l (HET-XL, ICLR 2023)
- https://www.lmsys.org/blog/2024-08-28-style-control/
- https://arxiv.org/abs/2504.20879 (The Leaderboard Illusion; quotes read from PDF pp. 4-7)
- https://arxiv.org/abs/2412.18407 (Statistical Framework for Ranking LLM-Based Chatbots)
- https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k (and -55k, VisionArena, search-arena-24k)
- https://arxiv.org/abs/2410.02197 (General Preference Model / Beyond Bradley-Terry)
- https://arxiv.org/abs/2002.08676 (Berthet et al., perturbed optimizers)
- https://arxiv.org/abs/2104.03059 (Differentiable Patch Selection; train-test gap, vanishing gradients at sigma=0)
- https://arxiv.org/abs/2408.15664 (Aux-loss-free MoE balancing)
- https://arxiv.org/abs/2605.11865 (variance-aware / Gaussian reward modeling line)
