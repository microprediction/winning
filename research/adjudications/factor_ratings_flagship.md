# Factor ratings beat scalar AND stratified ratings on real
# comparative data (flagship citable result, 2026-09-05)

Recorded from the bandits session's bounded report; supersedes that
thread's earlier bounded factor-ratings claims. SOURCE OF TRUTH for
any citation is the committed harness and CSVs, not this note (the
standing adjudication rule): bandits/experiments/exp28*.py
(exp28_factor_rating, exp28b_ceiling, exp28c_hostile_battery,
exp28d_three_way_ties, exp28e_real_competitor), per-battle results in
bandits/results/exp28*, README headline section (verified present,
2026-09-05).

## The claim
Factor ratings beat one-dimensional ratings on real human comparative
data, and beat the practitioner's alternative too. 65,178 Chatbot
Arena battles, 53 models, ability as a partially-pooled vector over
the arena's own category flags, all held out, paired bootstraps.

## The attack chain it survived
- Beats the global scalar: -0.0016 [-0.0029, -0.0004] nats,
  reproduced on four independent splits.
- Beats unrestricted per-matchup memorisation eight-fold (saturated
  bound 0.0002 -- the scalar family sits near the channel's
  information ceiling).
- Survives restoring the 40k dropped ties under a proper 3-way tie
  model: -0.0008 to -0.0016 across four splits. (Ties-as-half-wins
  reverses it, but that referee is minimised at 0.5 and punishes
  sharpness on an outcome the rating never emits -- not a valid
  rebuttal.)
- Beats STRATIFIED per-category ratings -- separate leaderboards per
  category, Arena's actual practice: -0.0008/-0.0008/-0.0007 on three
  splits, advantage monotone in cell sparsity, identical model on the
  73% uncategorised cell.

## The sharpest sentence
Stratified per-category ratings do NOT significantly beat the global
scalar at all (P 0.81-0.88 every split) -- existing leaderboard
practice spends its conditioning gain on sparse-cell estimation
noise. The factor structure converts category information into a
significant win because it POOLS each model's level across all
battles while only the offsets specialise. The contribution is
pooling, not conditioning.

## Bounds that must travel with any citation
- Absolute margin ~0.0016 nats, ~3.4% of extractable signal: binary
  comparisons are an information-poor channel.
- The identical structural idea is worth 16% on cardinal benchmark
  scores (exp27, Open LLM Leaderboard, 4,576 models x 6 benchmarks).
- exp28b's ceiling shows the factor rating captures most of the
  non-scalar signal that exists at this volume.
- What is PROVEN is identifiability of multi-dimensional ability from
  comparative feedback, not a large effect.

## Pre-registration discipline
All predictions registered before running, including two that went
against the author: "code carries the gain" (it's math: -0.0067 vs
+0.0001) and "family metadata adds on top of the model profile" (it
adds 0.0000).
