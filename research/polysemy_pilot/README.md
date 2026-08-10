# Polysemy pilot: stated vs revealed sense distributions (Luce vs Thurstone)

Pilot replication of Cekinmez, Wu, Marjieh & Griffiths, "Where did the
ambiguity go?" (arXiv 2608.00410, COLM 2026), extended with the
choice-set-restriction test from Cotton, "A Paradox in Machine Preference"
(2024). Run 2026-08-05 via the `claude` CLI: generation model Haiku 4.5,
judge Sonnet, 15 polysemous words from their Appendix C, 30 independent
samples per condition.

## Result 1 — their collapse replicates on Claude

Mean normalized sense entropy: **generated 0.265** (their text models: 0.25),
**stated 0.906** (their stated: 0.73). Haiku knows the ambiguity but does not
express it: 8 of 15 words collapse to a single sense in all 30 samples.

## Result 2 — stated reports are not just flattened locations

One-parameter stated→generated fits are nearly tied (Thurstone sigma=0.49,
RMSE 0.2934; Luce power-tilt gamma=2.0, RMSE 0.2943; Thurstone wins 10/15
words). Both fit poorly because the generated modal sense sometimes isn't the
stated argmax (port: generates *harbor* 100%, states *connector* most likely;
crane: generates *machine*, states *stretch*). No rank-preserving map can
capture that, so stated distributions are a biased window on the latent
locations — calibrate to revealed behavior instead.

## Result 3 — restriction test: Thurstone edges Luce, and the signature is a re-collapse

For the 5 words with non-degenerate unqualified distributions, we excluded the
modal sense in the prompt and compared zero-parameter predictions of the
restricted distribution (add-half smoothed unqualified counts):

| word  | Luce RMSE | Thurstone RMSE | actual restricted distribution |
|-------|-----------|----------------|--------------------------------|
| bolt  | 0.449 | **0.423** | fastener .76, run .17, lock .07 |
| seal  | **0.340** | 0.347 | close .89, stamp .11 |
| pitch | 0.626 | **0.580** | throw .06, tone .29, field .65 |
| bat   | 0.750 | **0.709** | animal 1.00, sports .00 |
| iron  | **0.076** | 0.130 | metal .93, press .07, golf .00 |

Pooled RMSE: **Luce 0.493, Thurstone 0.468** (Thurstone wins 3/5 words).

The qualitative signature matters more than the margin: under restriction the
model does not renormalize proportionally (Luce) — it **re-collapses onto a
new modal sense** (seal→close .89, iron→metal .93, bat→animal 1.00). That is
the low-noise contest behavior: remove the winner and the runner-up now wins
nearly always. Consistent with Cotton (2024) Fig. 1, where Thurstone's edge
concentrates in low-entropy, heavily-qualified cells.

## Caveats

- Small n (15 words, 30 samples, one generation model), CLI sampling
  temperature not controlled.
- "bat" restriction is confounded: excluding the *hit* sense plausibly
  suppresses *sports* (baseball) too; sense overlap in the inventory.
- Judge is a single LLM (Sonnet); the paper used two judges plus a human
  hand-label check.
- Rank flips under restriction (bat→animal) are captured by neither family;
  worth investigating as prompt-induced sense re-weighting.

## Files

- `stimuli.json` — words and sense inventories (subset of their Appendix C)
- `gen/` — 30 unqualified sentence samples per word
- `restricted/` — 30 modal-sense-excluded samples for 5 words
- `judged.json`, `restricted_judged.json` — per-sentence sense labels
- `stated.json` — model's self-predicted sense percentages
- `analysis.json`, `restriction_analysis.json` — computed distributions & fits
- `pilot_generate.py`, `pilot_judge.py`, `pilot_analyze.py`,
  `pilot_restrict.py` — the pipeline, in run order

## Exact-logprob GPT batteries (Sections 5–6 of the paper)

These use the OpenAI API (`max_tokens=1`, `top_logprobs=20`, key read from
`winning/.env`), so preferences are measured rather than sampled. Item
inventories live in `inventory.py`: `BASE_INVENTORY` (17 categories, frozen —
`exact_analyze.py` and `random_restrict.py` import it so their committed
results stay reproducible) and `INVENTORY` (50 categories, used by the
deletion battery and `fetch_unq_new.py`).

| battery | script | results | headline |
|---|---|---|---|
| original 2024 two-slot stimuli, 99 categories × every adjective × 3 models | `vol_battery.py` (`N_ADJ=0`) | `vol_battery_results.json` | 2,192 cells, ΔKL **+0.55** [+0.50, +0.59] |
| permutation-controlled deletion, top-8 items × 2 phrasings × 50 categories × 3 models | `perm_restrict.py` | `perm_results.json` | 692 cells (31 scorable categories), ΔKL **+0.025** [+0.016, +0.035] |
| single-deletion random elicitation | `random_restrict.py` | `random_results.json` | Thurstone 39/56 non-degenerate cells |
| Block–Marschak / RUM test | `bm_battery.py` | `bm_results.json` | 18/18 structures violate RUM |
| menus, duplicates, decoys | `red_bus.py`, `context_effects.py`, `transport.py` | matching `*_results.json` | regularity violations 8/18 |

ΔKL is `KL(actual‖Luce) − KL(actual‖Thurstone)`; positive favors Thurstone,
with bootstrap CIs over cells. Reproduce in order: `fetch_unq_new.py` (fills
`random_raw.json` for any category new to `INVENTORY`), then the battery
script — both cache raw API responses (`perm_raw.json`, `random_raw.json`)
and re-fetch only what is missing.
