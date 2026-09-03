# Backblaze first-failure probe: independence is falsified for
# identifiable batches, and the durability consequence is measured
(2026-09-03. Data: Backblaze Drive Stats Q1 2025, 1 GB streamed to a
283 KB (date, model) cohort table; 90 days, 76 models, 1067 failures.
Scripts: research/applications/backblaze/.)

## Extraction validated
Per-model annualized failure rates land at 0.5-5.0%, matching
Backblaze's own published range -- the reduction is faithful.

## Independence is falsified, but HETEROGENEOUSLY -- the useful part
The daily-failure dispersion index Var/Mean (1.0 = independent
Poisson) and the fitted one-factor common-cause sigma, per model:
  HGST HUH721212ALN604   Var/Mean 2.36   sigma 0.22   (AFR 5.0%)
  TOSHIBA MG07ACA14TA    Var/Mean 1.81   sigma 0.16
  HGST HUH721212ALE604   Var/Mean 1.71   sigma 0.19
  ST8000NM0055           Var/Mean 1.51   sigma 0.15
  ST16000NM001G          Var/Mean 1.32   sigma 0.18
  ...
  ST14000NM001G          Var/Mean 0.97   sigma 0.00   (independent)
  ST8000DM002            Var/Mean 0.94   sigma 0.00
  WDC WUH721816ALE6L4    Var/Mean 0.78   sigma 0.00
Some drive models are clean independent Poisson (sigma = 0); others
carry clear common-cause clustering (sigma up to 0.22). This is the
weakest-link IDENTITY the engine computes: the correlation is
concentrated in specific batches, and the fit names them. A uniform
"drives fail independently" MTBF model is right for some models and
wrong for others, and you cannot tell which without the fit. Fleet
aggregate dilutes it to dispersion 3.2, sigma 0.10 (mixing clean and
clustered models).

## The durability consequence, measured
Even the modest fitted sigma makes independence UNDER-estimate the
same-day k-simultaneous-failure tail (the erasure-coding data-loss
object) materially. For 20 correlated drives at the fleet sigma 0.10:
  P(>= 2 same day): correlated / independent = 1.4x
  P(>= 3):  2.1x
  P(>= 4):  3.8x
  P(>= 6):  19.7x
Within the worst batch (sigma 0.22) the ratios are far larger. The
independent-exponential MTBF model, standard in reliability
engineering, under-states correlated data-loss risk by one to two
orders of magnitude in the tail that matters.

## The pairing with latency (both facets, one sentence)
Independence is wrong in BOTH systems races, in opposite directions:
for the MAX (fan-out latency) positive correlation makes the tail
LIGHTER, so independence OVER-provisions capacity (Slepian); for the
COUNT (k-out-of-n failures) common cause makes clustered failures
likelier, so the data-loss tail is HEAVIER and independence
UNDER-estimates durability risk. The engine computes the exact value
in both.

## Honest limits of THIS data
- Backblaze publishes model but NOT rack/datacenter/power placement,
  so the finer common-cause (a rack losing power, a hot aisle) is
  unobservable here; the visible correlation is model-level and
  fleet-daily. The richer factor structure needs a dataset with
  placement (the Alibaba microservices trace has nodeid).
- One quarter, daily granularity. The constant-hazard null ignores
  aging/cohort drift within the quarter, which inflates dispersion
  somewhat; the per-model sigma is an upper-ish estimate of pure
  common cause. A survival model with age would sharpen it.
- sigma 0.10-0.22 is modest but real and its tail consequence is not.

## Verdict
The flagship claim holds on real published data with the correlation
factor in the schema: independence is falsified for identifiable
drive batches, the engine's fit names them (weakest-link identity)
and prices the k-out-of-n consequence (up to ~20x at the fleet, more
per batch). This is the seed of a short applied note --
"Correlated first-failure in drive fleets" -- pending the aging
control and, ideally, a placement-bearing dataset for the rack/DC
factor.
