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

## Aging control (analyze2.py): the common cause is real and
## environmental, not batch drift
The referee objection -- overdispersion could be aging/cohort drift,
not same-day common cause -- is answered three ways:
1. DETRENDED DISPERSION SURVIVES. Cohort-adjusting and removing a
   7-day smooth per-drive hazard trend takes the fleet daily
   dispersion from 3.16 to 2.88 -- almost all of the overdispersion
   is same-day clustering a smooth trend cannot explain, not drift.
2. SPIKE DAYS SPAN ALL MANUFACTURERS. Six days carry >3-sigma excess
   failures (~2x expected), and each spans 7 to 13 DISTINCT drive
   models across Seagate, HGST, Toshiba and WDC. A bad batch hits one
   model; these hit everything on the same calendar day -- an
   environmental/fleet-wide event, not a manufacturing defect.
3. CROSS-MANUFACTURER CORRELATION (the killer test). After removing
   each manufacturer's own trend, the manufacturers' daily failure
   residuals are all POSITIVELY correlated, mean +0.19, with
   Seagate x HGST at +0.40. Seagate and HGST fail by unrelated
   mechanisms and different firmware; same-day co-movement at +0.40
   can only be a shared environmental factor -- the common cause
   aging cannot fake, and exactly the factor a durability model
   needs. Independence is falsified at the fleet level, decisively.
This upgrades the verdict: the durability-relevant common cause is
present, environmental (not merely per-batch), and observable in the
public data even without rack/DC labels -- because it is fleet-wide.

## Replication (Q4 2024) and a rejected hypothesis (temperature)
Pulled a second quarter (Q4 2024, ~305k drives, 1032 failures) with
the SMART temperature column, to test replication and to try to
IDENTIFY the shared factor as thermal.
- REPLICATES IN SIGN, WEAKER IN MAGNITUDE. Detrended dispersion 1.73
  (Q1 2025 was 2.88); mean cross-manufacturer detrended correlation
  +0.06 (Q1 was +0.19), Seagate x HGST +0.21 (Q1 +0.40). So the
  common-cause effect is present in both quarters and always
  positive, but its strength varies quarter to quarter -- Q1 2025
  was a high-correlation quarter, Q4 2024 a milder one. The direction
  is robust; the magnitude is not a constant.
- TEMPERATURE IS NOT THE FACTOR (hypothesis rejected). Datacenter
  drive temperature is nearly constant (31.7-34.0 C, climate-
  controlled), correlates with the detrended daily failure residual
  at +0.019 (essentially zero), and partialling it out of the
  manufacturers' residuals changes their cross-correlation by nothing
  (+0.063 -> +0.063). Spike days are only marginally warm (+0.45
  sigma). The shared factor is environmental in the broad sense
  (it hits all manufacturers on specific days) but it is NOT drive
  temperature. Identifying it precisely -- power events, maintenance
  windows, handling, humidity, collection artifacts -- needs
  placement/operations data the public release does not carry.
Net: the CORE claim survives and replicates (independence is
falsified, positive same-day cross-manufacturer correlation in both
quarters, beyond aging), but two honest corrections -- the magnitude
varies across quarters, and the tempting thermal explanation is
false. The durability consequence should be quoted as a RANGE across
quarters, not the Q1 point.

## Out-of-sample payoff (validate.py): the correction is transferable
## and predicts the tail 37% better
The final test: is the correlation merely present, or does it PREDICT
better than independence out of sample? Fit the independent Poisson
and the overdispersed (negative-binomial, the one-factor marginal)
rate models on Q4 2024, carry the learned dispersion r=5.2 to Q1
2025, and score held-out per-day log-loss on the different quarter:
- overall: correlated beats independent by 13.0% held-out log-loss;
- TAIL (top-decile failure days -- the clustered heavy days erasure
  coding must survive): 37.0% better, 7.24 -> 4.56 log-loss.
The dispersion learned on one quarter transfers to the next, and the
independent Poisson is worst exactly where it matters (the heavy
days). This is the result that makes the thread a contribution: not
"correlation exists" but "a one-parameter common-cause correction,
fit on one quarter, prices the next quarter's failure-count tail 37%
better." r=5.2 (finite) is the measured distance from independence
(r -> infinity).

## SKEPTIC'S CONFOUND (confound.py): partly administrative, and it
## matters for the durability claim
The dangerous alternative: Backblaze's "failure date" may be when a
drive is MARKED/pulled (a maintenance cadence), not when it
physically died. If failures are recorded in weekday batches, all
manufacturers spike on the same ADMIN days -- cross-manufacturer
correlation with no physical common cause. Tested:
- WEEKDAY SIGNATURE EXISTS but is mild: weekday/weekend failure ratio
  1.16 (Q1) and 1.15 (Q4); Q1 is Friday-heavy (17.4 vs Mon 7.7), and
  3 of 6 Q1 spike days are Fridays. So there IS an administrative
  cadence in the recording.
- THE CORRELATION IS PARTLY, NOT WHOLLY, THIS. Removing day-of-week
  means takes the Q1 cross-manufacturer correlation from +0.19 to
  +0.12 (a third is weekly-administrative, two-thirds survives); Q4
  is barely affected (+0.063 -> +0.059).
- BUT day-of-week removal only kills the WEEKLY component. Irregular
  bulk-removal batching (episodic, not weekly) would survive it and
  is indistinguishable from physical co-failure using failure DATES
  alone.

CONSEQUENCE FOR THE DURABILITY CLAIM (the important part): the k-out-
of-n / 20x / 37%-tail argument requires the co-timing to be PHYSICAL
(drives actually failing together, threatening erasure-coded data).
If the co-timing is administrative (drives failing on different real
days but RECORDED together), durability is NOT affected -- the data
was lost or not, spread over real time. Failure-date data cannot
distinguish physical simultaneity from co-recording, and the
day-of-week test proves at least some of the clustering is
administrative. So:
- SURVIVES: independence is statistically falsified -- the daily
  count is overdispersed and the negative-binomial correction beats
  Poisson out-of-sample (13%, 37% tail). That is a fact about the
  RECORDED distribution regardless of cause.
- DOES NOT SURVIVE cleanly: the interpretation as a physical
  environmental common cause, and therefore the durability-risk
  number, which is now CONDITIONAL on the co-timing being physical --
  something this dataset cannot establish. A dataset with true
  failure timestamps and placement (or SMART-based time-of-death) is
  needed to separate the two.
This is the honest downgrade: a real statistical finding
(overdispersion, predictive) with an interpretation (physical common
cause / durability) that the data cannot secure and a reporting
confound partly present. The site note must say so.
