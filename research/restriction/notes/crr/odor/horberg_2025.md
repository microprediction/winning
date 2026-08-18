# Hörberg et al. 2025 — free identification then 4-AFC cue, same odors, same subjects (SNAC-K)

## Citation
Hörberg T, Olofsson JK, Raj R, Laukka EJ, Larsson M. "Free odor identification engages domain-general
cognitive abilities in old adults." *Chemical Senses* 2025;50:bjaf049. doi:10.1093/chemse/bjaf049.
PMC12603616. (Verified via Europe PMC core record.)

## Domain and stimuli
Olfaction. Sniffin' TOM (a modified Sniffin' Sticks identification test), **16 odors**. Participants
are older adults from SNAC-K (Swedish National study on Aging and Care in Kungsholmen), a large
population-based cohort.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**THIS IS THE FREE-MASTER / RESTRICTED-SUBSET PAIRING — and it is within-subject, same odors.**

Procedure: **free (uncued) identification is attempted first; if the free response is incorrect, the
participant is then shown 4 written alternatives** (the target plus three distractors) for that same
odor and chooses again.

So per odor per subject you get an ordered pair:
- master phase: response drawn from the participant's entire lexicon, effectively unbounded |S|
- restricted phase: response drawn from a fixed 4-element set T over the *same* odor

This is a menu *contraction* from unbounded to 4, not a nesting of two finite lists. Two structural
caveats that matter for any regularity argument:
1. The restricted phase is **conditional on failing the free phase** — a selection effect. You only
   observe T-choices for the subset of subject×odor cells where the free attempt was wrong.
2. Free responses are coded into categories (correct / misnaming / omission) rather than preserved as
   raw strings in the analysis, so the master-phase choice object is coarser than the restricted one.

Even so, this is the closest thing in the literature to the structure sought, and it exists at
n ≈ 2,500 scale (see `raj_2023.md`, same cohort and instrument).

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Analyzed at the level of individual item responses, but nothing is deposited.**
The authors model three free-response categories (correct responses, misnamings, omissions) "on the
level of individual item responses" using Bayesian multilevel models with by-participant and by-odor
random intercepts. So per-subject per-odor records exist and were used — they are simply not released.
The paper prints model coefficients, not choice shares over descriptor sets.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12603616/ — fetched, **open access**.
Data: **restricted.** Data availability statement, verbatim:
> "The data that support the findings from this study are available from the Swedish National Study on
> Aging and Care in Kungsholmen database committee upon reasonable request."
No URL, no DOI, no repository. Application route: https://www.snac-k.se/for-researchers/application-form/

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Unusable now; needs a data request. Second-highest-priority email after Monell/SCENTinel.**
Nothing to digitize — the paper prints model output, not the tables you would need. But the underlying
records are exactly the free-then-cued paired structure over 16 odors in a cohort of thousands, already
collected and already cleaned. Ask the SNAC-K database committee, and ask jointly for the `raj_2023.md`
distractor-level variables since it is the same instrument and cohort.

Worth requesting specifically: (a) the raw free-identification *strings* rather than the
correct/misnaming/omission coding, and (b) the chosen descriptor on the cued phase, not just accuracy.
Without (b) the restricted phase is only binary and the pairing loses most of its value.

## What the authors concluded, quoted verbatim where possible
The title states the finding: **free odor identification engages domain-general cognitive abilities**
in old adults. The argument is that uncued naming loads on semantic and executive resources in a way
cued identification does not — i.e. the two phases are not merely easier and harder versions of one
task, they recruit partly different processes. That is a substantive warning for anyone treating the
free phase as a "master set" version of the same choice problem: the authors' own conclusion is that
changing the response format changes what is being measured, not just how many options are on offer.
