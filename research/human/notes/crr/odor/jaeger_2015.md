# Jaeger et al. 2015 — nested CATA term lists (10–17 vs 20–28), sensory science, no data deposited

## Citation
Jaeger SR, Beresford MK, Paisley AG, Antúnez L, Vidal L, Silva Cadena R, Giménez A, Ares G.
"Check-all-that-apply (CATA) questions for sensory product characterization by consumers:
Investigations into the number of terms used in CATA questions." *Food Quality and Preference*
2015;42:154–164. doi:10.1016/j.foodqual.2015.02.003. (Author list, volume and pages verified via
OpenAlex.)

## Domain and stimuli
Sensory/consumer science rather than clinical olfaction — aroma and flavour attributes of food and
beverage products, judged by consumers. **7 studies, 735 consumers** in total.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**NESTED BY CONSTRUCTION — this is the sensory-science analogue of the design sought, and the closest
methodological match found outside `negoias_2010.md` and `parma_2021.md`.**

Short lists of **10–17 terms** versus long lists of **20–28 terms**, with the long lists built by
**adding terms to the short lists**. So T ⊂ S by construction, over the same products.

Two important differences from a forced-choice odor identification task:
1. **CATA is check-all-that-apply, not pick-one.** Respondents may select any number of terms, so the
   choice object is a subset rather than a single alternative. Regularity in the Luce/Thurstone sense
   does not transfer directly; the natural analogue is whether the marginal selection rate of a term
   present in both lists falls when the list is lengthened.
2. There is no "correct" answer, so there is no accuracy dimension — only attribute-application rates.

The design is nonetheless exactly a menu augmentation with the added elements known, at n = 735.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Unknown in detail; certainly nothing deposited.** Closed access blocked inspection of the tables.
Expect printed per-term citation proportions by list-length condition — which, if present, would be
digitizable pooled choice shares over nested attribute sets.

Nothing in any repository. Verified: OSF has **zero** nodes matching "check-all-that-apply"; DataCite
(covering Zenodo, Mendeley, Dryad, figshare, institutional repositories) returns 21 CATA sensory
records, **every one with a single fixed term list**; the `cata`, `tempR`, `SensoMineR` and
`FactoMineR` R packages were installed and every bundled dataset enumerated — no nested-list dataset.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
https://doi.org/10.1016/j.foodqual.2015.02.003 — **paywalled** (Elsevier; ScienceDirect 403s automated
fetches). OpenAlex reports `best_oa_location: null` — no OA copy anywhere, no preprint, no Wayback
capture of a full text.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Needs library access first, then very likely needs digitizing.**
Highest-value non-olfactory lead in the directory: the nested manipulation was actually run, at
n = 735, across 7 studies. Get the PDF and check whether per-term proportions are tabulated by
condition. If they are, transcribe them — that yields pooled choice shares over nested sets without any
data request.

Then email **Gastón Ares** and **John Castura** for the raw arrays. Castura ships raw CATA data inside
his own R packages, which makes him the likeliest to say yes.

Companion studies with the same structure, all closed and undeposited: Jaeger et al. 2018
(doi:10.1016/j.foodqual.2017.09.013, 9-term vs 15-term TCATA); Ares et al. 2015
(doi:10.1016/j.foodqual.2015.01.015, 12 vs 20 terms); Pineau et al. 2012
(doi:10.1016/j.foodqual.2012.04.004). All 11 open-access works citing Jaeger 2015 were checked and
**none** manipulates list length.

## What the authors concluded, quoted verbatim where possible
Not verified verbatim (paywalled). From the title and the literature that cites it, the established
conclusion is that the **number of terms in a CATA question influences the results** — longer lists
change term-application rates and can affect product discrimination and configuration — leading to
practical guidance on list length. That is, in this field's own terms, an acknowledged menu effect,
which makes the absent raw data especially annoying.
