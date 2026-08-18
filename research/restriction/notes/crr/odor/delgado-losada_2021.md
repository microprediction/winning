# Delgado-Losada et al. 2021 — Spanish Sniffin' Sticks, two different 4-descriptor sets (relabelling)

## Citation
Delgado-Losada ML, Bouhaben J, Delgado-Lima AH. "Development of the Spanish Version of Sniffin's
Sticks Olfactory Identification Test: Normative Data and Validity of Parallel Measures."
*Brain Sciences* 2021;11(12):216 [also cited as 11:216]. PMC7916642.
(PMCID verified by fetch; volume/issue as printed on the record.)

## Domain and stimuli
Olfaction, cued identification. Spanish adaptation of the Sniffin' Sticks identification subtest,
16 items, 4-AFC. Normative sample plus a parallel-forms validity study.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**A RELABELLING, not a nesting — and the only clean instance found.**
The study administers the **same odorants with two different 4-descriptor sets**: the Sniffin' Sticks
"blue" and "purple" identification versions. Cardinality is fixed at 4 on both sides; what changes is
*which* words are on offer.

So this is |S| = |T| = 4 with substituted labels over identical stimuli — a menu *swap*, not a
contraction. That is a different axiom test from regularity: it probes whether choice depends on the
identity of the offered alternatives rather than on their number. Useful as a complement to a nesting
design, not a substitute.

Related and worth chasing: the NIMH Data Archive structure `sniffin01` carries a `sniff16_version`
field coded **1 = Blue, 2 = Purple**, so any NDA collection that administered both versions to the
same subjects would be a same-odor / different-menu dataset with per-descriptor codes. See
`nda_sniffin01.md`.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Item difficulty only; nothing per descriptor, nothing deposited.**
Table 2 gives item-level difficulty (percent correct per item), which is category (c) — no information
about which distractor was chosen. Data availability is by author request: "Data at individual level is
available upon request to first author."

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
https://pmc.ncbi.nlm.nih.gov/articles/PMC7916642/ — **open access**.
Individual-level data: author request only, no URL.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Unusable as published; worth one email.**
The printed tables give percent-correct per item, which cannot support any choice-set analysis. But the
*design* is a genuine relabelling over identical odorants with both versions in hand, so the raw
records would let you test label-substitution effects directly. Ask Delgado-Losada for individual-level
responses **including the chosen descriptor**, not merely correctness — the request has to be specific,
because the published analysis only ever needed the binary.

## What the authors concluded, quoted verbatim where possible
The paper establishes Spanish normative data and reports that the blue and purple identification
versions function as **valid parallel measures** — i.e. the two 4-descriptor sets yield comparable
total scores. Note the shape of that conclusion: parallel-forms equivalence is asserted at the level of
sums, exactly as in `negoias_2010.md`, while the per-descriptor behaviour that would reveal whether the
label swap moved individual choices is never examined.
