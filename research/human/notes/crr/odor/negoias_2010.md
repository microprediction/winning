# Negoias et al. 2010 — 3-AFC vs 6-AFC cued odor identification

## Citation
Negoias S, Troeger C, Rombaux P, Halewyck S, Hummel T. "Number of Descriptors in Cued Odor
Identification Tests." *Archives of Otolaryngology — Head & Neck Surgery* 2010;136(3):296–300.
doi:10.1001/archoto.2009.231. PMID 20231651. (Verified via Europe PMC core record.)

## Domain and stimuli
Olfaction, cued odor identification. 32 odorant pens — an extension of the 16-item Sniffin' Sticks
identification subtest (the 32-item SSET). 238 participants: 128 with olfactory complaints, 110
controls. University clinic, Brussels, 1 Mar 2008 – 30 Apr 2009. Randomized **crossover**: every
participant took both the 3-AFC and the 6-AFC version **in the same session**, 10-minute break,
order randomized across participants.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**Nested, and this is the single best-designed instance found in the whole olfactory literature.**
Master set S = 6 descriptors per odor; restricted set T = 3 descriptors per odor; same odor, same
subject, minutes apart.

Construction rule, inferred and consistent: T(3) = the standard Sniffin' Sticks 4-AFC list minus one
distractor; S(6) = that same standard list plus two new distractors. So on most items you get a
three-level chain **T(3) ⊂ M(4) ⊂ S(6)**, where M is the standard 4-AFC list for which open
chosen-descriptor data exists (see `tolomeo_2026.md`, `nhanes_2014.md`).

Verified examples (Table 2):
- item 1: {orange, strawberry, pineapple} ⊂ {blackberry, pineapple, grape, strawberry, orange, apple}
- item 4: {spruce, onion, peppermint} ⊂ {spruce, onion, garlic, peppermint, clove, chive}
- item 8: {mustard, rubber, turpentine} ⊂ {rubber, turpentine, menthol, carrot, nut, mustard}

Two independent extraction passes of Table 2 both concluded **~22 of 32 items are strictly nested**:
items 1–5, 7–11, 14, 15, 17–21, 23, 25, 27, 30, 32. Apparent failures: 6 (apple), 13 (plum), 16
(fish), 22 (raspberry), 26 (grass), 28 (strawberry), 31 (cookie). Items 12 and 24 hinge on whether
"Black pepper" and "Pepper"/"Red pepper" are the same token; item 29 lists mismatched targets across
the two columns, probably a typo. On several items M ⊄ S (a standard distractor was dropped from the
6-list), e.g. items 7, 9, 12 — so the clean claim is T ⊂ S, with T ⊂ M ⊂ S holding on a subset.

**CAVEAT: both extraction passes read the JAMA HTML page, which renders Table 2 unreliably**
(duplicated words such as "Pear, Pear", self-contradictory readings on repeat queries). The nesting
list above must be re-checked against the print PDF before it is used in an argument.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Only pooled summary scores. No per-item and no per-descriptor numbers anywhere.**
- Table 1: probable causes of olfactory dysfunction in the patient group.
- Table 2: the descriptor sets — target plus distractors for all 32 items, both conditions. This is
  the design, not data, and it is the only genuinely reusable content in the paper.
- Table 3: descriptive statistics / percentile distributions of total score. Controls span 22–31
  (3-AFC) and 16–31 (6-AFC); patients 5–32 and 1–30.
- Table 4: mean (SD) totals. Controls 28.06 (2.26) 3-AFC vs 24.61 (3.23) 6-AFC; patients 19.61
  (6.77) vs 14.89 (7.75). Correlation between formats r = 0.92.

No supplementary material, no eTables, no deposited files. Confirmed three ways: Europe PMC core
record gives `hasSuppl: N` and no data links; Unpaywall gives `is_oa: false`, `oa_status: "closed"`,
zero OA locations; the article page has no Online-Only Material section (only an "Additional
Contributions" acknowledgment naming Ilona Croy).

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
https://jamanetwork.com/journals/jamaotolaryngology/fullarticle/496101 — fetched, resolves,
**paywalled**. Abstract and table structure visible; the PDF is not.
https://api.unpaywall.org/v2/10.1001/archoto.2009.231 — fetched: closed, zero OA locations.
No OA copy exists anywhere. No thesis copy either (DNB SRU, Qucosa, OpenAIRE all returned nothing
for Troeger / Geruchsidentifikation / Deskriptoren).
Direct `curl` of the JAMA page returns HTTP 403; only the WebFetch path works.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Unusable as a data source. Usable as a design source.**
The choice shares over S and over T were never published in any form, so there is nothing to
digitize — the numbers do not exist outside the authors' files. Table 2, however, fully specifies 32
nested descriptor triples and sextuples, which makes the experiment exactly replicable.

Two routes forward:
1. Email Thomas Hummel (Smell & Taste Clinic, TU Dresden; corresponding author) and Simona Negoias
   (Bern) for the 2010 raw response sheets, n=238 with 22 strictly nested items.
2. Get the print PDF via library access purely to certify the Table 2 nesting list.

Of the 23 papers citing it (enumerated via Semantic Scholar), **none replicated the n-AFC
manipulation**. The nearest relatives use the same 32 odorants with a fixed 4 descriptors
(`sorokowska_2015.md`), which would give 3/4/6 menu sizes over one odorant set — but across
different samples, with no item-level data on any side.

## What the authors concluded, quoted verbatim where possible
> "cued odor identification tests with various numbers of verbal descriptors produce similar
> results, however an increasing number of alternative descriptive items seem to allow for better
> discrimination between individuals with and without olfactory loss"

Also reported: age-related performance differences favored the 3-AFC test — older participants
scored relatively lower on 6-AFC than on 3-AFC — and the 6-AFC format better discriminated severity
of olfactory loss. Note that regularity on totals holds in the trivial direction (adding
alternatives lowers the hit rate); the per-descriptor shares that would let you test anything
sharper are exactly what is missing.
