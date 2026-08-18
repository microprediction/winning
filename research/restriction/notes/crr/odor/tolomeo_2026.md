# Tolomeo et al. 2026 — Sniffin' Sticks SS-16 IRT, nominal response model, labels stripped

## Citation
Tolomeo E, Ceraudo L, Kolb R, Dalton PH, Liuzza MT, Parma V. "Improving olfactory assessment: an item
response theory analysis of the American English version of the Sniffin' sticks identification
subtest." *Frontiers in Psychology* 2026;17:1661164. doi:10.3389/fpsyg.2026.1661164. PMID 41635505.
PMC12863060.

## Domain and stimuli
Olfaction, cued identification. American English version of the Sniffin' Sticks Extended Test (SSET)
identification subtest: 16 odorized felt-tip pens, 4-AFC. Convenience sample recruited at the Monell
Chemical Senses Center, Philadelphia. 379 analyzed (226 female, 59.6%), mean age 44.61 (SD 18.17,
range 18–83); the deposited file has 397 rows.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**No restriction — fixed 4-alternative list per odor, one condition.**

Its relevance to a nested design is indirect but real: the standard SS-16 4-AFC lists analyzed here
are exactly the **M(4)** middle level in the T(3) ⊂ M(4) ⊂ S(6) chain implied by Negoias's
construction (see `negoias_2010.md`). So if the Negoias 3-AFC/6-AFC data is ever recovered, this
dataset supplies real-world choice shares at the intermediate menu size over overlapping items.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Per-subject, 4-way categorical — but the descriptor identities are lost.**

OSF files (all fetched, HTTP 200):
- `IRT_NRM.csv` (35,005 B) — **397 subjects × 16 items, coded 1 = correct, 2/3/4 = distractors.**
- `IRT_LABELED_DICHOTOMOUS.csv` (34,987 B) — same shape, 0/1 only.
- `Codebook Sniffin'Sticks Identification Subtest.docx` (9,419 B).
- Analysis `.R` (12,884 B).

I downloaded and parsed `IRT_NRM.csv`. Shares (opt1 = correct):

| item | correct | 2 | 3 | 4 |
|---|---|---|---|---|
| orange | .804 | .111 | .063 | .023 |
| leather | .559 | .222 | .118 | .101 |
| cinnamon | .637 | .207 | .126 | .030 |
| peppermint | .945 | .033 | .018 | .005 |
| banana | .720 | .154 | .068 | .058 |
| lemon | .549 | .343 | .055 | .053 |
| licorice | .806 | .098 | .053 | .043 |
| turpentine | .483 | .340 | .113 | .064 |
| garlic | .861 | .098 | .023 | .018 |
| coffee | .904 | .043 | .030 | .023 |
| apple | .476 | .297 | .164 | .063 |
| clove | .841 | .103 | .035 | .020 |
| pineapple | .554 | .196 | .161 | .088 |
| rose | .869 | .096 | .020 | .015 |
| anise | .700 | .151 | .098 | .050 |
| fish | .861 | .058 | .050 | .030 |

**KNOWN DEFECT — the 2/3/4 codes are frequency-ranked, not printed-position.** Counts are monotone
(2 > 3 > 4) in **all 16 of 16** items, which is only possible if codes were assigned by observed
frequency. The codebook confirms this obliquely ("Least chosen distractor = 1, intermediate = 2, most
frequent = 3" for the recode) and never names which word each code is. So the mapping from code to
descriptor **is not in the deposit**. It can be inferred by cross-referencing Table 3 of the paper,
which names all four options per item in ak0→ak3 order with the caption stating the least-chosen
distractor is the reference category — but that is an inference, not ground truth, and it is weakest
for near-ties (pineapple's 2nd and 3rd distractors differ by little).

Table 3 option sets (from the paper, D1/D2/D3 order as printed): orange {Strawberry, Pineapple,
Blackberry}; leather {Smoke, Glue, Grass}; cinnamon {Chocolate, Vanilla, Honey}; peppermint {Onion,
Chive, Fir}; banana {Coconut, Walnut, Cherry}; lemon {Peach, Apple, Grapefruit}; licorice {Cookies,
Cherry, Spearmint}; turpentine {Mustard, Rubber, Menthol}; garlic {Carrot, Sauerkraut, Onion}; coffee
{Wine, Cigarette, Smoke}; apple {Orange, Peach, Melon}; clove {Mustard, Pepper, Cinnamon}; pineapple
{Peach, Plum, Pear}; rose {Raspberry, Cherry, Chamomile}; anise {Honey, Rum, Fir}; fish {Ham, Cheese,
Bread}.

Second defect: **`id` is not unique** — ids 1, 92, 307 each appear twice (397 rows, ids 1–400, 6 gaps).
Join the two CSVs **positionally**; row-by-row they agree on all 6,343 non-missing cells, whereas an
id-keyed join spuriously produces 14 mismatches.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
Article: https://pmc.ncbi.nlm.nih.gov/articles/PMC12863060/ — **open access**.
Data: https://osf.io/3hjmw/?view_only=f93d1dd6812549b08d5503a207c4e49d

**The `?view_only=` token is mandatory.** OSF node `3hjmw` is `public: false` (confirmed via
`api.osf.io/v2/nodes/3hjmw/`), with **no public registration, no fork, and no DOI** (all three
endpoints checked, empty). Direct file downloads, each fetched HTTP 200:
- IRT_NRM.csv — https://osf.io/download/h456t/?view_only=f93d1dd6812549b08d5503a207c4e49d
- IRT_LABELED_DICHOTOMOUS.csv — https://osf.io/download/grqcu/?view_only=f93d1dd6812549b08d5503a207c4e49d
- Codebook — https://osf.io/download/4fs72/?view_only=f93d1dd6812549b08d5503a207c4e49d
- R script — https://osf.io/download/3pqsh/?view_only=f93d1dd6812549b08d5503a207c4e49d

Without the token, `https://osf.io/download/h456t/` returns HTTP 200 but the body is an OSF sign-in
HTML page, not the CSV — an easy way to silently get garbage.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Usable now at the 4-way categorical level; needs the authors' key for descriptor names.**
Fine as-is for anything that only needs "which of four", ranked. Not fine for anything that needs to
know *which word* — for that, email Parma / Tolomeo (Monell) for the answer-sheet mapping.

**ARCHIVE IMMEDIATELY.** This is a private project reachable only through an anonymized peer-review
token that the authors left in the published data-availability statement. It can be revoked at any
time and there is no public mirror, no DOI, no registration. Directly implicates the
no-losable-experiment-data rule: copy the four files into the repo today.

## What the authors concluded, quoted verbatim where possible
The paper's contribution is a distractor-level critique of the instrument. On the leather item the
authors note that "normosmics are likely to be attracted by grass instead of leather" — i.e. both the
correct answer and one distractor rise in selection probability with olfactory ability, so that
distractor is malfunctioning. Their general conclusion is that nominal-response-model distractor
analysis reveals item flaws that dichotomous discrimination parameters conceal, and that some SS-16
items should be revised. This is the one paper in the set whose *purpose* is per-descriptor choice
behaviour.
