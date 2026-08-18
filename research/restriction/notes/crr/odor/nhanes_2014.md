# NHANES 2013–2014 Taste & Smell (CSX_H) — 8-item Pocket Smell Test, chosen descriptor public

## Citation
Centers for Disease Control and Prevention / National Center for Health Statistics. *National Health
and Nutrition Examination Survey 2013–2014: Taste & Smell (CSX_H)*. Public-use data file and
codebook, NHANES 2013–2014 cycle. Instrument: 8-item "scratch and sniff" Pocket Smell Test,
Sensonics Inc. Adults aged 40+.
(No journal first author — filed under the survey name for findability.)

## Domain and stimuli
Olfaction, cued identification. 8 microencapsulated "scratch and sniff" strips, 4-AFC, **forced
choice enforced even when the respondent smells nothing**. 3,708 examined records; ~3,520 valid per
item (181–189 coded 0 = not done per item).

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**No restriction — a single fixed 4-alternative list per odor, every trial.**

Worth stating precisely why this still matters, per the coordinator's note. A fixed 4-AFC list is not
a restriction of any larger master set *within this study*: the respondent never sees a different menu
for the same odor, so nothing here identifies a menu effect on its own. Two things nonetheless make
it valuable:

1. **The descriptor pool is reused across items.** Leather is the target on CSXLEAOD and a distractor
   on CSXSBOD and CSXSOAOD; Black pepper distracts on two items; Peanut on two; Chocolate and
   Strawberry are each target once and distractor once. So across items you observe overlapping
   4-subsets of a shared ~20-word descriptor universe. The odor changes with the menu, so this is not
   a clean regularity test, but it does give cross-item descriptor-level structure.
2. **It pairs with the UPSIT as a single-element swap.** Both are Sensonics microcapsule tests, and on
   three odors the menus differ by exactly one substituted distractor — see `mitchell_2025.md` for the
   table. A one-element swap is arguably a sharper IIA probe than nesting.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Both per-subject codes and pooled counts, which is the ideal combination.**

The `.xpt` carries eight variables coded 1–4 = which named descriptor was chosen (0 = not done):
CSXCHOOD, CSXSBOD, CSXSMKOD, CSXLEAOD, CSXSOAOD, CSXGRAOD, CSXONOD, CSXNGSOD.
The HTML codebook publishes the code→word mapping *and* the pooled counts.

I downloaded the `.xpt`, wrote an IBM-float XPT parser, and **my parsed counts reproduce the
published codebook exactly on all eight items** (reclen 321, 3,708 records). Target in bold:

| Variable | Target | Menu with counts (code order 1,2,3,4) |
|---|---|---|
| CSXCHOOD | Chocolate | Lemon 120, **Chocolate 2901**, Smoke 356, Black pepper 150 |
| CSXSBOD | Strawberry | **Strawberry 2819**, Garlic 29, Leather 553, Gasoline 124 |
| CSXSMKOD | Smoke | Garlic 124, Grass 223, **Smoke 3047**, Peach 129 |
| CSXLEAOD | Leather | Mint 206, Flower 429, **Leather 2693**, Apple 192 |
| CSXSOAOD | Soap | **Soap 3250**, Black pepper 99, Leather 101, Peanut 70 |
| CSXGRAOD | Grape | Gasoline 444, **Grape 2192**, Rose 774, Peanut 109 |
| CSXONOD | Onion | Chocolate 53, Strawberry 33, **Onion 3321**, Fruit punch 112 |
| CSXNGSOD | Natural Gas | Orange 111, Cinnamon 184, Cola 175, **Natural Gas 3049** |

Joins to the rest of NHANES on SEQN, so age, sex, and the full demographic/health battery come free.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
Codebook: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/CSX_H.htm — fetched, **open**.
Data: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/CSX_H.xpt — fetched, **open**,
HTTP 200, 1,196,800 bytes, parsed successfully. No application, no registration, no DUA.
The older path `/Nchs/Nhanes/2013-2014/CSX_H.htm` now returns 404 — use the `/Public/2013/DataFiles/`
form above.

Caveat: the equivalent 2011–2012 file (CSX_G) is the same exam but **restricted to the NCHS Research
Data Center**. Only the one cycle is public.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Usable now — the best zero-friction chosen-descriptor dataset found.** Largest fully public
per-subject dataset in this directory (~3,520 × 8 = ~28,000 choices), no gatekeeper, and the pooled
counts serve as a built-in correctness check on any parse. Read with `pandas.read_sas(...,
format='xport')` or `pyreadstat`; neither was installed in this session, hence the hand-rolled parser.

Limitation to be honest about: **it cannot test a menu effect by itself.** Fixed 4-AFC, one condition.
Use it as the anchor for the fixed-menu shares, and pair it with UPSIT for the single-swap comparison
or with Negoias's published descriptor sets if you re-run the nested design.

## What the authors concluded, quoted verbatim where possible
Not an inferential paper — a survey data release, so there is no authorial conclusion to quote. The
codebook documents the design intent: an 8-item forced-choice test where respondents must choose one
of four printed descriptors even if they perceive no odor, which is why the distractor distributions
are informative rather than degenerate. NHANES analytic guidance requires the survey weights for any
population estimate; unweighted counts as above are fine for choice-structure work but not for
prevalence claims.
