# Clarke & Anderson (1957) — 10-item master, two 5-item subsets, naive listeners

## Citation

Clarke, F. R., & Anderson, C. D. (1957). Further Test of the Constant-Ratio Rule in Speech
Communication. *Journal of the Acoustical Society of America*, **29**(12), 1318-1320.
doi:10.1121/1.1908778
Both authors: Hearing and Communication Laboratory, Department of Psychology, Indiana University,
Bloomington, Indiana. Published December 1957.

(Townsend & Landon 1982 cite the title as "Further **tests** of the constant-ratio rule in speech
communication"; Crossref and the AIP record both read "Further **Test**". Use the Crossref form.)

## Stimuli and master response set

Speech material (not named in the abstract; the laboratory's monosyllable and digit vocabularies are
the candidates — the 10-item master and 5-item halves fit the digit set naturally).
**Master response set: 10 items, one 10x10 closed confusion matrix.**
**Listeners were naive**, which is the point of the paper: Clarke (1957) used practised listeners,
and this note tests whether the CRR survives the practice manipulation.

Hodge (1967, p. 429) states the purpose: "*Clarke and Anderson (1957) have shown that the CRR makes
similar predictions for practiced and naive Ss.*"

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Nested, and apparently a clean partition.** Two five-item subsets drawn from the ten-item master.
Verbatim from the abstract: "*The use of the constant-ratio rule to predict the confusion matrices
for each of two five-item subsets given the confusion matrix for a ten-item master set is tested with
naive subjects.*"

The two subsets differ substantially in difficulty (obtained articulation scores 68.9% vs 82.6%),
which is consistent with a hard-half / easy-half split rather than a random split. Whether they are
disjoint (a true partition of the 10 into 5+5) or overlapping cannot be settled from the abstract,
though "two five-item subsets" of a ten-item master most naturally reads as disjoint.

Stimulus set and allowable response set are restricted together (Clarke's standing design
assumption).

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

NOT DIRECTLY VERIFIED (paywalled, 3 pages). What the abstract establishes:

- **Cell entries are reported as proportions**, and predicted-vs-obtained deviations were tabulated
  cell by cell: "*Ninty [sic] percent of the predicted cell entries (expressed as proportions)
  deviated by less than 0.05 from the obtained cell entries.*" A statement about the distribution of
  per-cell deviations across 90% of cells implies the cell-level numbers were computed and, in a
  3-page note, most likely printed.
- **Summary articulation scores are printed**: predicted 67.9% vs obtained 68.9% for subset 1;
  predicted 78.4% vs obtained 82.6% for subset 2.
- A 10x10 master plus two 5x5 subsets is 150 cells — printable in three JASA pages, but tight. It is
  possible that only the two 5x5 predicted/obtained pairs are printed and the 10x10 master is
  summarised. **This is the specific fact to check on retrieval.**
- The .05 deviation criterion here is tighter than the .10 criterion Clarke (1957) proposed, which
  suggests the authors regarded this as a strong confirmation.

Table numbers unknown.

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://api.crossref.org/works/10.1121/1.1908778` — FETCHED 200. **Open.** Full author abstract,
  both affiliations, pagination 1318-1320.
- `https://api.unpaywall.org/v2/10.1121/1.1908778?email=...` — FETCHED 200. `is_oa: false`,
  `oa_status: "closed"`, zero OA locations.
- `https://api.openalex.org/works/https://doi.org/10.1121/1.1908778` — FETCHED 200. Single location,
  `is_oa: false`; 19 citing works, all of which I enumerated (Luce 1977 "The choice axiom after
  twenty years", Townsend & Landon 1982, Hodge 1967, Hodge & Pollack 1962, Pollack & Decker 1960,
  Holloway 1970, Holloway 1971, Morgan 1974 "On Luce's choice axiom", and non-auditory descendants).
- `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1121/1.1908778?fields=...` — FETCHED 200,
  `openAccessPdf.status: "CLOSED"`.
- Wayback CDX for `*aip.org*1.1908778*` and `*scitation.org*1.1908778*` — FETCHED, **no snapshots**
  of any landing page or PDF for this DOI (unlike the 1957 Exp-I paper, which has a 2014 Scitation
  snapshot).
- Publisher page `https://pubs.aip.org/asa/jasa/article/29/12/1318/...` — AIP is Cloudflare-403 to
  both curl and WebFetch from this environment; **paywalled** in any case.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NEEDS LIBRARY ACCESS.** Three pages, so cheap to obtain and cheap to key in. Second-highest
priority in the branch after Clarke (1957) itself, for three reasons: (a) 10 -> 5+5 is the cleanest
nested-restriction geometry in the whole auditory literature and needs no interpretation; (b) it is a
naive-listener design, so it is not contaminated by the practice effects that Hodge (1967) later
showed to change CRR accuracy systematically; (c) the two subsets differ by ~14 percentage points in
articulation score, which is exactly the regime where a Gaussian (Thurstone) and a Gumbel (Luce)
renormalization make measurably different predictions.

Risk to flag before committing effort: it is possible the 10x10 master is not printed. If so the
paper drops to unusable on its own and becomes useful only in combination with Clarke (1957).

## What the authors concluded about CRR, quoted verbatim where possible

The authors' own abstract, verbatim and in full:

> "The use of the constant-ratio rule to predict the confusion matrices for each of two five-item
> subsets given the confusion matrix for a ten-item master set is tested with naive subjects. Ninty
> percent of the predicted cell entries (expressed as proportions) deviated by less than 0.05 from the
> obtained cell entries. The predicted articulation score for the first subset was 67.9%, and the
> obtained articulation score was 68.9%. For the second subset the predicted and the obtained
> articulation scores were 78.4% and 82.6%, respectively."

That is a conclusion FOR the CRR, with no statistical test and with an eyeball criterion. Note that
even on the authors' own favourable reading, subset 2 is off by 4.2 percentage points of articulation
score in the direction of the CRR *under*-predicting performance on the easier subset — the same
systematic direction Hodge (1967) later found repeatedly and attributed to response bias.

Hodge (1967, p. 429) summarises the paper's contribution as a practice-invariance result:

> "Clarke and Anderson (1957) have shown that the CRR makes similar predictions for practiced and
> naive Ss, but it seemed important to test their result with other stimulus objects. To the extent
> that the CRR is not affected by practice, the rule is broadened because the behavioral changes
> associated with practice typically represent stimulus and response interactions involving more than
> just pairs of stimuli and their responses."

Hodge then found the opposite for visual and kinesthetic stimuli: "*All the 4 by 4 and 2 by 2
difference measures in each series declined systematically with practice on the 8 by 8 matrix, thus
indicating that CRR predictions improve with practice on the task.*" (Hodge 1967, p. 433.)
