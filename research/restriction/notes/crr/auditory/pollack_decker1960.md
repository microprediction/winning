# Pollack & Decker (1960) — 8 initial consonants master, three 4x4 subsets, swept over S/N

## Citation

Pollack, I., & Decker, L. (1960). Consonant Confusions and the Constant Ratio Rule. *Language and
Speech*, **3**(1), 1-6. doi:10.1177/002383096000300101
Both authors: Operational Applications Laboratory, Air Force Cambridge Research Center, Bedford,
Massachusetts.

The lead article of *Language and Speech* volume 3, issue 1 — the journal's own opening statement on
the CRR.

## Stimuli and master response set

Spoken **initial English consonants heard against noise**. The master set is 8 items and is stated
exactly in the abstract, verbatim:

> "/f, h, l, r, w, y/, the cluster /hw/ and the absence of the initial consonant /#/"

So the master response set is the 8 alternatives {f, h, l, r, w, y, hw, #}, giving an **8x8 master
confusion matrix**. Note that one alternative is the *null* consonant /#/ — a "nothing there"
response inside a closed set, which is unusual and useful: it is a genuine alternative that a
Thurstonian model must give a latent value.

**Signal-to-noise ratio was swept over a wide range**, so the design is a family of 8x8 masters
indexed by S/N rather than a single master. Verbatim: "*over a wide range of S/N ratios*".

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Nested. Three 4x4 subsets of the 8-item master, at each S/N.** Verbatim from the abstract:

> "The average deviation between observed consonant confusions for three sets of 4 times 4 matrices
> and confusions predicted on the basis of the constant-ratio rule from the 8 times 8 matrix averaged
> about four per cent over a wide range of S/N ratios."

Three 4-subsets of an 8-set: if disjoint they could not exhaust 8 (3x4 = 12 > 8), so the three
quadruples **must overlap**. That is more informative than a partition, because overlapping subsets
give multiple independent estimates of the same odds ratio and hence a within-experiment consistency
check on the rule that a partition cannot give. Which three quadruples were used is not recoverable
from the abstract and is the key design fact to extract on retrieval.

The stimulus set and the allowable response set are restricted together (the standard CRR protocol of
this literature).

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

NOT DIRECTLY VERIFIED (paywalled, 6 pages). What is established:

- **Both master and subset matrices exist as printed data at multiple S/N values**, since the reported
  quantity is a per-cell deviation between observed 4x4 entries and 4x4 entries predicted from the 8x8
  entries, averaged "over a wide range of S/N ratios". You cannot report that without the matrices.
- **Deviations are in percentage-point units** ("averaged about four per cent"), so entries are
  proportions/percentages, not raw counts. Whether raw counts or trial totals are also printed is
  unknown, and matters: without N, no likelihood-ratio test is possible and only descriptive
  comparison is available.
- **A derived structural result is printed**: "*A tentative representational structure for the
  selected consonants, based on the confusion analysis, is presented.*" — i.e. at least one
  figure/diagram of consonant relations, in the pre-MDS style of 1960.
- Six pages carrying several 8x8 matrices plus several triples of 4x4 matrices at multiple S/N values
  is a lot. It is likely that **only a subset of the S/N conditions has full matrices printed**, with
  the rest summarised by the average-deviation statistic. This is the specific risk to check.
- Table numbers unknown.

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://api.crossref.org/works/10.1177/002383096000300101` — FETCHED 200. **Open.** Full author
  abstract (quoted above), both affiliations, volume 3, pages 1-6.
- `https://api.unpaywall.org/v2/10.1177/002383096000300101?email=...` — FETCHED 200. `is_oa: false`,
  `oa_status: "closed"`, zero OA locations.
- `https://journals.sagepub.com/doi/10.1177/002383096000300101` — FETCHED via WebFetch, **HTTP 403**;
  and via curl, **403** (Cloudflare "Just a moment..."). **Paywalled.**
- Wayback CDX for `*sagepub.com*002383096000300101*` and
  `journals.sagepub.com/doi/abs/10.1177/002383096000300101` — FETCHED, **no snapshots**. Not even
  Wayback-only.
- `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1177/002383096000300101` — FETCHED 200,
  `openAccessPdf.status: "CLOSED"`, abstract elided by publisher.
- Also tried without success: Internet Archive Scholar, fatcat (timeouts), CORE, archive.org
  full-text, Europe PMC (not indexed), HathiTrust (403), ResearchGate (403). AFCRC technical-note
  version: not located — see the note on `egan1957_techreport5750.md` for the full record of DTIC
  routes tried, all of which returned zero results.

Worth one targeted attempt that I could not make from here: Pollack and Decker were at the **Air Force
Cambridge Research Center**, and AFCRC-TN technical notes of 1958-1960 are the obvious parallel
publication channel (cf. Anderson's AFCRC-TN-58-60 from the same programme). A DTIC search on
"Pollack" + "consonant" + 1959-1960 with a human browser session is the highest-value cheap shot at
getting the full matrices free.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NEEDS LIBRARY ACCESS, then digitizing. Top-three priority in this branch.** Reasons it may be the
single best target after Clarke (1957):

1. The master set is **fully specified in the open abstract** ({f, h, l, r, w, y, hw, #}), so the
   analysis can be written before the paper arrives.
2. The three 4x4 subsets **overlap**, giving redundant constraints on the same odds ratios — the
   sharpest possible test of proportional renormalization, and the case where Gaussian and Gumbel
   renormalization diverge most legibly.
3. The S/N sweep gives a **discriminability axis**. Thurstone-sigma and Luce-gamma predictions
   separate as a function of overall confusability, so a sweep is worth more than any single
   condition. This is the closest thing in the 1960s auditory literature to a designed
   Gaussian-vs-Gumbel experiment.

Six pages, one Sage paywall. Cheap.

## What the authors concluded about CRR, quoted verbatim where possible

The authors' abstract, verbatim and in full:

> "The constant-ratio rule of Clarke was evaluated with spoken initial English consonants heard
> against noise: /f, h, l, r, w, y/, the cluster /hw/ and the absence of the initial consonant /#/.
> The average deviation between observed consonant confusions for three sets of 4 times 4 matrices and
> confusions predicted on the basis of the constant-ratio rule from the 8 times 8 matrix averaged about
> four per cent over a wide range of S/N ratios. A tentative representational structure for the
> selected consonants, based on the confusion analysis, is presented."

A conclusion FOR the rule, on a 4-percentage-point average deviation, with no statistical test — the
same evidentiary standard Morgan (1974) later showed to be inadequate on Clarke's and Egan's data.
Note that Pollack & Decker do not report the *distribution* of deviations, only the average, so the
paper as summarised cannot rule out large systematic failures in particular cells; Hodge (1967) later
found exactly such localised failures (adjacent, highly confusable pairs) in comparable designs.

How the branch classified this result, verbatim from Hodge (1967, *Percept. Psychophys.* 2, p. 429):

> "A number of tests has shown that the CRR makes satisfactory predictions of the confusion patterns
> associated with multidimensional objects such as various speech stimuli (Clarke, 1957; Clarke &
> Anderson, 1957; Carterette & Wyman, 1961; Pollack & Decker, 1960), words (Egan, 1957), visually
> presented monosyllables (Anderson, 1959), and multidimensional auditory tones (Hodge & Pollack,
> 1962)."

And from Townsend & Landon (1982, *JMP* 25, p. 122), Pollack & Decker appear in the list of studies
that "concluded that the CRR does make satisfactory predictions with choice data" — followed
immediately by the caveat that all of them lacked "any statistical test of the deviations of the
results from the predictions of the CRR".
