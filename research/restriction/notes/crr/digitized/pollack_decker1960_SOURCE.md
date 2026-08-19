# Pollack & Decker (1960) — digitized confusion matrices

Pollack, I., & Decker, L. (1960). Consonant confusions and the constant ratio rule.
*Language and Speech*, **3**(1), 1–6. DOI `10.1177/002383096000300101`.

Also issued as **Air Force Cambridge Research Center Technical Note 59-17** (stated in
the article's own first footnote), supporting AFCRC Project 7681, *Auditory Presentation
of Information*. Author affiliation at the time was the Operational Applications
Laboratory, AFCRC, Bedford, Massachusetts — **not** Michigan. (Pollack moved to Michigan
later; `deepblue.lib.umich.edu` is not a plausible route for *this* paper.)

## READ THIS FIRST — what the paper does and does not contain

The paper contains **four observed 8×8 master confusion matrices** (Table 2), one per
S/N ratio. It does **not** tabulate the three 4×4 sub-matrices, nor the six 2×2
sub-matrices. For those, only the *mean absolute deviation* between observed and
CRR-predicted entries is published (Table 1), as one scalar per (set, S/N) pair.

So the working assumption in the pre-acquisition notes — that this paper supplies
"three overlapping 4×4 subsets, giving redundant odds constraints" as observed data —
**is not correct about what is printed.** The overlapping subsets were *run*
(/l,r,w,y/, /f,h,l,r/, /f,h,hw,#/, plus 2×2s /f,h/ /f,w/ /h,#/ /l,r/ /r,w/ /w,hw/),
but their observed matrices were reduced to summary deviations before publication.

What this digitization therefore delivers:

- speech-domain confusion data, which the restriction-map comparison previously lacked
  entirely (the literature's origin point, Clarke 1957, is speech);
- four independent 8×8 masters over the same fixed 8-alternative response set across a
  12 dB discriminability sweep (−5, −9, −13, −17 dB);
- Table 1 as a *validation target*: a correct CRR implementation applied to these
  masters must be capable of reproducing those mean-deviation figures, once the observed
  subset matrices are supplied from elsewhere.

What it does **not** deliver: (master, observed-subset) matrix pairs. A parameter-free
restriction map cannot be scored against a rival on this dataset alone. See
"Could the 4×4 matrices be recovered from Fig. 1?" below for the one partial route, and
why it was not taken.

## Source obtained

- **Route that worked:** Internet Archive microfilm serials collection (`sim_*`).
  - Item: `https://archive.org/details/sim_language-and-speech_january-march-1960_3_1`
    ("Language and Speech January-March 1960: Vol 3 Iss 1"; collections
    `pub_language-and-speech`, `sim_microfilm`, `periodicals`; 251 images — the item in
    fact holds the whole of volume 3, not just issue 1).
  - Found by: `https://archive.org/advancedsearch.php?q=identifier:sim_language-and-speech*`
    (56 items; only 1958 v1, 1959 v2, 1960 v3 iss 1 and 1961 v4 iss 1 are full scans,
    the rest are volume indexes).
  - No `access-restricted-item` flag, no lending status, no licence URL. Both the OCR
    text and the full Text PDF download anonymously.
  - PDF fetched: `https://archive.org/download/sim_language-and-speech_january-march-1960_3_1/sim_language-and-speech_january-march-1960_3_1.pdf`
    — HTTP 200, 48,759,541 bytes, 251 pages, page size 462.96 × 629.28 pt, producer
    "Internet Archive PDF 1.4.7", scanned 2021-11-16, with an OCR text layer.
  - OCR text also fetched: `..._djvu.txt` — HTTP 200, 594,721 bytes.
- **Article location in that PDF:** PDF pages **5–10** = journal pages **1–6**.
  - p. 1 (PDF 5): title, abstract, introduction, TN 59-17 footnote
  - p. 2 (PDF 6): Procedure; Results: constant-ratio rule
  - p. 3 (PDF 7): **Fig. 1** (empirical test of the CRR)
  - p. 4 (PDF 8): **Table 1** (mean absolute deviations) and **Table 2** (the four 8×8
    confusion matrices) — this is the page digitized here
  - p. 5 (PDF 9): **Fig. 2** (confusion vectors)
  - p. 6 (PDF 10): discussion, conclusions, references
- **Extraction:** `pdftotext -layout` and `pdftotext -bbox` for the text layer and for
  column x-positions; then `pdftoppm -r 900` crops of Table 1, Table 2 and Fig. 2 read
  visually. The microfilm OCR of Table 2 is severely corrupted (e.g. the −5 dB /r/ row
  came out as `.tnqwhe`, the −9 dB /hw/ row as `;' ££ i ASS. B`), so **every cell in the
  delivered CSVs was read from the 900 dpi image**, with the text layer used only to fix
  column assignment geometrically.

### Acquisition routes tried, with outcomes

| Route | Outcome |
|---|---|
| Crossref API (`api.crossref.org`) | **Success (metadata).** Confirmed DOI `10.1177/002383096000300101`, vol 3(1), pp. 1–6, Jan 1960, plus the full abstract. |
| `doi.org` → `journals.sagepub.com/doi/10.1177/002383096000300101` | **HTTP 403.** Bot-blocked (5,735 byte challenge page), browser UA did not help. |
| `journals.sagepub.com/doi/pdf/10.1177/002383096000300101` (the Crossref text-mining link) | **HTTP 403.** Same block. |
| Unpaywall (`api.unpaywall.org`) | **Closed.** `is_oa: false`, `oa_status: "closed"`, `oa_locations: []`, `has_repository_copy: false`. |
| OpenAlex (`W2725331477`) | **Closed.** `best_oa_location: null`, `any_repository_has_fulltext: false`, single location = the SAGE landing page. |
| Semantic Scholar Graph API (`c93bbb5cadc614ab58c544aceb1ba1fa947f3e3f`) | **Closed.** `isOpenAccess: false`, `openAccessPdf.status: "CLOSED"`, empty URL. |
| Wayback CDX, SAGE landing page | **No usable capture.** Two captures only (2018-06-20), both HTTP 302 redirects (one a `?cookieSet=1` bounce); no page body, no PDF. |
| Wayback CDX, SAGE PDF path | **Zero captures.** |
| `scholar.archive.org` full-text search for the exact title | **Not the paper.** Two hits, both the *same* 1960 PMLA annual bibliography page that merely lists the title ("L&S, 11, 1-6" — note PMLA's own volume typo). Fatcat release `91750dfc-9a8e-4247-808c-47748b040cd5` is that PMLA item. |
| CORE API v3 | **Empty response** (unauthenticated request; would need an API key). Not pursued once IA succeeded. |
| DTIC (`discover.dtic.mil`) | **Inconclusive.** The results page is a client-side JS app and returns no server-rendered hits. Worth a manual retry by a human: the article *is* AFCRC TN 59-17, and its sibling AFCRC report is cited in the same paper's reference list with an ASTIA number (Anderson 1959 = AFCRC TN 58-60, ASTIA AD-160 706), so an ASTIA/DTIC copy of TN 59-17 plausibly exists and would be an independent open source. Not needed here. |
| **Internet Archive `sim_` microfilm serials** | **SUCCESS — full text obtained.** See above. |
| HathiTrust, ResearchGate, `deepblue.lib.umich.edu` | **Not attempted** — unnecessary once IA succeeded. Note the Michigan lead is based on a wrong affiliation (see header). |

WebSearch was unavailable for this session (search budget already exhausted at
200/200), so every route above was reached through direct API/HTTP calls rather than
through a search engine. The Crossref bibliographic query was what pinned the DOI.

## Files delivered

Nine CSVs, all in this directory.

| File | Contents |
|---|---|
| `pollack_decker1960_snr-5db_fhlrwhwyNC.csv` | 8×8, S/N = −5 dB |
| `pollack_decker1960_snr-9db_fhlrwhwyNC.csv` | 8×8, S/N = −9 dB |
| `pollack_decker1960_snr-13db_fhlrwhwyNC.csv` | 8×8, S/N = −13 dB |
| `pollack_decker1960_snr-17db_fhlrwhwyNC.csv` | 8×8, S/N = −17 dB |
| `pollack_decker1960_snr-*_fhlrwhwyNC_asprinted.csv` | the same four, but printed-blank cells left as empty fields instead of `0` |
| `pollack_decker1960_table1_crr_deviations.csv` | Table 1, the CRR mean-absolute-deviation summaries |

First column header is `stimulus`; one column per response.

**Response set and column naming.** The paper's set is
/f, h, l, r, w, y/ + the cluster /hw/ + /#/ (absence of an initial consonant), each
paired with /ɑ/ as in *father* (so /fɑ/, /hɑ/, …). Column order follows the printed
table: `f, h, l, r, w, hw, y, NC`. **`NC` is the paper's `/#/`** — renamed because a
literal `#` in a CSV header is a comment character to many parsers. Note the printed
column order puts `hw` between `w` and `y`, which is *not* the order used in the paper's
prose ("/f,h,l,r,w,y/, the cluster /hw/, and /#/").

## Cell convention

- **Cells are percentages, not counts.** Table 2's caption: "each entry represents the
  percentage of responses associated with each of the stimulus items", and the body text
  (journal p. 5): "Each entry of Table 2 is the nearest rounded percentage entry of the
  confusion matrix." Values are therefore integers 0–100.
- **Rows are presented stimuli, columns are listener responses.** Confirmed three ways:
  the row-block label reads "Stimulus" (rotated) and the column-block label "Response";
  and the body text (journal p. 6) says "at a S/N ratio of −5 db, the response syllable
  /rɑ/ was emitted on 12 per cent of the occasions in which the stimulus syllable /lɑ/
  was read" — the digitized −5 dB `l → r` cell is exactly **12**.
- **Trials per row: n = 360, in every matrix.** Procedure, journal p. 2: "The number of
  observations associated with each alternative of each matrix was 360 observations.
  Thus, the total number of observations associated with each 8 × 8, 4 × 4 and 2 × 2
  matrix was approximately 2900, 1450 and 700 observations, respectively." Consistent:
  8 × 360 = 2880 ≈ 2900, 4 × 360 = 1440 ≈ 1450, 2 × 360 = 720 ≈ 700. So each row of
  each 8×8 here rests on 360 presentations of that stimulus.
- **Blank printed cells are written as `0`.** A blank means the rounded percentage was
  0, i.e. at most 1 of the 360 trials (1/360 = 0.28 %, which rounds to 0). A blank
  therefore cannot distinguish 0 observations from 1. There are 33 such cells:
  21 at −5 dB, 11 at −9 dB, 1 at −13 dB (the `w` row's `#` cell), 0 at −17 dB — i.e. the
  blanks disappear as the noise rises, as expected. They are recorded as empty fields
  in the `_asprinted.csv` variants if you would rather treat them as censored.
  **No cell was left blank because it was unreadable** — see the honesty section.

### Counts are NOT recoverable here (unlike Townsend & Landon 1982)

Do not try to convert these percentages back to counts. With n = 360 the granularity of
one observation is 0.28 %, finer than the 1 % printing granularity, so each printed
percentage `p` is consistent with 3 or 4 different integer counts
(`k ∈ [⌈3.6(p−0.5)⌉, ⌊3.6(p+0.5)⌋]`). The count vector is not identified. This is the
opposite of the Townsend & Landon case, where 3-decimal proportions out of n = 240 forced
the counts uniquely. **Percentages are the primary and only delivered data.**

One exception, and it is derived rather than read — see the asterisk note below: the two
asterisked cells at −17 dB are pinned to exactly 42/360.

## Verification results

**All 256 cells (4 matrices × 8 rows × 8 columns) were resolved. 0 cells are `NA`.**

### (a) Row-sum check — 32/32 rows pass

Each row is a percentage distribution over 8 responses, each entry rounded to the
nearest integer, so the printed row must sum to 100 ± 4 in the worst case. Observed row
sums are only ever **99, 100 or 101** — every row within ±1:

| S/N | f | h | l | r | w | hw | y | # |
|---|---|---|---|---|---|---|---|---|
| −5 dB | 99 | 99 | 99 | 100 | 101 | 101 | 101 | 100 |
| −9 dB | 100 | 99 | 100 | 101 | 100 | 101 | 100 | 100 |
| −13 dB | 99 | 100 | 99 | 99 | 100 | 101 | 101 | 101 |
| −17 dB | 100 | 100 | 101 | 100 | 101 | 99 | 100 | 100 |

Mean of the diagonal (average correct identification) falls monotonically with S/N, as
it must, giving the intended discriminability axis: **78.6 %** at −5 dB, **68.4 %** at
−9 dB, **48.1 %** at −13 dB, **28.9 %** at −17 dB (chance = 12.5 %).

### (b) Integer-count feasibility check — 32/32 rows pass

For each row, asked whether there exists an integer count vector `k` with `Σk = 360` and
`round(100·k_j/360) = p_j` for every printed `p_j`. Feasible iff
`Σ⌈3.6(p_j−0.5)⌉ ≤ 360 ≤ Σ⌊3.6(p_j+0.5)⌋`. All 32 rows are feasible, with 360 comfortably
inside every interval (the narrowest is [352, 366] and the widest [344, 376]).

Note this is a genuine but **weak** check — the intervals are wide because 1 % ≫ 1/360.
The per-cell "proportion × n must be an integer" self-correction used for
Townsend & Landon has essentially no discriminating power at this granularity and was
**not** used. The real work was done by check (c).

### (c) Independent cross-check against Fig. 2 — 48/48 arrows, exact

This is the strong check, and it is independent of Table 2 because Fig. 2 was drawn from
the underlying data, not from the printed table.

Fig. 2 draws one labelled arrow per off-diagonal confusion above a per-panel cutoff.
The cutoffs are stated in the text (journal p. 6): **7.5 %** at −5 dB, **10.1 %** at
−9 dB, **10.6 %** at −13 dB, **11.8 %** at −17 dB. Each arrow carries its percentage.
So the figure independently determines (i) exactly which off-diagonal cells clear the
cutoff and (ii) their values.

Read all four panels at 450–900 dpi and compared with the digitized matrices:

| S/N | cutoff | cells ≥ cutoff in Table 2 | arrows in Fig. 2 | set difference | label mismatches |
|---|---|---|---|---|---|
| −5 dB | 7.5 | 8 | 8 | none | none |
| −9 dB | 10.1 | 8 | 8 | none | none |
| −13 dB | 10.6 | 15 | 15 | none | none |
| −17 dB | 11.8 | 17 | 17 | none | none |

**48 arrows, 48 exact agreements in both cell identity and value.** The check is
sharp at the 10-vs-11 and 12-vs-13 boundaries that OCR most often garbles:

- −9 dB `r → hw = 10` correctly has *no* arrow (10 < 10.1), while all the 11s and 12s in
  that panel do. Confirms 10, not 11.
- −13 dB `f → h = 10` and `w → l = 10` correctly have no arrows (10 < 10.6), while
  `r → y = 11` and `w → y = 11` do. Confirms both 10s.
- −13 dB `w → r = 13`: initially mis-transcribed from the figure as `w → l`, resolved by
  a 900 dpi zoom on the /l,r,w,y/ cluster — the arrow leaves /w/ heading up-and-right
  toward /r/, and `w → l = 10` is below cutoff so cannot be it. The *table* reading
  (`w`-row `r` = 13) was correct throughout; the error was in my reading of the figure.
  This was the only discrepancy in the whole cross-check, and it resolved in favour of
  the table.

One consistency note in the other direction, recorded but not a problem: at −5 dB
`h → # = 9` clears the 7.5 % cutoff and does get an arrow in Fig. 2, even though the
Conclusions describe /f,h,#/ as "a group which is not in evidence except at the most
unfavourable S/N ratios". The paper says the cutoffs were "adjusted in order to obtain
simple structure", so the prose is a summary judgement, not a claim about the arrow set.

### The asterisks at −17 dB

Two cells at −17 dB print as `12*`, with the footnote `*less than 11.8`:
`f → l` and `y → hw`. These are not uncertain readings. The footnote exists because
11.8 % is that panel's Fig. 2 cutoff, and both cells round *up* to 12 while their true
values fall *below* the cutoff — which is why neither has an arrow in Fig. 2. This is
itself a fourth confirmation of the digitization.

It also pins the counts. For a printed 12 the candidate counts are 42, 43, 44, 45
(→ 11.67, 11.94, 12.22, 12.50 %), and only **42** is below 11.8. So
`f → l` and `y → hw` at −17 dB are exactly **42/360 = 11.67 %**. The delivered CSVs
carry the printed `12`; the 11.67 % refinement is recorded here and is a derivation, not
a reading.

### Table 1

Read from a 900 dpi crop. Two OCR slips in the text layer were corrected against the
image: the `l, r, w, y` row's −17 dB entry OCR'd as `33` is **5.5**, and the row labels
`ee`, `2`, `azz` are `f, h, l, r`, `l, r` and `2 × 2`.

| Set | −17 | −13 | −9 | −5 |
|---|---|---|---|---|
| l, r, w, y | 5.5 | 5.2 | 4.5 | 3.7 |
| f, h, l, r | 5.9 | 4.7 | 3.4 | 2.8 |
| f, h, hw, # | — | 3.3 | 1.8 | 4.6 |

The missing −17 dB entry for `f, h, hw, #` is a real absence, not an unread cell: the
Procedure says only the first two 4×4 sets were also run at the lowest S/N ratio. It is
written `NA` in `pollack_decker1960_table1_crr_deviations.csv`.

2×2 predictions, all at S/N = −13 dB (mean absolute deviation, percentage points):

| Set | from 8×8 | from 4×4 |
|---|---|---|
| f, h | 6.8 | 5.0, 5.6 |
| f, w | 15.4 | — |
| h, # | 3.8 | 5.8 |
| l, r | 7.6 | 2.2, 8.1 |
| r, w | 7.4 | 11.3 |
| w, hw | 10.2 | — |

Two values appear for `f, h`, `l, r` where the pair sits inside two different 4×4 sets
(the paper: "Some 2 × 2 combinations were obtained from more than one 4 × 4 matrix").
`f, w` and `w, hw` have no 4×4 parent among the three sets run, hence the blanks.

## Cells NOT resolved

**None.** All 256 cells of the four 8×8 matrices were read directly from the 900 dpi
renders, and all four independent checks pass. No cell is written as `NA` in any of the
eight matrix CSVs, and no value in them is an inference.

The only `NA` anywhere in this release is the single Table 1 entry
(`f, h, hw, #` at −17 dB) that the authors did not run.

## Could the 4×4 matrices be recovered from Fig. 1?

Partially, in principle. Not attempted, and no such numbers are in the CSVs.

Fig. 1 plots, for each cell of each 4×4 matrix, the deviation (observed − CRR-predicted,
ordinate) against the **observed** percentage (abscissa), in four panels by S/N, with
point shape coding the set (circle = /l,r,w,y/, square = /f,h,l,r/, triangle =
/f,h,hw,#/) and filled points marking the diagonal-averaged intelligibility scores. So
each point in principle carries an observed 4×4 value on its abscissa, and
`abscissa − ordinate` gives the CRR prediction — which, since the CRR prediction is
computable exactly from the 8×8 masters digitized here, could be matched back to
identify *which* cell (i, j) each point is.

Three reasons this is a research exercise rather than a digitization, and why it was not
done:

1. Points are unlabelled. Shape gives the set, nothing gives the cell.
2. The paper states outright: "Some points in the densely packed region have been
   omitted." The recovered set would be knowingly incomplete, and you cannot tell which
   cells are missing.
3. The left-hand cluster (observed 0–20 %, where most off-diagonal cells live) has
   heavily overlapping markers on a 1960 microfilm scan. Positions there are not
   readable to the ~1 percentage point that the identification step would need.

The scan quality of Fig. 1 is better than expected — axis ticks and the ±10 dashed
guides are crisp, and the sparse right-hand cluster (observed 55–100 %, the diagonal
cells) is well separated. So recovering the *diagonal* 4×4 cells alone may be tractable.
Recorded as a live option; deliberately not exercised, so that nothing in this directory
is a graph estimate.
