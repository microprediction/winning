# Morgan (1973a) — prints Conrad's 26-letter and Hull's 35-item auditory masters in full; OPEN but scans unreadable

## Citation

Morgan, B. J. T. (1973). Cluster analyses of two acoustic confusion matrices. *Perception &
Psychophysics*, **13**(1), 13-24. doi:10.3758/BF03207229
M.R.C. Applied Psychology Unit, 15 Chaucer Road, Cambridge CB2 2EF, England.

Same author as Morgan (1974) "On Luce's choice axiom" and first author of Morgan, Chambers & Morton
(1973). This is the paper cited as "Morgan (1973a)" in that trio.

The two source datasets, both auditory recognition in white noise:

- **Conrad, R. (1964). Acoustic confusions in immediate memory.** *British Journal of Psychology* 55,
  75-84. doi:10.1111/j.2044-8295.1964.tb00899.x
- **Hull, A. J. (1973). A letter-digit matrix of auditory confusions.** *British Journal of Psychology*
  **64**(4), 579-585. doi:10.1111/j.2044-8295.1973.tb01384.x (cited by Morgan as "Hull, in press")

**PDF obtained (bronze OA via Wayback). The tables are printed but the available scan cannot be read.**

## Stimuli and master response set

Two masters, described verbatim by Morgan (p. 13):

> "This paper deals with the acoustic confusion matrices of Conrad (1964) and Hull (in press). both
> derived from recognition tasks. In Conrad's case. the stimulus set comprised the letters of the
> alphabet; he used 10 untrained speakers, the presentation rate of the stimuli was one letter every 5
> sec, and the sound level of the speech signal was continuously monitored and (on average) an equal
> amount of white noise added. Three-hundred Ss were used. and the final overall error rate was 61%. In
> Hull's case. the stimulus set comprised the letters of the alphabet and all the digits. 1-9; six
> trained speakers were used. the presentation rate of the stimuli was one every 5 sec, and again equal
> noise on speech was employed. In this case, 135 Ss were used and the final overall error rate was
> 34.5%. In both cases. the stimulus lists were balanced with respect to several desirable
> characteristics."

So:
- **Table 1 = Conrad (1964): 26 x 26**, letters A-Z, 300 subjects, 10 untrained speakers, 61% error rate,
  approximately 1,440 presentations per item (from the table caption, OCR-degraded).
- **Table 2 = Hull (1973): 35 x 35**, letters A-Z plus digits 1-9, 135 subjects, 6 trained speakers,
  34.5% error rate.

Orientation, verbatim (p. 13): "*the results of the experiments were summarized by a stimulus/response
matrix, the (ij)th element of which denotes the number of times that 'i' was the response to the
presentation of Stimulus j*" — so rows are responses, columns are stimuli, entries are **counts**. Note
also: "*Note that the letter 'z' has the English. rather than the American. pronunciation.*"

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Within either experiment: NONE.** Each is a single closed master with no subset conditions. On the
brief's inclusion criterion this paper does not qualify.

**Across the two experiments: an accidental nesting that is nearly, but not quite, a CRR test.**
Conrad's 26-letter set is a **proper subset** of Hull's 35-item letter-plus-digit set, and both are
auditory recognition of spoken alphanumeric names in equal-level white noise at the same 5-second
presentation rate, from the same laboratory tradition. That is a 26-in-35 nested restriction over the
same stimulus type.

It is *not* a clean test, and the confounds must be stated plainly: different subjects (300 vs 135),
different talkers (10 untrained vs 6 trained), and above all **very different overall error rates (61%
vs 34.5%)**, so the two matrices sit at different points on the discriminability axis. A CRR
comparison across them conflates set-size with difficulty. Morgan himself attributes cross-table
differences to "*the particular levels of background noise employed in the two experiments*".

Even so, the direction and magnitude of the cross-table odds shifts are startling and worth recording.
From Morgan's **Table 4** (transcribed below), the asymmetric pair (K,A):

- In Conrad's 26-letter set: K->A = 200, A->K = 46. Odds ratio 4.3.
- In Hull's 35-item set: K->A = 488, A->K = 8. Odds ratio 61.

A fourteen-fold change in the odds ratio between the same two letters when nine digits are added to the
response set. Under proportional renormalization the *within-row* ratios among surviving alternatives
should be preserved; even allowing generously for the noise-level confound, a shift of that size is not
what a scale-free renormalization predicts. Treat this as a striking suggestive observation, not
evidence, and treat it as motivation to acquire a properly controlled dataset.

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

**Both full master matrices are printed as raw counts: Table 1 (26x26, Conrad) and Table 2 (35x35,
Hull). Also Table 3 (the symmetrized similarity matrix derived from Table 1, rows and columns permuted
to match the clustering of Fig. 3), Table 4 (selected asymmetric confusions from Tables 1 and 2 side by
side), Table 5 (Wickelgren's 1965c 8x8 vowel-consonant digram confusions in short-term memory), Table 6,
and Figs. 1-5 (SLINK dendrograms, B(2) clusters, MDSCAL projections).**

**Critical caveat: Tables 1, 2 and 3 are unreadable in the freely available scan.** The Springer
bronze-OA PDF is a low-quality scan of the table pages; `pdftotext` returns garbage for those pages
(digits rendered as `I`, `II`, `1'1\`, `.'<:`, `Ihl`, `4luf~1` and so on). Individual values cannot be
recovered, and neither can the row and column marginals needed to arithmetic-check a transcription. The
body text of the paper OCRs cleanly; only the dense numeric tables are lost.

**Table 4, which does OCR cleanly, transcribed in full** — "Asymmetric Confusions from Tables 1 and 2
that Can Be 'Explained' By an Initial Masking Effect". Table 1 = Conrad 26-letter; Table 2 = Hull
35-item:

| Stimulus | Response | Table 1 (Conrad, 26) | Table 2 (Hull, 35) |
|---|---|---|---|
| K | A | 200 | 488 |
| A | K | 46 | 8 |
| J | A | 151 | 78 |
| A | J | 10 | 5 |
| V | E | 119 | 66 |
| E | V | 55 | 41 |
| P | E | 111 | 188 |
| E | P | 171 | 33 |
| B | E | 271 | 153 |
| E | B | 107 | 107 |
| C | E | 195 | 39 |
| E | C | 14 | 18 |
| D | E | 252 | 181 |
| E | D | 166 | 66 |
| T | E | 163 | 31 |
| E | T | 124 | 45 |
| G | E | 208 | 97 |
| E | G | 34 | 21 |

(The "111" for P->E in Table 1 is OCR'd as "III"; read as 111. No marginals are available to check these
against, so treat them as single-source transcriptions.)

**Transcription caution.** The body text contains a sentence, split across two columns and badly
interleaved by OCR, that reads in part "*Table 2. the worst being A to K (8) and K to A (488), ... and
in Table 1. K to A (127) should be compared with N...*". The value 127 conflicts with the 200 in Table
4 for the same cell. One of the two is misread, or the text refers to a different cell. **Do not rely on
the K->A Table 1 value until a clean scan is checked.**

Useful additional data that DOES OCR cleanly: **Table 5**, an 8x8 count matrix of vowel-consonant digram
confusions in short-term memory from Wickelgren (1965c), rows = response, columns = stimulus, first rows
reading 642/62/86/16/44/40/23/17; 38/567/13/42/23/26/14/9; 51/20/622/35/17/19/39/29;
31/79/75/587/34/27/43/38; 28/26/15/7/531/32/25/14; ... (remaining rows not extracted here). Tangential
to the CRR question but a clean, open, small auditory-memory matrix.

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://web.archive.org/web/20181030110140if_/https://link.springer.com/content/pdf/10.3758%2FBF03207229.pdf`
  — **FETCHED 200, 1,183,256 bytes, PDF. This is the URL that worked.** Wayback returned 503 on six
  consecutive attempts before succeeding on the seventh; retry patiently.
- `https://link.springer.com/content/pdf/10.3758/BF03207229.pdf` — canonical URL. **OPEN (bronze OA):**
  `https://api.unpaywall.org/v2/10.3758/bf03207229?email=...` FETCHED 200, `is_oa: true`,
  `oa_status: "bronze"`, this URL as the publisher OA location. Direct curl returned the 3 KB Springer
  "Client Challenge" bot page during this session; a human browser will get the PDF.
- `https://api.crossref.org/works/10.3758/bf03207229` — FETCHED 200, **open**, metadata (volume 13,
  pages 13-24, Feb 1973). No abstract deposited.
- `http://web.archive.org/cdx/search/cdx?url=link.springer.com/content/pdf/10.3758/BF03207229.pdf&matchType=prefix`
  — FETCHED, confirms snapshots at 20181030110140 (200, application/pdf) and 20231203110007 (revisit).

Access to the two underlying sources:

- **Hull (1973)**, *BJP* 64, 579-585: `https://api.crossref.org/works/10.1111/j.2044-8295.1973.tb01384.x`
  — FETCHED 200, **open**, complete abstract retrieved. `https://api.unpaywall.org/v2/...` FETCHED 200,
  `is_oa: false`, `oa_status: "closed"`. Wiley paywall. Hull's abstract, verbatim:
  > "A matrix is presented of the errors of perception made by 135 men and women listening to three male
  > and three female speakers reading aloud different randomized lists constructed from the letters of
  > the alphabet and the digits 1-9, heard in white noise. Data from a short-term memory (STM)
  > experiment, using simultaneous visual presentation and immediate ordered recall of two selected
  > vocabularies of nine letters and the digits 1-9, are cited as evidence of phonemic confusion between
  > letters and digits in STM."
  Note "*two selected vocabularies of nine letters and the digits 1-9*" — **that is a restricted-set
  manipulation in the STM half of Hull (1973)**, and it makes Hull a candidate spine member in its own
  right. Worth checking.
- **Conrad (1964)**, *BJP* 55, 75-84, doi:10.1111/j.2044-8295.1964.tb00899.x — Wiley, paywalled.
  Widely reproduced elsewhere; Conrad's matrix is a standard dataset in the short-term-memory literature.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NEEDS DIGITIZING FROM A BETTER SCAN — the paper is open but the numbers are not legible in the
available copy.** This is a distinct failure mode from the paywalled papers and should be recorded as
such: the data are published, free, and in hand, and still unusable.

Three routes to fix it, in order of cost:

1. **Get a better scan of the same paper.** The Springer bronze PDF is a poor scan; Springer's own
   current delivery, or a library microfilm/print copy of *Perception & Psychophysics* 13(1), may be
   legible. Cheapest possible fix.
2. **Go to the sources.** Hull (1973) *BJP* 64, 579-585 prints the 35x35 matrix as its whole point ("*A
   matrix is presented...*"), and Conrad (1964) *BJP* 55, 75-84 prints the 26x26. Two Wiley paywalled
   papers, both short, both likely better typeset than a reprint of a reprint.
3. Manual key-in from an image, if a legible image can be had. 26x26 = 676 cells and 35x35 = 1,225
   cells; with row and column marginals available for checking this is a few hours of careful work and
   worth it, because a 35-item auditory master is the largest in the branch.

**On the brief's criterion this paper does not qualify** (no restricted response sets within an
experiment), so do not treat it as a CRR test. Treat it as (a) two large open auditory master matrices
awaiting a legible scan, (b) the Table 4 cross-set comparison transcribed above, and (c) a pointer to
Hull (1973), whose STM half *does* use "two selected vocabularies" and may be a genuine
master-plus-restriction study.

## What the authors concluded about CRR, quoted verbatim where possible

**Morgan does not discuss the constant-ratio rule in this paper.** It is a methods paper about cluster
analysis and multidimensional scaling. His abstract, verbatim and in full:

> "Three methods of cluster analysis are used to illustrate two acoustic confusion matrices. It is shown
> how the methods complement each other and. together, 'explain' the large, unwieldy matrices."

The nearest relevant statement is his motivation for macroanalysis, verbatim (p. 13):

> "As they stand. the matrices convey a certain amount of basic information about the confusions; for
> example. it is immediate and not surprising that B and C confuse more readily than B and R in each
> case. However. it is clear that the matrices are poor illustrators of all the information they
> contain."

And his structural conclusion about asymmetry, which is the substantive finding relevant to
renormalization, verbatim (pp. 23-24):

> "This initial masking effect can, in fact. explain some of the asymmetries in Tables 1 and 2. for were
> a digram stimulus. such as K. susceptible to initial masking. it would tend to be confused for A,
> while no initial masking effect could explain the A to K confusion. We might. therefore. expect higher
> K to A than A to K confusions, and this is exactly what occurs. All such pairs are given in Table 4.
> It is only in the (P,E) and (T,E) confusions. presented in Table 4, that the data cannot be explained
> by an initial masking effect."

and, on the failure of the scaling to recover phonetic features, verbatim (p. 22):

> "So we have not been able to identify the MDSCAL dimensions with phonetic features."

Morgan's CRR verdict is in his other two 1973-74 papers, not this one. See `morgan1973.md` ("somewhat
dubious ... 'constant ratio rule'") and `morgan1974.md` (the likelihood-ratio test that rejected it on
Clarke's and Egan's data).
