# Clarke & Anderson 1957 — Further test of the constant-ratio rule in speech communication

## Citation

Clarke, F. R. and Anderson, C. D. (1957). "Further Test of the Constant-Ratio Rule in Speech
Communication." *Journal of the Acoustical Society of America* 29(12): 1318-1320.
doi:10.1121/1.1908778. OpenAlex W1993689353.

Not to be confused with two other Clarke items in this sweep: the seed paper Clarke (1957),
*JASA* 29(6): 715-720, doi:10.1121/1.1909023, and the one-page meeting abstract Clarke (1959),
*JASA* 31(6 Suppl): 835, doi:10.1121/1.1930396 (see "Usability verdict" of the sweep report —
that one is an abstract with no numbers and reports a CRR *failure* for tonal displays).

## Domain and stimuli

Auditory / speech intelligibility. A **ten-item master set** of spoken items presented to
**naive** listeners (the paper stresses naivety, in contrast to the highly practised crews used
elsewhere in this literature). The specific ten items, the talker arrangement, the masker and
the S/N ratio are stated in the article body, which I was unable to obtain — the published
abstract names only the set sizes and the resulting articulation scores. Aggregate articulation
scores are reported, so the data are almost certainly aggregate over listeners.

## Master and restricted response sets

Yes — this is a purpose-built master-plus-restricted CRR test, and it is the closest existing
human-data analogue of the project's design. From the published abstract, verbatim:

> "The use of the constant-ratio rule to predict the confusion matrices for each of two
> five-item subsets given the confusion matrix for a ten-item master set is tested with naive
> subjects."

- **Master menu:** one **10 x 10** confusion matrix over ten items, all ten response
  alternatives available.
- **Restricted menus:** **two disjoint (or at least distinct) five-item subsets**, each run as
  its own **5 x 5** condition with only those five responses allowed, over the same stimuli.
- **Direction of prediction is out-of-sample in the right direction**: the 10x10 is the
  calibration set and the two 5x5s are predicted from it by proportional renormalisation. This
  is exactly the full-menu-calibrate / restricted-menu-predict protocol the project uses, and
  the paper scores it **cell by cell**, not merely on percent correct.

## What numbers are printed or deposited

**Partly known, partly UNVERIFIED — I could not obtain the article body.** What the abstract
guarantees is printed or stated:

- A cell-level accuracy statistic over the restricted menus: "Ninty percent of the predicted
  cell entries (expressed as proportions) deviated by less than 0.05 from the obtained cell
  entries." [*sic*, "Ninty" is the publisher's own typo, reproduced identically by Crossref and
  OpenAlex.] A per-cell criterion of 0.05 means the authors held obtained and predicted
  matrices side by side; whether both are tabulated in the paper is what remains unverified.
- Four articulation (percent-correct) scores: subset 1 predicted **67.9%** vs obtained
  **68.9%**; subset 2 predicted **78.4%** vs obtained **82.6%**. Note the sign: CRR
  *under*-predicts obtained accuracy on both subsets, by 1.0 and 4.2 points — the opposite sign
  to the over-prediction Pollack & Decker (1960) and Hodge & Pollack (1962) report.
- Three pages (1318-1320) is ample room for a 10x10 master matrix plus two 5x5 obtained/predicted
  pairs, and the seed paper Clarke (1957) in the same volume does print its matrices, so the
  prior that the tables are there is good — but I have not seen them and am not asserting it.
- No data deposit (1957).

**This is the single highest-value acquisition target in the auditory/speech cluster.** Getting
the PDF (interlibrary loan, an institutional AIP/JASA subscription, or a library scan of *JASA*
vol. 29 no. 12) should settle whether the project has a directly scoreable human dataset. Note
that the Internet Archive's microfilm holdings of *JASA* stop at 1945 for full issues — only
annual index volumes exist for 1946-1969 — so the Internet Archive route that worked for
Pollack & Decker and Hodge & Pollack does **not** work here.

## Access with a fetched url

Metadata and the complete publisher abstract obtained; **article body NOT obtained**.

- https://api.crossref.org/works/10.1121/1.1908778 (fetched; returns the publisher's own JATS abstract verbatim, quoted above, plus pages 1318-1320 and date 1957-12-01)
- https://api.openalex.org/works/W1993689353 (fetched; title, both authors, venue, volume, pages, DOI)
- https://api.openalex.org/works/doi:10.1121/1.1908778?select=title,abstract_inverted_index,biblio (fetched; abstract reconstructed from the inverted index, matching Crossref word for word including the "Ninty" typo)
- https://api.openalex.org/works?filter=cites:W1993689353 (fetched; 19 citing works, used to confirm forward citations and to find Hodge 1967)

Attempted and failed: https://pubs.aip.org/asa/jasa/article/29/12/1318/742857 returns HTTP 403
to automated fetches; the Internet Archive has no full-text scan of *JASA* vol. 29
(https://archive.org/advancedsearch.php?q=identifier%3Asim_journal-of-the-acoustical-society-of-america*
was fetched and returns only `sim_journal-of-the-acoustical-society-of-america_1957_29_index`,
an index volume); api.fatcat.wiki returned no JSON.

Two independent forward citations corroborate the design and the favourable result, both read
in full: Pollack & Decker (1960), *Language and Speech* 3: 1, p. 1 — "Application of the rule,
to date, has been encouraging (Anderson, 1959 ; Clarke, 1957 ; Clarke and Anderson, 1957 ;
Egan, 1957)"; and Hodge & Pollack (1962), *J. Exp. Psychol.* 63: 129 — "The rule has been
successfully applied in a number of situations employing 'complex' informational displays, e.g.,
words or syllables presented in noise (Clarke, 1957; Clarke & Anderson, 1957; Pollack & Decker,
1960)".

## Usability verdict

**Design is exactly right; sufficiency of the printed numbers is UNVERIFIED.** If the paper
tabulates the 10x10 master matrix and the two obtained 5x5 matrices, it is a clean, directly
scoreable human test-bed: calibrate a parameter-free Gaussian model on the 10x10, predict both
5x5s out of sample, and score against the CRR renormalisation whose cell-level error
distribution ("90% within 0.05") and articulation-score errors (+1.0 and +4.2 points obtained
minus predicted) are already published as the benchmark to beat. If only the summary statistics
are printed, it degrades to the same status as Hodge & Pollack (1962) — but even then the four
printed articulation scores permit a coarse, one-number-per-subset comparison, which is more
than the abstract-only candidates offer.

Classification, pending acquisition of the PDF: **CRR-TEST-BUT-NUMBERS-NOT-PRINTED** — recorded
conservatively because I have not seen the tables. Re-classify as
**CRR-TEST-WITH-PRINTED-MATRICES** if the article body proves to tabulate the matrices, which I
judge more likely than not.

## Conclusion about CRR quoted verbatim

The authors' conclusion as stated in the published abstract (p. 1318), quoted verbatim from the
Crossref record:

> "The use of the constant-ratio rule to predict the confusion matrices for each of two
> five-item subsets given the confusion matrix for a ten-item master set is tested with naive
> subjects. Ninty percent of the predicted cell entries (expressed as proportions) deviated by
> less than 0.05 from the obtained cell entries. The predicted articulation score for the first
> subset was 67.9%, and the obtained articulation score was 68.9%. For the second subset the
> predicted and the obtained articulation scores were 78.4% and 82.6%, respectively."

The Discussion/Conclusions section of the article body was **NOT OBTAINED**. Tried:
doi.org resolution to pubs.aip.org (HTTP 403), the AIP article page directly (HTTP 403),
Semantic Scholar (no open-access PDF, `openAccessPdf.status = CLOSED`), Unpaywall
(no OA location), Internet Archive microfilm (*JASA* full issues end at 1945), and
api.fatcat.wiki (no JSON response). Web search quota for this session was exhausted before
these attempts, so no Google Scholar / ResearchGate mirror hunt was possible.

## Additional section — Clarke (1959) reported CRR FAILING, and the data were never published

Recorded here because it is a substantive point about this literature's **publication record**,
not about any one paper, and because it is easy to lose: the one CRR test in Clarke's own hands
that came out *negative* exists only as a one-page meeting abstract, its explanation exists only
as a personal communication quoted in someone else's footnote, and its data were never published
at all. Two later reviewers then cited it as a success.

**The negative result.** Clarke, F. R. (1959), "Proportion of Correct Responses as a Function of
the Number of Stimulus-Response Alternatives", *JASA* 31(6 Suppl): 835, doi:10.1121/1.1930396,
OpenAlex W2078466290 — a one-page Acoustical Society meeting abstract, not an article. Its own
abstract, reconstructed verbatim from OpenAlex, tests three models (a "simple threshold model", a
"constant-ratio rule", and signal detectability theory) across four conditions — speech,
sinusoidal signals at varying intervals, and signals varying in amplitude or frequency — and
reports that "the simple model failed to account for any of the data", with simplified versions
of the other two aligning reasonably with most conditions. The specific failure that matters is
for **elementary tonal displays**, i.e. amplitude and frequency stimuli.

**Hodge & Pollack's report of it.** *J. Exp. Psychol.* 63: 129 (1962), Introduction, p. 129:

> "However, a preliminary experiment (Clarke, 1959) with elementary auditory displays, e.g., the
> frequency and intensity of a tone, has suggested that the rule may break down. Considering the
> subtle complexities associated with the responses to such displays, the failure of the CRR—or
> any other single rule—is reasonable."

**The explanation, which exists only as a personal communication.** Same paper, Discussion,
p. 141:

> "The successful application of the rule in the single-dimensional cases of the present
> experiments is directly opposite Clarke's (1959) finding that responses to intensity and
> frequency stimuli cannot be described by the CRR. Clarke's difficulty with the rule, however,
> was due to strong contextual effects present in his data."

That last sentence carries footnote 5, whose entire content is:

> "F. R. Clarke, personal communication, 1960"

So the sole published account of *why* the only negative CRR result from the rule's own author
came out negative is an unsourced attribution to a private letter, in a paper arguing the opposite
conclusion. "Strong contextual effects" is, in the project's vocabulary, precisely a
menu-composition effect on choice shares — i.e. an IIA violation. The data that would show it
were never printed: Clarke (1959) is an abstract, and no full paper corresponding to it appears
anywhere in Clarke's publication list (checked via
https://api.openalex.org/works?filter=raw_author_name.search:Frank%20R%20Clarke,from_publication_date:1954-01-01,to_publication_date:1970-12-31 — fetched; 19 items, of which the CRR-related ones are the 1957 seed paper, this 1959 abstract, the 1957 *JASA* 29(6 Suppl):781 abstract "Confusion Matrices and the Constant-Ratio Rule", and Clarke & Anderson 1957).

**And then it was cited as a success.** Engstrand & Moeller (1967), *Human Factors* 9(5): 440
— full text read, see `engstrand1967.md` — write:

> "The few formal tests of CRR conducted to date have tended to substantiate its predictive
> usefulness. Clarke and Anderson (1957), Clarke (1957, 1959), Pollack and Decker (1960) and
> Hodge and Pollack (1962) successfully predicted performance on auditory tasks."

Clarke (1959) is in that list of successes. It was a failure. Engstrand & Moeller's own reference
list correctly marks it "(Abstract)", so they had the citation in front of them.

**Why this matters to the project.** The received wisdom that CRR/IIA "held up well" in the
classical psychophysics literature rests on a citation record with a survivorship problem at its
centre: the affirmative tests were written up as full papers with summary statistics, the one
negative test from the rule's originator was written up as a one-page abstract, its data were
never published, and its explanation survives only as a footnoted personal communication that a
subsequent review silently converted into a success. This is worth a sentence in the paper. It
also means the literature cannot be used to argue either way at cell level — which is the
independent argument for generating the data directly.

Sources for this section, all fetched: https://api.openalex.org/works/W2078466290;
https://api.openalex.org/works?filter=raw_author_name.search:Frank%20R%20Clarke,from_publication_date:1954-01-01,to_publication_date:1970-12-31;
https://archive.org/download/sim_journal-of-experimental-psychology-general_1962-02_63_2/sim_journal-of-experimental-psychology-general_1962-02_63_2_djvu.txt (pp. 129, 141 and footnote 5);
https://archive.org/download/DTIC_AD0668614/DTIC_AD0668614_djvu.txt (p. 440 and reference list).
