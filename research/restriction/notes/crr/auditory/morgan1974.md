# Morgan (1974) — the likelihood-ratio test that broke the auditory CRR corpus, and cites Thurstone

## Citation

Morgan, B. J. T. (1974). On Luce's choice axiom. *Journal of Mathematical Psychology*, **11**(2),
107-123. doi:10.1016/0022-2496(74)90002-9
Mathematical Institute, University of Kent, Canterbury, Kent, England.
Published May 1974. (Announced in Morgan, Chambers & Morton 1973 as "Morgan, B. J. T. A statistical
test of Luce's choice axiom. *Journal of Mathematical Psychology*, 1973b, in press.")

**This is the pivotal paper of the whole branch and it is not usually recognised as such.** It is the
first and, for a long while, the only paper to apply a proper significance test to the CRR, and it
overturned the field's verdict on the original data.

## Stimuli and master response set

**No new experiment.** This is a statistical paper that reanalyses other people's published matrices.
The datasets it reaches for are recoverable from its reference list, which I retrieved in full from
OpenAlex (40 references). It cites, in one paper, **the entire auditory/speech CRR data corpus**:

- Clarke (1957), *JASA* 29, 715-720 — doi:10.1121/1.1909023
- Clarke & Anderson (1957), *JASA* 29, 1318-1320 — doi:10.1121/1.1908778
- Clarke (1959), *JASA* 31, 835 — doi:10.1121/1.1930396
- Egan (1957), *JASA* 29, 482-489 — doi:10.1121/1.1908936
- Pollack & Decker (1960), *Language and Speech* 3, 1-6 — doi:10.1177/002383096000300101
- Hodge & Pollack (1962), *JEP* 63, 129-142 — doi:10.1037/h0042219

plus its own author's auditory matrices:
- Morgan (1973a), "Cluster analyses of two acoustic confusion matrices", *Percept. Psychophys.* 13,
  13-24 — doi:10.3758/bf03207229 (bronze OA)
- Hull (1973), "A letter-digit matrix of auditory confusions", *British Journal of Psychology* 64,
  579-585 — doi:10.1111/j.2044-8295.1973.tb01384.x (35-element letter+digit auditory master)
- Conrad (1964), "Acoustic confusions in immediate memory", *BJP* —
  doi:10.1111/j.2044-8295.1964.tb00899.x

and, decisively for the present project's framing:
- **Thurstone, L. L. (1927). A law of comparative judgment. *Psychological Review* —
  doi:10.1037/h0070288**
- Luce (1959) *Individual Choice Behavior*; Luce (1963) Handbook chapter; Luce (1963) "A threshold
  theory for simple detection experiments"
- Tversky (1972) "Elimination by aspects"; Restle (1961); Rumelhart & Greeno (1971); Krantz (1967);
  Nakatani (1972) confusion-choice model
- The statistical machinery: **Wilks (1938)** large-sample likelihood-ratio distribution
  (doi:10.1214/aoms/1177732360, bronze OA) and **Hilton (1971)** "An Algorithm for Detecting
  Differences Between Transition Probability Matrices", *JRSS-C* (doi:10.2307/2346633).

So Morgan (1974) is the one place in the 1957-1974 literature where Luce's axiom, Thurstone's 1927
law, and every auditory CRR dataset appear between the same two covers.

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Inherited from the reanalysed sources** — nested restrictions in every case (Clarke's 6->3 and
10->5, Pollack & Decker's 8->4, Hodge & Pollack's 8->4 and 8->2, Egan's message-set sweep). Morgan
contributes no new restriction design; he contributes the test.

Morgan's method treats master and subset matrices as **transition matrices** and asks whether the
subset matrix is consistent with the master under proportional renormalization, using a
likelihood-ratio statistic (Wilks 1938; Hilton 1971). This requires **cell frequencies**, which is why
it is strong indirect evidence that Clarke (1957) and Egan (1957) print counts, not just proportions.

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

NOT VERIFIED (Elsevier paywall; 17 pages). Neither Crossref nor OpenAlex carries an abstract for this
DOI, so I have no author-supplied description of the contents.

What can be said: since Morgan reanalyses Clarke's and Egan's matrices, **there is a real chance the
paper reproduces those matrices** (as re-typeset tables or at least as the fitted/observed pairs with
test statistics). A 17-page *JMP* paper has the room. **If it does, Morgan (1974) is a single-purchase
shortcut to two of the three most wanted datasets in this branch** — better value than buying Clarke
(1957) and Egan (1957) separately, and with the significance tests already computed. This is the
highest-expected-value single check remaining in the whole exercise.

Established results the paper contains, from Townsend & Landon (1982, *JMP* 25, p. 122):
- A likelihood-ratio test of equality between two transition matrices, proposed as the correct test of
  CRR predictions.
- Applied to Clarke (1957) and Egan (1957): both **depart significantly** from CRR predictions.
- A stated caveat that pooling over subjects may explain the discrepancy with the original authors'
  conclusions, and a recommendation to use individual-subject data.

Table numbers unknown.

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://api.crossref.org/works/10.1016/0022-2496(74)90002-9` — FETCHED 200. **Open**, metadata only:
  title, author (Morgan), affiliation "Mathematical Institute, University of Kent, Canterbury, Kent,
  England", volume 11, pages 107-123, published 1974-05-01. **No abstract deposited.**
- `https://api.unpaywall.org/v2/10.1016/0022-2496(74)90002-9?email=...` — FETCHED 200.
  `is_oa: false`, `oa_status: "closed"`, `has_repository_copy: false`, zero OA locations.
- `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/0022-2496(74)90002-9` — FETCHED 200.
  `openAccessPdf.status: "CLOSED"`, abstract null.
- `https://api.openalex.org/works/https://doi.org/10.1016/0022-2496(74)90002-9` — FETCHED 200.
  Full 40-item reference list retrieved (enumerated above). Single closed location.
- ScienceDirect (`sciencedirect.com/science/article/pii/0022249674900029`) — Wayback CDX prefix search
  FETCHED, **no snapshots**. Elsevier landing pages are bot-protected from this environment.
  **Paywalled.**
- **Kent Academic Repository** (`https://kar.kent.ac.uk/cgi/search?q=Luce+choice+axiom`) — FETCHED,
  **HTTP 302** to a redirect stub; no results page obtained. Morgan spent his career at Kent, so KAR
  is the single most likely home for a green-OA copy. **Worth one manual attempt:** search KAR for
  author "Morgan, Byron J. T." and title "On Luce's choice axiom".

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NEEDS LIBRARY ACCESS — and this should be the FIRST paper acquired in the whole branch.**

Reasons it outranks even Clarke (1957) in acquisition order:
1. It may reproduce Clarke's and Egan's matrices, in which case one purchase yields two datasets.
2. It contains the already-computed significance tests, so it settles immediately whether the CRR
   fails on the founding auditory data and by how much — the single most load-bearing empirical claim
   the project can make about this literature.
3. It cites Thurstone (1927) alongside Luce (1959) and Luce's threshold theory, so it is the natural
   place to look for a 1974 statement of the Gaussian-vs-Gumbel question, and the natural paper to
   position the present work against.
4. Seventeen pages of *JMP* — cheap ILL, and Kent Academic Repository may have it free.

If the paper turns out to be purely methodological with no reproduced matrices, its value drops to
citation-only, and the acquisition order reverts to Clarke (1957) -> Egan (1957) -> Pollack & Decker
(1960).

## What the authors concluded about CRR, quoted verbatim where possible

**No abstract for this paper is deposited with Crossref, OpenAlex, or Semantic Scholar**, and the full
text is paywalled, so I have no verbatim sentence from Morgan (1974) itself. This is a gap that should
be closed on acquisition.

The nearest thing to a verbatim statement of Morgan's position is his own summary in the companion
paper, which I did obtain in full. Morgan, Chambers & Morton (1973, *Percept. Psychophys.* 14, p. 380),
verbatim — "Morgan, 1973b" in that sentence is this paper:

> "Unless one subscribes to the somewhat dubious (see Cane, 1960; Morgan, 1973b) 'constant ratio rule'
> (CRR) (Clarke, 1957), also known as 'Luce's choice axiom' (Luce, 1959), then these factors will
> influence the predictions of the experiments and confound any comparisons one might wish to make
> between experiments."

Note "somewhat dubious" — that is the author of the test describing the axiom in 1973, before his own
paper appeared.

Morgan also flags the direction of the expected failure, same paper (p. 382), verbatim:

> "Thus, prediction from the full alphabetic confusion matrix to small vocabularies using the CRR could
> overestimate the errors for low-confusion subsets such as C F J. The reasons for this would be that,
> with such subsets, the vowel alone gives all the information necessary. When these letters occur with
> a larger vocabulary, the consonants must be coded as well, presumably reducing the efficiency of the
> vowel coding."

That direction — CRR over-predicting errors in easy restricted subsets, i.e. under-predicting accuracy
— is exactly the sign Hodge (1967) Table 3 shows for the auditory 2x2 subsets (positive algebraic
diagonal differences in 6 of 8 conditions; see `hodge1962.md`). Two laboratories, two decades, same
signed failure.

And the summary of Morgan's result as the field received it, verbatim from Townsend & Landon (1982,
*JMP* 25, p. 122):

> "However, there is some question as to the validity of the results of these past studies due to what
> was at the time a lack of any statistical test of the deviations of the results from the predictions
> of the CRR. Morgan (1974) pointed out this problem and proposed that a likelihood ratio test of
> equality between two transition matrices (Hilton, 1971; Wilks, 1938) be utilized to statistically test
> CRR predictions. Morgan (1974) applied his likelihood ratio test to the data reported by Clarke (1957)
> and Egan (1957) and found their data to depart significantly from the predictions of the CRR. Both
> Clarke (1957) and Egan (1957) presented their data pooled over subjects and reported that the CRR held
> with their data. Morgan (1974) suggested that the discrepancy between his conclusion and the
> conclusions of Clarke and Egan might have been due to the use of pooled data, and could possibly be
> cleared up through the use of individual subject data and analysis."
