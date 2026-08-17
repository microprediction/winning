# Clarke (1957), Experiment I — CV syllables, three 6x6 masters + six 3x3 submatrices

## Citation

Clarke, F. R. (1957). Constant-Ratio Rule for Confusion Matrices in Speech Communication.
*Journal of the Acoustical Society of America*, **29**(6), 715-720. doi:10.1121/1.1909023
Affiliation: Hearing and Communication Laboratory, Department of Psychology, Indiana University,
Bloomington, Indiana. Received 11 Feb 1957; published June 1957.

This is the paper that names the constant-ratio rule (CRR). It is the root of the auditory/speech
branch. Luce (1959, *Individual Choice Behavior*) later showed the CRR to be equivalent to part (i)
of his choice axiom, i.e. it is proportional renormalization / IIA under another name.

## Stimuli and master response set

Consonant-vowel (CV) nonsense syllables, presented in noise, closed-set identification.
The master set is 6 CVs; **three separate 6x6 master confusion matrices** were obtained
(per the author's own abstract, three masters were examined).

Verified from the abstract, verbatim: "*three 6x6 master matrices for CV's (consonant-vowel
syllables) and six 3x3 submatrices*".

Exact phoneme inventory of the 6 CVs is NOT recoverable from the abstract or from any citing
paper I could reach; it requires the printed article.

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Nested.** Six 3x3 submatrices drawn from the 6-item master sets. Each submatrix uses a
3-element subset of the master stimulus set, with the response alternatives restricted to exactly
that same 3-element subset (a "closed" submatrix in Clarke's terminology). Listeners knew the
restricted set and confined their responses to it.

Because 6 items yield 20 possible 3-subsets and only six were run, the six 3x3 sets are a selected
family, not an exhaustive partition; whether the six subsets are disjoint triples (two disjoint
triples per master x three masters) or overlapping is not determinable without the article. Design
note in Clarke & Anderson (1957) that "the only variables which differ systematically in obtaining
the two matrices are the different sets of messages and the allowable responses" implies stimulus
set and response set were restricted together (not response-only restriction).

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

NOT DIRECTLY VERIFIED — the article is paywalled and I could not obtain the body text. What is
established indirectly, and how:

- **The master matrices and submatrices are printed, in a form from which cell frequencies are
  recoverable.** Evidence: Morgan (1974, *J. Math. Psychol.* 11, 107-123) applied a likelihood-ratio
  test of equality between transition matrices to "the data reported by Clarke (1957)". Such a test
  requires cell counts, not just proportions. Reported in Townsend & Landon (1982, *JMP* 25, 119-162,
  p. 122): "*Morgan (1974) applied his likelihood ratio test to the data reported by Clarke (1957)
  and Egan (1957) and found their data to depart significantly from the predictions of the CRR.*"
- **Data are pooled over subjects, not per-subject.** Townsend & Landon (1982, p. 122): "*Both
  Clarke (1957) and Egan (1957) presented their data pooled over subjects and reported that the CRR
  held with their data.*"
- **Entries are expressed as proportions in the predicted-vs-obtained comparison**, and Clarke's
  presentational device is a scatterplot against the 45-degree line. Hodge (1967, *Percept.
  Psychophys.* 2, 429-437, p. 430): "*The CRR states that the predicted and obtained proportions
  should fall exactly on the 450 [45-degree] line (Clarke, 1957).*"
- **Clarke's goodness criterion is an absolute difference of .10 between obtained and predicted
  proportions.** Hodge (1967, p. 431): "*the percentage of the absolute differences which exceed a
  difference of .10, a criterion value suggested by Clarke (1957)*".
- **No statistical test is offered.** Hodge (1967, p. 430): "*Although a satisfactory statistical
  test is not available (Clarke, 1957)*". This is the gap Morgan (1974) closed.

Table numbers are unknown. Establishing them is the single highest-value follow-up for this branch.

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://api.crossref.org/works/10.1121/1.1909023` — FETCHED 200. **Open.** Returns the complete
  author abstract (quoted above) plus affiliation and pagination. No body text, no tables.
- `https://web.archive.org/web/20141022183942/http://scitation.aip.org/content/asa/journal/jasa/29/6/10.1121/1.1909023`
  — FETCHED 200 (63 KB). **Wayback-only.** 2014 Scitation landing page. Contains the abstract and
  the strings "*The full text of this article is not currently available*", "*You have no
  subscription access to this content*", "*Buy: USD30.00*". No tables, no figure/table captions.
- `https://pubs.aip.org/asa/jasa/article/29/6/715/739456/Constant-Ratio-Rule-for-Confusion-Matrices-in`
  — FETCHED, **HTTP 403** (Cloudflare "Just a moment..."). Current publisher page; **paywalled**
  ($30 per-article purchase as of the 2014 snapshot).
- `https://api.unpaywall.org/v2/10.1121/1.1909023?email=...` — FETCHED 200. `is_oa: false`,
  `oa_status: "closed"`, zero OA locations.
- Also tried and failed: Semantic Scholar Graph API (`openAccessPdf.status: "CLOSED"`),
  OpenAlex (single closed location), Internet Archive Scholar, fatcat, CORE, archive.org
  full-text search, Google Books API, HathiTrust (403), ResearchGate (403), DTIC/search.gov
  (no hits), IUScholarWorks. No preprint, repository copy, or technical-report version of this
  specific JASA paper was located.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NEEDS LIBRARY ACCESS, then digitizing.** This is the highest-priority acquisition in the whole
auditory CRR branch: it is the origin paper, it is the only study Luce (1959) cited as evidence for
his axiom, and it is one of the two datasets Morgan (1974) showed to *violate* the axiom. Six pages
of 1957 JASA; obtain via institutional subscription, ILL/document delivery, or the ASA digital
library, then OCR/hand-key the 6x6 masters and 3x3 submatrices. Until then no zero-parameter
Gaussian-vs-Gumbel forecast can be scored on it.

## What the authors concluded about CRR, quoted verbatim where possible

Clarke's own abstract, verbatim and in full:

> "Three experiments are reported which give support to an empirical rule which may be used for
> predicting the entries in a closed confusion matrix for any subset of items drawn from a master
> set of items with a known confusion matrix. This rule, the constant-ratio rule, states that the
> ratio between any two entries in a row of a submatrix is equal to the ratio between the
> corresponding two entries in the master matrix. For this statement of the rule it is assumed that
> the only variables which differ systematically in obtaining the two matrices are the different
> sets of messages and the allowable responses. This is an empirical rule which was formulated after
> examination of three 6x6 master matrices for CV's (consonant-vowel syllables) and six 3x3
> submatrices. Two more experiments using monosyllables and digits were then conducted to test
> further the rule. Although no direct experimental evidence is reported, the use of the
> constant-ratio rule for predicting a master matrix given some of its possible submatrices is
> discussed."

So Clarke concluded FOR the CRR. Two important later reversals bear on this experiment:

- Clarke himself retreated for unidimensional stimuli. Hodge (1967, p. 429): "*In an experiment
  with simple auditory stimuli, e.g., tones varying in frequency or intensity, Clarke (1959) noted
  that the rule tended to fail. In Clarke's opinion, [footnote: F. R. Clarke, personal
  communication, 1960] the rule failed because the ordering inherent in single dimensional stimuli
  produces contextual constraints or biases which, by definition, are incompatible with the CRR.*"
- Morgan (1974) reversed the verdict on *this* dataset. Townsend & Landon (1982, p. 122): "*Morgan
  (1974) applied his likelihood ratio test to the data reported by Clarke (1957) and Egan (1957)
  and found their data to depart significantly from the predictions of the CRR.*" Morgan attributed
  the discrepancy to Clarke's pooling over subjects and to the absence of any significance test in
  1957.
