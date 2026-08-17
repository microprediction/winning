# Elliffe & Davison 2010 — four-alternative choice violates the constant-ratio rule

## Citation

Elliffe, D. & Davison, M. (2010). Four-alternative choice violates the constant-ratio rule. *Behavioural Processes*, 84(1), 381–389. doi:10.1016/j.beproc.2009.11.009. PMID 19941942. (Published online 24 Nov 2009; often cited as 2009.)

## Domain and stimuli

**Animal subjects on concurrent schedules — six pigeons on a four-key concurrent variable-interval schedule.** Not a human confusion-matrix dataset. The alternatives are four physical response keys, distinguished only by location; there is no stimulus-similarity manipulation. A 27:9:3:1 distribution of reinforcers across the four keys was **reassigned to new key locations every 10 reinforcers**, i.e. this is a rapidly changing ("frequently changing") procedure with many reinforcer-ratio changes within a session, not a steady-state one. Reinforcer rate, not perceptual similarity, is the independent variable.

## Master and restricted response sets

**There is no restricted menu. All four keys were available and lit in every component of every session.** This is the single most important fact about this paper for the project.

What the authors call a test of the constant-ratio rule is a *within-full-menu ratio-invariance* test, not a menu-deletion test. Because the four rates form a 27:9:3:1 series, several pairs of alternatives share the same reinforcer *ratio* at different absolute rates:

- two 9:1 pairs — B27:B3 and B9:B1;
- three 3:1 pairs — B27:B9, B9:B3, B3:B1.

CRR (in the "relative homogeneity + relative independence" reading the Auckland group uses, after Prelec 1984) requires the log response ratio to be the same within each of those families. Elliffe & Davison found it was not: preference for the richer member of a pair depended on the absolute level of the two rates as well as their ratio, with greater preference in the higher-absolute-rate pair (B27:B3 > B9:B1). The "renormalisation" being tested is therefore the *marginal pairwise response ratio computed inside the always-four-alternative menu*, not a prediction of what happens when alternatives are deleted.

## What numbers are printed or deposited

Not obtained. I could not get past the abstract for this paper, so I cannot describe the tables at first hand. What is establishable:

- The paper is behind Elsevier's paywall.
- Unpaywall reports a green OA copy (cc-by, submittedVersion) at the University of Auckland repository, handle 2292/13165, but `url_for_pdf` is **null**, and the DSpace item has only a `TEXT` bundle (a single extracted-text bitstream, `Four-alternative choice violates the constant-ratio rule.pdf.txt`, 50,173 bytes) with **no `ORIGINAL` bundle**. Requesting that bitstream's content returns HTTP 401 "Authentication is required". So the deposit is metadata-plus-restricted-text only; the OA claim is effectively hollow.
- Indirect evidence about the analyses, from Bensemann's PhD thesis (Chapter 3 replicates this experiment with the same six pigeons): the CRR analysis in this line of work is conducted as generalized-matching regressions of pairwise log response ratios on pairwise log obtained reinforcer ratios, plus comparisons of log response ratios among equal-ratio pairs — i.e. **slopes and log-ratio summaries, not printed four-way response-share matrices**. Bensemann's own replication prints no four-alternative response-count matrix either, only figures plus a table of median collection times.

Not obtained: whether Elliffe & Davison print per-bird response counts on the four keys.

## Access with a fetched url

- Abstract and full bibliographic record, fetched: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=19941942&retmode=xml` (returns the complete PubMed abstract, Behav Processes 84(1):381-9).
- Semantic Scholar record, fetched: `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.beproc.2009.11.009?fields=title,abstract,year,authors,externalIds,openAccessPdf,venue` (reports GREEN/CC-BY at hdl.handle.net/2292/13165).
- Repository item metadata, fetched: `https://researchspace.auckland.ac.nz/server/api/pid/find?id=hdl:2292/13165` (title, authors, abstract, DOI).
- Repository bundle/bitstream listing, fetched: `https://researchspace.auckland.ac.nz/server/api/core/items/dc0da056-0ab2-4b32-98bd-5ea508e880b5/bundles` and `https://researchspace.auckland.ac.nz/server/api/core/bundles/f15ffb47-f5f8-463d-8153-2d888f1f93fc/bitstreams`.
- Bitstream content, fetched and **failed**: `https://researchspace.auckland.ac.nz/server/api/core/bitstreams/68c98539-049a-4a68-8be1-5aa59383b635/content` → `{"status":401,"error":"Unauthorized"}`.
- Unpaywall, fetched: `https://api.unpaywall.org/v2/10.1016/j.beproc.2009.11.009?email=peter.cotton@gsmc.ai` (`url_for_pdf: null`).
- Also tried and failed: `https://hdl.handle.net/2292/13165` (302 to researchspace, JS-rendered landing page with no bitstream links in the HTML). A web search for a public PDF was not possible — the session's WebSearch budget was exhausted.

## Usability verdict

**NOT-A-RESTRICTED-SET-STUDY.** Despite the title being the most promising in the whole candidate list, this paper never removes an alternative: all four keys are present throughout. Its "constant-ratio rule" is the operant/matching-law reading (relative preference within the full menu should depend only on that pair's reinforcer ratio), not Luce's deletion-and-renormalise reading. It therefore cannot supply a full-menu-calibrated prediction scored on a restricted menu. Secondarily, the numbers are not obtainable anyway: paywalled with an inaccessible repository deposit, so it would be **CRR-TEST-BUT-NUMBERS-NOT-PRINTED** even on the authors' own version of the test.

Worth citing in the paper as evidence that ratio invariance fails in a non-human, non-perceptual, reinforcement-driven setting — but not usable as a dataset.

## Conclusion about CRR quoted verbatim

From the abstract (p. 381): "However, preference between a pair of keys depended not only on the relative reinforcer rates on those keys, but also on the absolute levels of those rates. This contradicts the constant-ratio rule that underpins the matching approach to choice, but is predicted by a contingency-discriminability model that assumes that organisms may occasionally misattribute reinforcers to a response that did not produce them."

The body text is NOT OBTAINED. Tried: doi.org/Elsevier (paywall), the Auckland ResearchSpace handle and its DSpace REST bitstream (401 Unauthorized, no ORIGINAL bundle), Unpaywall (`url_for_pdf: null`), Semantic Scholar openAccessPdf (points at the same dead handle), CORE title search (no matching record returned), PubMed Central (Behavioural Processes is not in PMC). WebSearch budget for the session was exhausted before a web search for a mirrored PDF could be run.

A second-hand statement of the same finding, quoted verbatim from Bensemann's PhD thesis (p. 62), which used the same six pigeons: "The results of both Davison et al. and Elliffe and Davison found violations of the constant-ratio rule." And (p. 71): "Davison et al. (2007) and Elliffe and Davison (2010) reported that subjects showed greater preference for the richer alternative in the B27:B3 pair than for the richer alternative in the B9:B1 pair."
