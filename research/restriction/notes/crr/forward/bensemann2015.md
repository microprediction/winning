# Bensemann, Lobb, Podlesnik & Elliffe 2015 — steady-state four-alternative choice obeys the constant-ratio rule

## Citation

Bensemann, J., Lobb, B., Podlesnik, C. A. & Elliffe, D. (2015). Steady-state choice between four alternatives obeys the constant-ratio rule. *Journal of the Experimental Analysis of Behavior*, 104(1), 7–19. doi:10.1002/jeab.157. PMID 25989016. (Note: the candidate list gave the authors as Bensemann & Lobb only; PubMed lists four authors, including Elliffe — the co-author of the 2010 paper this one contradicts.)

The same experiment is Chapter 3 ("Experiment 2") of: Bensemann, J. (2015). *The Properties of Reinforcement: Strengthening Versus Signalling*. PhD thesis, University of Auckland, handle 2292/27320 — which is fully open access and is the source used here.

## Domain and stimuli

**Animal subjects on concurrent schedules — the same six pigeons (numbered 81–86) that served in Elliffe & Davison (2010), on a four-alternative concurrent VI schedule.** Not a human confusion-matrix dataset. Alternatives are four keys distinguished by location only. Base schedule exponential VI 20 s (p = .05 of arranging food per second); reinforcer allocation probabilities per condition are printed in thesis Table 3.1. The first eight conditions are the eight permutations of the 27:9:3:1 probability set (.675/.225/.075/.025), equivalent to VI 29.63 s, VI 88.89 s, VI 266.67 s, VI 800 s; the ninth condition is 1:1:1:1 (four VI 80 s). Dependent scheduling; changeover ratio 1 (a one-response switching penalty).

## Master and restricted response sets

**There is no restricted menu. All four alternatives were lit and available in every session of every condition.** As with Elliffe & Davison (2010), the "constant-ratio rule" being tested is ratio invariance *within* the four-alternative menu, not deletion-and-renormalisation.

Design difference from Elliffe & Davison (2010) — this is the whole point of the paper:

| | Elliffe & Davison (2010) | Bensemann et al. (2015) |
|---|---|---|
| Subjects | 6 pigeons | the *same* 6 pigeons |
| Alternatives available | 4 (always) | 4 (always) |
| Reinforcer ratio | 27:9:3:1 | 27:9:3:1 (8 permutations) + 1:1:1:1 |
| Schedule-to-key assignment | **changed every 10 reinforcers** (frequently changing, many changes per session) | **held constant for 50 consecutive sessions per condition** (steady state) |
| Result | CRR violated | CRR obeyed |

Data reduction in both papers: responses are relabelled by rank of the reinforcer rate at that key (B27, B9, B3, B1) and pooled across permutations, so each reinforcer rate visits each physical key twice and location bias cancels.

The CRR test: compare log response ratios among pairs sharing a reinforcer ratio but differing in absolute rate — the two 9:1 pairs (B27:B3 vs B9:B1) and the three 3:1 pairs (B27:B9, B9:B3, B3:B1).

## What numbers are printed or deposited

Thin. From the thesis chapter (which is the same analysis as the JEAB paper):

- **Table 3.1 (p. 66):** reinforcer-allocation probability for each of the four keys in each of the nine conditions. Design numbers, not data.
- **Table 3.2 (p. 95):** median reinforcer collection times per subject for Alt 27 / Alt 9 / Alt 3 / Alt 1 in Conditions 1–8. E.g. bird 81: 3.39, 6.28, 13.10, 43.00 s; bird 82: 1.29, 4.01, 9.44, 22.48 s. (Latencies, not choice shares.)
- **Text (p. 70):** generalized-matching sensitivity (slope) values "ranged from 0.61 to 0.73 with an average of 0.69; that is, all subjects showed undermatching." Per-bird slopes are not tabulated in the thesis text I extracted.
- **Statistical tests (p. 71):** Wilcoxon matched-pairs signed-ranks test on the two 9:1 pairs, p = .345; Friedman ANOVA by ranks on the three 3:1 pairs, p = .223. "Only 2 of 6 subjects clearly showed more preference for the richer alternative in the B27:B3 pair."
- Everything else is figures: Figure 3.1 (per-subject response allocation across sessions 1–50), Figure 3.2 (sessional deviations from the last-5-session mean), Figures 3.3–3.8 (preference pulses), plus switching bar graphs.

**No four-alternative response-count or response-share matrix is printed**, per subject or in aggregate. The overall B27/B9/B3/B1 proportions exist only as figure points.

## Access with a fetched url

- Abstract and full bibliographic record, fetched: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=25989016&retmode=xml`.
- Semantic Scholar record, fetched: `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1002/jeab.157?fields=title,abstract,year,authors,externalIds,openAccessPdf,venue` — openAccessPdf status CLOSED, abstract elided by publisher.
- **Full open-access text of the same experiment**, fetched and parsed: the PhD thesis PDF at `https://researchspace.auckland.ac.nz/server/api/core/bitstreams/12433dbd-62a0-4d29-8124-a711c75c0aeb/content` (910,766 bytes, text layer extracts cleanly; Chapter 3 = this experiment). Located via `https://researchspace.auckland.ac.nz/server/api/discover/search/objects?query=Bensemann+constant-ratio` and `https://researchspace.auckland.ac.nz/server/api/pid/find?id=hdl:2292/27320` (both fetched).
- The JEAB version itself (Wiley) was not obtained; it is paywalled and not in PMC (JEAB is only open on PMC up to ~2013).

## Usability verdict

**NOT-A-RESTRICTED-SET-STUDY.** No alternative is ever removed; the paper tests ratio invariance inside a fixed four-alternative menu. It cannot supply a full-menu-calibrated prediction scored on a restricted menu. Even for the authors' own version of the test the numbers are figure-only, so it would additionally be **CRR-TEST-BUT-NUMBERS-NOT-PRINTED**.

Its real value to the project is dialectical: it shows that the operant literature's one prominent "CRR violated" result (Elliffe & Davison 2010) is *procedure-dependent* — the same six birds obey CRR once the environment is held steady. Any claim that the operant literature establishes CRR failure should be stated with that qualification.

## Conclusion about CRR quoted verbatim

From the JEAB abstract (p. 7): "We found that subjects' responding was consistent with the constant-ratio rule in the steady-state procedure. Additionally, local analyses showed that preference after reinforcement was towards the alternative that was likely to produce the next reinforcer, instead of being towards the just-reinforced alternative as in frequently changing procedures. This suggests that the effect of a reinforcer on preference is fundamentally different in rapidly changing and steady-state environments."

From the thesis, Results §3.3.2 (p. 71): "A Wilcoxon matched-pairs signed-ranks test confirmed that there was no systematic reinforcer-rate effect in the two 9:1 ratio pairs (p = .345). An equivalent analysis was conducted on the three 3:1 ratio pairs. A Friedman ANOVA by ranks showed that there was also no systematic pattern among the log response ratios obtained from responses to the B27:B9, B9:B3, and B3:B1 pairs (p = .223). Overall, the results of the present experiment were consistent with the constant-ratio rule."

From the thesis Discussion (p. 101), stating the disagreement explicitly: "The results of the present experiment were consistent with the constant-ratio rule and, therefore, at odds with the results of Elliffe and Davison (2010) and Davison et al. (2007). Their results violated the constant-ratio rule due to an overall reinforcer rate effect that was absent in the present experiment."
