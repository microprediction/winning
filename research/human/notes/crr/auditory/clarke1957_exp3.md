# Clarke (1957), Experiment III — spoken digits, master + subsets

## Citation

Clarke, F. R. (1957). Constant-Ratio Rule for Confusion Matrices in Speech Communication.
*Journal of the Acoustical Society of America*, **29**(6), 715-720. doi:10.1121/1.1909023
Hearing and Communication Laboratory, Department of Psychology, Indiana University.

Third of the three experiments in the founding CRR paper.

## Stimuli and master response set

Spoken **digits** in noise, closed-set identification. Clarke's abstract: "*Two more experiments
using monosyllables and digits were then conducted to test further the rule.*" Lee (1968, p. 217)
calls the three classes "*spoken syllables, words, and numbers*".

Master set size is not stated in the abstract. Most likely the ten digits 0-9 or the nine digits
1-9; not recoverable without the article. Note that the digit vocabulary was the standard
intelligibility material of the era, and that Clarke & Anderson (1957) used a **10-item** master set
with two 5-item subsets — plausibly the digits, which would make Clarke & Anderson essentially a
naive-subject replication of this experiment.

Cross-branch value: **the digit vocabulary is the one place in this literature where a modern,
freely available master confusion matrix already exists.** Morgan, Chambers & Morton (1973) print
four complete 9x9 digit matrices (recognition in noise, two voices; serial recall, two voices) —
see `morgan1973.md` in this directory. Those give a modern master; Clarke (1957) Exp III would give
the master-plus-subset structure over the same vocabulary.

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Nested** — digit subsets, with stimulus set and allowable response set restricted together
(Clarke's stated design assumption). Number and size of subsets not recoverable from the abstract.

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

NOT DIRECTLY VERIFIED (paywalled). Indirect evidence identical to Experiments I and II:

- Cell-level data adequate for a likelihood-ratio test on transition matrices are printed (Morgan
  1974 re-tested Clarke's data; via Townsend & Landon 1982, p. 122).
- Pooled over subjects (Townsend & Landon 1982, p. 122).
- Predicted vs obtained plotted as proportions against the 45-degree line, .10 absolute-difference
  criterion (Hodge 1967, pp. 430-431).
- No significance test (Hodge 1967, p. 430).

Table numbers unknown.

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

Same article as Experiments I and II:

- `https://api.crossref.org/works/10.1121/1.1909023` — FETCHED 200, **open**, abstract only.
- `https://web.archive.org/web/20141022183942/http://scitation.aip.org/content/asa/journal/jasa/29/6/10.1121/1.1909023`
  — FETCHED 200, **Wayback-only**, abstract + "Buy: USD30.00".
- `https://pubs.aip.org/asa/jasa/article/29/6/715/739456/Constant-Ratio-Rule-for-Confusion-Matrices-in`
  — FETCHED **403**. **Paywalled.**
- `https://api.unpaywall.org/v2/10.1121/1.1909023?email=...` — FETCHED 200, `closed`, no OA copy.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NEEDS LIBRARY ACCESS, then digitizing.** Highest cross-branch leverage of the three Clarke
experiments, because the digit vocabulary already has a modern open master matrix (Morgan, Chambers
& Morton 1973, four 9x9 tables, obtained and transcribed in this directory). Pairing Clarke's
1957 digit subsets with the 1973 masters would let a zero-parameter Gaussian-vs-Gumbel contest be
run on digits across a 16-year replication gap.

## What the authors concluded about CRR, quoted verbatim where possible

Clarke, verbatim (author's abstract):

> "Three experiments are reported which give support to an empirical rule which may be used for
> predicting the entries in a closed confusion matrix for any subset of items drawn from a master
> set of items with a known confusion matrix."

The Cambridge group later rejected the rule explicitly while working on digits. Morgan, Chambers &
Morton (1973, *Percept. Psychophys.* 14, 375-383, p. 380), verbatim:

> "Unless one subscribes to the somewhat dubious (see Cane, 1960; Morgan, 1973b) 'constant ratio
> rule' (CRR) (Clarke, 1957), also known as 'Luce's choice axiom' (Luce, 1959), then these factors
> will influence the predictions of the experiments and confound any comparisons one might wish to
> make between experiments."

and (p. 382), on what the CRR would get wrong if used to predict small vocabularies from a large
master:

> "Thus, prediction from the full alphabetic confusion matrix to small vocabularies using the CRR
> could overestimate the errors for low-confusion subsets such as C F J."

("Morgan, 1973b" in that reference list is "A statistical test of Luce's choice axiom, *Journal of
Mathematical Psychology*, in press" — published as Morgan (1974), *JMP* 11, 107-123, "On Luce's
choice axiom"; see `morgan1974.md`.)
