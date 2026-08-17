## Citation

Lacouture, Y., & Marley, A. A. J. (2004). Choice and response time processes in the
identification and categorization of unidimensional stimuli. *Perception & Psychophysics*,
66(7), 1206-1226. DOI 10.3758/BF03196847.

## Domain and stimuli

Visual unidimensional stimuli, 10 levels. Experiment 1 crosses task (identification
versus categorization) with response mode (manual versus vocal) in a 2x2 within-subject
design. Experiment 2 collects full RT distributions. A leaky competing accumulator model is
fit.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)

**A coarsening, not a nesting — and deliberately not contiguous.** Identification uses
10 stimuli with 10 responses, keys 1-10 or spoken numerals. Categorization uses the same 10
stimuli but two responses, labelled A and B, with "in the categorization task only the two
extreme keys on the response panel" used.

The A/B labels are disjoint from {1..10}, so nothing is nested. Worse for our purposes, the
mapping is interleaved rather than a contiguous split — "adjacent stimuli were associated"
with different categories — so category A is not an interval of the stimulus dimension. That
makes it a partition chosen to be maximally unlike a restriction. A failure here would say
nothing about either map, exactly as the branch README cautions about relabellings.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)

**One genuine 10x10 confusion matrix is printed.** Table 5, "Confusion Matrix for the
Data and Simulation": the top half is the observed 10x10 as percentages to one decimal
(rows: 89.6 10.4 0 0 ...; 4.1 89.3 6.6 0 ...), the bottom half is the model simulation. It
appears to be a single participant or pooled block from Experiment 2, with roughly 400 trials
per stimulus.

Tables 2 and 3 are condition-level means of RT and probability correct (e.g. vocal
identification 1,308 ms / PC .72; manual categorization 1,267 ms / PC .67). Per-stimulus
detail for the categorization conditions is in Figures 2 and 3 only, not tabulated.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)

**Open**, publisher, Springer legacy Psychonomic archive. Fetched, HTTP 200,
application/pdf, 444,930 bytes, 21 pages:

    https://link.springer.com/content/pdf/10.3758/BF03196847.pdf

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**Unusable for the restricted-response-set test** — the two-response condition is an
interleaved coarsening with new labels, not a restriction over shared alternatives.

Minor positive: Table 5's 10x10 is a clean, citable master identification matrix for
unidimensional visual stimuli, if a master is ever wanted without a matching restriction. It
cannot serve as the calibration input for a held-out restricted matrix, because no restricted
matrix over the same labels is printed.

## What the authors concluded, quoted verbatim where possible

On the relation between the two tasks:

> "A revised model, with the independent accumulator decision process replaced by a leaky
> competing accumulator decision process, fits the probabilities of correct responses and the
> full distributions of RTs in unidimensional absolute identification. The revised model is
> also applied successfully to a particular class of unidimensional categorization tasks."

On why they did not fit the matrix itself — relevant, because it is the same reason the
earlier literature stopped at summary statistics:

> "remember that the parameters of the simulated process were adjusted to simultaneously fit
> the average PC and the mean RT associated with each stimulus, not the whole confusion
> matrix; we proceeded in this way because of the relatively small proportions of errors."
