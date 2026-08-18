## Citation

Dodds, P., Donkin, C., Brown, S. D., Heathcote, A., & Marley, A. A. J. (2011).
Stimulus-specific learning: disrupting the bow effect in absolute identification.
*Attention, Perception & Psychophysics*, 73(6), 1977-1986. DOI 10.3758/s13414-011-0156-0.
Authors and pagination confirmed via Crossref.

## Domain and stimuli

Absolute identification. Experiment 1 compares line length against tone loudness;
Experiment 2 manipulates presentation probability; Experiment 3 uses tones varying in
frequency and manipulates the order in which set sizes are practiced. Two 1-hour sessions
per participant, 20 blocks of 80 trials.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)

**Genuinely nested, and the labels are explicitly global — verified verbatim from the
paper.** The restricted set is the middle two stimuli of eight, and:

> "The first 5 blocks in the first session used only the middle two stimuli (N = 2), and all
> subsequent blocks used all stimuli (N = 8). When the participants were presented with only
> the middle two stimuli in the first 5 blocks in the first session, they responded to these
> with the numerals 4 and 5."

So master {1..8}, restricted {4,5}, same labels, **within subjects**. Design intent stated
earlier in the paper: "The stimulus set for the smaller set size was created from the middle
stimuli of the larger set size (e.g., the two stimuli for N = 2 were the same as the middle
two stimuli from N = 8)."

Experiment 3 counterbalances the order (N=2-first versus N=8-first), which is exactly the
control the Rouder `chunk` data mostly lacks.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)

**Nothing at cell level.** Trial counts are stated — "every participant received 200
presentations each of stimuli 4 and 5 when N = 2 and 150 presentations of each of the eight
stimuli when N = 8", totalling 350 presentations each of stimuli 4 and 5 — but the results
are reported as accuracy and RT by stimulus position, in figures. No confusion matrices, no
data availability statement, nothing on figshare, OSF or GitHub under any of the five
authors.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)

**Open**, publisher. Fetched, HTTP 200, application/pdf, 340,312 bytes:

    https://link.springer.com/content/pdf/10.3758/s13414-011-0156-0.pdf

Also mirrored on figshare (two records, listed by Unpaywall as repository copies).

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**Unusable as a scoring target** — manipulates set size with nested labels but publishes
only accuracy and RT, no cell-level shares. Recorded here because the design is exactly
right and it is the best modern template: if anyone runs a fresh experiment for this
project, Experiment 3 is the design to copy, and the authors may still hold the trial data.

## What the authors concluded, quoted verbatim where possible

On the fragility of the bow effect under within-subject set-size manipulation:

> "Two other phenomena (practice effects and improved performance for frequently-presented
> stimuli) have an important but less explored consequence for the bow effect: Standard
> within-subjects manipulations of set size could disrupt the bow effect. We found this
> disruption for stimulus types that support practice effects (line length and tone
> frequency), suggesting that the bow effect is more fragile than has been thought."

> "Experiment 1 and 2 indicated that the bow effect can be disrupted by design factors, such
> as within-subjects manipulations and stimulus presentation probabilities, at least when the
> stimuli are line lengths."

Directly relevant caution for the `chunk` data: practice on a restricted set transfers
stimulus-specifically to the same stimuli in the full set, so a full-set matrix collected
*after* subset training is not the same object as one collected before. Use the pretest, or
counterbalanced designs.
