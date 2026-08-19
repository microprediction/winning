# The constant-ratio-rule seam

The constant-ratio rule (CRR) is proportional renormalization under an older name:
removing response alternatives should leave the odds between the survivors unchanged.
Psychology has tested that claim on real restricted response sets since 1957, which is
where the data this project needs already exists. What that literature did *not* do is
score renormalization against a parameter-free Gaussian race out of sample. Several of
those papers report CRR as approximately right with a systematic residual they treated
as nuisance; if a Gaussian race predicts the residual, the result stops being a new
violation and becomes the missing correction to a sixty-year-old empirical law.

## What a usable source looks like

Two things must be printed or deposited, over the same stimuli and ideally the same
subjects:

1. a **master confusion matrix** over the full response set — the calibration input;
2. **one or more restricted response sets**, with their own matrices — the held-out target.

A paper that varies set size but publishes only percent correct is not usable, because
neither map can be scored without cell-level shares. Note it in one line and move on so
nobody searches it twice.

Nested restrictions are worth more than relabellings. A relabelled set changes what the
task *is*, so a failure there is not evidence against either map.

## Directory layout

One subdirectory per search branch so concurrent agents never collide, one file per
experiment named `firstauthor_year.md`:

| Directory | Branch |
|---|---|
| `auditory/` | Clarke 1957, Pollack & Decker, Hodge & Pollack, Holloway, Morgan |
| `visual/` | Townsend & Landon 1982, Hodge 1967, Getty 1979, Lupker, Keren & Baggen |
| `memory/` | Wills et al. 2000 categorization, short-term memory, tactile, Eriksen & Hake |
| `absolute_id/` | absolute identification with varying response sets |
| `odor/` | odor identification, free naming versus cued lists |
| `forward/` | citation sweep, plus `PRIOR_ART.md` |

## Headings every note carries

    ## Citation
    ## Domain and stimuli
    ## Master response set and restricted response sets (nested, overlapping, or relabelling)
    ## What numbers are printed or deposited
    ## Access (a fetched url; open, paywalled, or Wayback-only)
    ## Usability verdict
    ## What the authors concluded, quoted verbatim where possible

Negative results get a file too. The point of the notes is that the next pass over this
literature starts where this one stopped.

## Standing caution from the datasets already run

Two failure modes recur and both are visible in advance from the design, not the fit.
**Near-substitutes** break every contraction map, because removal can reverse an
ordering rather than shrink it — Scottish verdicts and the nested tone matrices both do
this. **Quality-changing removal** breaks the premise that the surviving alternatives
are unchanged, as with exam distractors. When reading a candidate source, check the
stimulus set for both before recording it as usable.
