# Restriction: what happens to population shares when an alternative is withdrawn

Self-contained empirical programme behind `papers/thurstone_humans/paper.tex`. It asks one
question. Given population shares over a menu, and told that some alternatives are gone,
what are the shares over the survivors?

There are exactly two parameter-free answers. Proportional renormalization, which is the
Gumbel point of the independent additive random-utility class and which psychology has
called the **constant-ratio rule** since 1957. And re-running a Gaussian contest among the
survivors, which is Thurstone's Case V. Everything between them costs a fitted number. Both
calibrate from the same full-menu shares and neither sees a restricted-menu observation, so
they can be scored against each other out of sample.

## Layout

| Path | Contents |
|---|---|
| `*.py` | one script per population; each prints its own table and is runnable alone |
| `data/` | every input, committed, with a `SOURCE.md` per collection giving provenance, access route and caveats |
| `results/` | the output of each script, checked in so a claim can be traced to a run |
| `results/STATUS.md` | the running log: what is done, what failed, what is unverified |
| `notes/crr/` | a census of the constant-ratio-rule literature, one note per experiment, negatives included |

## Running anything

Requires numpy and scipy only. The shared calibration and win-probability routines live in
`../polysemy_pilot/exact_analyze.py`, which every script adds to `sys.path` as a sibling; the
scripts do not import the `winning` package.

    python tones.py
    python getty.py 200          # optional argument is Monte Carlo replicates

## Conventions that matter

**Every claim is scored out of sample against a null in which the axiom holds by
construction.** Contraction of a noisily estimated share vector lowers log loss whether or
not renormalization is wrong, so a positive gain against zero proves nothing. The null
regenerates the data from an exact Luce process whose worths are the observed shares and
reruns the identical pipeline. Where the null is stronger still it also resamples the
calibration row, charging the race for calibration noise.

**No losable data.** Inputs are committed here the day they are obtained, as column extracts
rather than raw archives, with the fetched URL recorded. Two collections in this project were
lost to a wiped scratch directory before that rule existed.

**Negative results are recorded, not discarded.** Four populations beat the race and are in
the paper as boundary conditions. Sources that cannot be scored get a note saying why, so the
next pass over the literature starts where this one stopped.

## Where the answer stands

The Gaussian contest wins where alternatives are distinct unordered items and loses where
they sit on a perceptual continuum, where a removed option is a near-substitute for a
survivor, or where removal changes how easily the survivors can be told apart. Wherever
shares are concentrated the two maps agree and the comparison carries no information.
`results/STATUS.md` has the current tally and every caveat.
