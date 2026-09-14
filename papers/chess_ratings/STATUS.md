# Status

Note, no venue. Numbers from `research/chess`, principally exp35 for
the arena and chess fits, exp41 for the tuned comparison
(`results/exp41_output.txt`), exp31 for opponent strength, exp37 for
the density sweep and exp39 for the calibration check.

## Correction, 2026-09-13

The first version of this note, committed the same day, reported a
headline of 0.0137 to 0.0182 nats and an estimator column of −0.0199,
and argued that stratification helps a weak base estimator and not a
strong one. Those figures came from a comparison in which Glicko-2 was
under-tuned. Its `tau` was swept and the selected value cited as
evidence of symmetric tuning, but `tau` is inert over a single month:
the sweep returned bit-identical loss and the selection was a tie-break
on the first grid entry. Its rating period and initial deviation, which
do move its loss, were never swept, and our own arms had the level
ridge pinned.

With both sides tuned on the same validation split:

| row | first version | corrected | change |
|---|---|---|---|
| headline | −0.0160 | −0.0103 | 36% smaller |
| estimator | −0.0199 | −0.0080 | 60% smaller |
| structure | −0.0035 | −0.0033 | unchanged |
| stratification's value to Glicko-2 | +0.0074 | −0.0011 | sign flip |

The result survives. The factor form beats Glicko-2 per time control on
all three splits with intervals excluding zero, and the factor
structure, which is the note's subject, is unchanged. The stratification
section is rewritten, because the contrast it argued from was an
artifact of the default rating period.

The colour and density figures are not recomputed and say so in place.

The guard that catches this class of defect ships as
`winning.ratings.tuning`.
