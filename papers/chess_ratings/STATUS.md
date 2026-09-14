# Status

Note, no venue.

Numbers come from `research/chess`: exp41 for the tuned comparison
and its tables (`results/exp41_output.txt`), exp40 for the prequential
comparison, exp31 for opponent strength,
exp39 for the calibration check, exp36 for colour and exp37 for the
density sweep.

Both systems are tuned on the validation split, Glicko-2 on its rating
period and initial deviation and our arms on their level and offset
ridges. Glicko-2's `tau` is not tuned because it is inert over a single
month: sweeping it returns bit-identical held-out loss, so selecting it
would be a tie-break. `winning.ratings.tuning` reports that condition
on any sweep.

The colour and density figures come from runs against Glicko-2 at its
default rating period and are not recomputed. They will shrink by
roughly the headline's proportion, since they share its baseline. The
density claim at issue is flatness across inclusion thresholds, which
does not depend on the level.

Glicko-2's rating period selects the largest value offered, on every
arm and split, so the headline is an upper bound. The parameter
saturates rather than running away, so the residual is bounded near
0.0001.
