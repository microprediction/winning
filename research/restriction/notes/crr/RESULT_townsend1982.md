# Result: Townsend & Landon (1982), the origin of the constant-ratio-rule tests

Run with `research/restriction/townsend_rows.py` on the digitized Tables 1-4. Fold this into
`research/restriction/results/STATUS.md` when the concurrent authoring pass is finished; it is
kept separate only to avoid a write clash.

## Why this dataset is unusually clean

Calibration and target come from the **same subject**, in different blocks, so there is no
population-heterogeneity confound of the kind every ranking-derived dataset carries. The
restriction is observed, not imputed. Responses are spoken letter names, so no rank
recoding could smuggle information across. Each row has exactly 240 trials.

Each row of a confusion matrix is its own restriction problem: the master row is a
distribution over five responses, the subset row a distribution over survivors. 38 usable
restricted rows, 9,120 held-out trials.

## Pooled

| | log loss |
|---|---|
| renormalization (the CRR itself) | 0.9454 |
| Gaussian race | 0.9412 |

gain **+0.0042** [+0.0022, +0.0064] row bootstrap
fitted-Luce null median −0.0011, excess over null **+0.0053**, p = 0.002 (400 reps)

Dropping the two rows containing cells the published table prints inconsistently makes no
material difference: with the arithmetically forced repairs applied, 40 rows and 9,600
trials give +0.0040 [+0.0020, +0.0062], excess +0.0052, p = 0.002.

## By subset, and this is the informative part

| subset | removed | gain | excess over null | p |
|---|---|---|---|---|
| {A,E,F,H} | X, the dissimilar letter | +0.0006 [−0.0002, +0.0015] | +0.0005 | 0.129 |
| {A,E,X} | F,H | +0.0040 [−0.0003, +0.0079] | +0.0059 | 0.010 |
| {F,H,X} | A,E | +0.0093 [+0.0051, +0.0141] | +0.0109 | 0.005 |

Removing the odd letter out does essentially nothing. Removing a *near-substitute pair* is
where the Gaussian race earns its keep. That is the opposite of the boundary condition the
other near-substitute datasets show, and the difference is which side of the removal the
substitutes sit on: here the near-substitutes are removed, in the tone matrices and the
Scottish verdicts they survive.

## The residual is only partly explained, and the note matters

Townsend & Landon diagnosed the failure as mass concentrating onto the surviving
near-substitute instead of spreading proportionally, and attributed it to Debreu. Case V
has no correlation parameters, so it cannot represent similarity. What it does instead is
contract. Take subject 1, stimulus A, subset {A,E,X}, observed / CRR / race:

    A  0.621 / 0.672 / 0.635      race corrects downward, right direction
    E  0.217 / 0.129 / 0.149      race corrects upward, right direction, far too little
    X  0.163 / 0.199 / 0.217      race corrects upward, WRONG direction

So the race gets the sign right on two cells of three and wins on net because the dominant
terms improve. It does not capture the concentration itself: E should have risen to 0.217
and X should have fallen. A similarity-structured residual survives, exactly as the
authors said. The honest claim is that a parameter-free Gaussian recovers part of the
sixty-year-old residual, not that it explains it.

## Prior art to acknowledge

Lee (1970) proposed this comparison analytically -- detection theory and the CRR diverge
"particularly for the univariate distributions, and this corresponds to empirical findings
that the CRR holds better for multidimensional than for unidimensional stimuli," with the
gap usable "in diagnosis of the basis of empirical confusion matrices." The contribution
here is the empirical follow-through and the out-of-sample scoring, not the idea.
