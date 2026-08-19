# Duffy & Smith (2025) — line-length choice with induced values

Duffy, S., & Smith, J. (2025). An economist and a psychologist form a line: what can
imperfect perception of length tell us about stochastic choice? *Theory and Decision*,
99(3), 701-734. doi 10.1007/s11238-025-10040-4. Open Access, CC-BY.

Trial-level data from the authors' OSF node `f7gu4`, the file they cite in the paper as
holding "the dataset and the results of our simulations". Downloaded 2026-08-19:

    https://osf.io/download/nsya2/   ->   NoCLUALData-OSF.csv   3,655,461 bytes
                                          11,200 rows, 112 subjects x 100 trials, 74 columns

## Why this dataset is a prediction rather than a threat

Subjects see two to six grey lines and are paid for selecting the longest. Value is induced
and one-dimensional by construction, which the authors make the selling point of the design.
Abstract, verbatim: "objects are valued according to only a single attribute with a continuous
measure and we can observe whether the choice was optimal."

So the alternatives lie on a single physical continuum. That is where this paper's boundary
rule says linear renormalization is the better map, and it is what the run below finds. It
joins the tone matrices, the Rouder line-length data and the Getty condition whose survivors
are mutual confusions as the fourth continuum collection pointing the same way.

The authors reject neither the ratio rule nor IIA across twenty-four specifications, which is
consistent, and their own test has little power against Case V. But the direction here does
not rest on their non-rejection; it rests on the run.

## Not the paper's usual protocol

Lengths are redrawn each trial, so there are no fixed alternatives carrying stable shares to
calibrate from and no share vector to invert. The parameter-free protocol used everywhere else
in this project cannot be applied. `duffy_smith.py` runs the parametric version of the same
question instead: fit one scale parameter per model on menus of one size by maximum
likelihood, hold it fixed, and score menus of another size by log likelihood per observation.
One free parameter each, so the comparison is fair between the two models but is a fitted
comparison rather than a parameter-free one.

## Result

`results/duffy_smith.txt`. 11,055 valid trials.

    calibrate  predict    beta(px)  sigma(px)   advantage to Luce      t
    n=2        n=3..6       7.94      10.34        +0.0059          3.93
    n=6        n=2          7.78      11.46        +0.0037          2.28
    n=5,6      n=2,3        7.89      11.63        +0.0051          4.54

The middle row is the restriction direction proper: calibrate on the full menu, predict the
reduced one. Linear renormalization wins all three.

## Difference from the first pass

An earlier exploratory run, recorded in `notes/crr/forward/duffy_smith_2025.md`, reported
+0.0026, +0.0028 (t = 1.86) and +0.0046 on 10,989 trials, with beta parameterized as a rate
(0.126 to 0.133 per pixel) rather than as a scale in pixels; 1/0.131 = 7.6 px against the 7.8
px here, so the fits agree. The 66-trial difference is a different validity filter. Direction,
order of magnitude and the ranking of the three splits all replicate; this file supersedes the
note's figures, since it is the one with committed code behind it.
