# Gumbel-twin control variate (2026-08-28)

softmax_probabilities is analytic, so the Gumbel race coupled through
common uniforms is a free control variate for frequency simulation of
any base. Measured (run_cv.py, n=20, M=20k, 30 reps, per-coordinate
regression beta):

| base              | median VRF | TV improvement |
|-------------------|-----------:|---------------:|
| GEV xi=0 (=Gumbel)|        inf | x13.5 (beta-hat noise only; beta=1 is exact) |
| GEV xi=0.05       |       11.7 | x3.6 |
| GEV xi=0.1        |        6.6 | x2.5 |
| GEV xi=0.2        |        4.0 | x2.0 |
| GEV xi=0.4        |        2.7 | x1.6 |
| normal (variance-matched) | 3.1 | x1.9 |

Reading: excellent for bases near Gumbel (Peter's conjecture), a
consistent ~2x TV for the normal race -- the same order as the old
indicator-CV finding from the general-Sigma program, so it does not
change the dense-Sigma story, but it is free (same uniforms, analytic
mean) wherever anyone insists on simulating. Candidate for a
CV-boosted frequency arm in winning.methods.

## Equal-compute accounting (Peter's objection, measured)

The twin's compute could buy plain draws instead. run_cv_equal_compute.py
gives both estimators the same wall clock (beta=1, coupled uniforms):

| base            | cost ratio | net TV improvement at equal time |
|-----------------|-----------:|---------------------------------:|
| GEV xi=0.05     |      x1.65 | x1.8 |
| GEV xi=0.2      |      x1.63 | x1.2 |
| normal (matched)|      x1.77 | x1.3 |

The Gumbel quantile is cheap but the coupled twin still nearly doubles
the loop (RNG + argmin dominate), so the equal-draw VRFs of 3-12
deflate to 1.2-1.8 net. Verdict: NOTED, not packaged as an arena arm --
a 20-80 percent improvement is real but does not justify a new
contestant when the lattice prices the same objects exactly; the recipe
stays here and in the softmax_probabilities docstring for anyone
simulating a near-Gumbel base anyway.
