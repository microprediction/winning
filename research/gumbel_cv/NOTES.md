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
