# Research ideas

## Gumbel-augmented control variate for high-rank factor integration
*(Peter, 2026-08-18, noted verbatim: "running a race and an augmented
race that makes it more like Gumbel and using logit as a control
variate (for higher rank problems)")*

The regime: k > 4, where product Gauss-Hermite explodes and the factor
integral goes scrambled-Sobol/MC. Each sampled factor draw f still
costs a full O(NL) lattice pass for the conditional probit shares.

The idea: pair the true race with an augmented race pushed toward
Gumbel, whose conditional shares are (near) closed-form softmax of the
conditional locations - O(Nk) per draw, no lattice. Estimate

    p = E_f[ probit(f) ]
      = E_f[ probit(f) - softmax_c(f) ] + E_f[ softmax_c(f) ]

with the first expectation over the expensive lattice draws (low
variance: the two conditional races co-move strongly across f, since
the factor part of the location is common) and the second driven to
negligible error on a vastly larger draw set at closed-form cost.
Regression-optimal beta instead of beta = 1 for free. The
"augmentation" dial: the tempered race machinery just shipped
(temperature = tau convolves the base toward Gumbel), so the control
can be moved along the probit-to-Gumbel bridge to maximize correlation
- an interpolated control variate, not just the softmax endpoint.

Connections in the tree:
- winning.factor.races temperature= (the Gumbel bridge, verified);
- qmc_nodes (the k > 4 path this would accelerate);
- research/experiments/exp34 (measured softmax-vs-probit bias: the
  control's bias is exactly what the difference term corrects);
- the softmax-thurstone notes in kinetics (E[softmin] identity).

Validation plan when picked up: k in {6, 8, 10}, N in {200, 1000};
variance of the CV estimator vs plain Sobol at matched lattice-pass
counts; report effective speedup and the correlation achieved; check
the optimal temperature of the control is interior (it may be).

## Thurstone demos migration (Peter, 2026-08-18)
Transfer the thurstone examples (laplacian_newton_demo,
diffeomorphism_demo, dynamic_calibration_demo, kalman_tracker_demo,
global_calibration family, ...) into winning/research/demos - but
CURATED, not dumped ("it could clutter"): one demo per concept,
self-contained, rewritten against winning.thurstone / winning.factor
imports, indexed in the demos README. Anything redundant with an
existing demo or experiment stays in thurstone's git history.
