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

## Thurstone demos migration (status: undecided, 2026-08-18)
Peter mused about transferring the thurstone examples into
winning/research/demos, then pulled back ("maybe leave some in thurstone
for now... I don't know"). Current default: LEAVE THEM IN THURSTONE.
They keep working there against the shim, nothing is lost, and
winning/research/demos stays uncluttered. If migration ever happens the
agreed principle is curation, not dumping: one demo per concept,
rewritten against winning imports, filed under demos/siam2021/.

## VIX-first joint SPX-VIX calibration (noted 2026-09-03, flagged
## "may not be relevant" by Peter)
Zaugg & Grzelak, arXiv:2608.01479 [U beyond abstract]: calibrate
explicit VIX dynamics first, then derive the latent SPX volatility
as the process consistent with the calibrated VIX through its
rolling-window definition -- "an interpretable and tractable
decomposition of the joint problem" against black-box global
optimization. The structural rhyme with this repo, if any: invert
the observable (VIX prices / choice shares) to pin the latent
(volatility process / abilities), decomposing a joint calibration
into a tractable stage-wise one. No race/order-statistics content;
park unless a stochastic-volatility application of the engine ever
materializes (implied win probabilities from vol surfaces?).
