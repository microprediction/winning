"""exp39: our arm is overconfident against a competitor that is not.

REGISTERED, NOT YET RUN.

The referee question this answers. Glicko-2 propagates rating
uncertainty into every prediction -- `win_probabilities` passes each
player's RD into `gaussian_win_probabilities`, so a player it knows
little about gets a prediction pulled toward 0.5. Our arm does not. It
fits a MAP point estimate and prices every game with a fixed unit
noise, `race_probabilities(-mu, D=ones(2))`. Two estimators are being
compared while only ONE of them carries its own uncertainty.

That asymmetry is not neutral and it should be stated in whichever
direction it runs. My expectation is that it runs AGAINST us -- an
overconfident predictor is penalised by a proper scoring rule, so the
published margin of -0.0161 would be an understatement. But that is a
belief about a sign, not a measured fact, and the whole point of the
programme is to stop asserting those.

Note what is NOT a defence here: the validation-tuned ridge does
shrink mu toward zero, which reduces overconfidence in aggregate, and
one could argue calibration is therefore already handled. It is not
the same thing. Shrinking the MEAN and widening the PREDICTIVE are
different operations, and only the second represents "I do not know
much about this player."

ARMS, all sharing the split, the games and the tuned ridge of exp35:

  published     D = 1                       exactly as reported
  tempered      D = s, s tuned on validation  cheap proxy for the
                                            missing uncertainty
  laplace       D = 1 + var(mu_w - mu_b)    the principled version:
                                            per-parameter posterior
                                            variance from the diagonal
                                            of the MAP Hessian

The `laplace` arm is a diagonal approximation and is labelled as one.
The full Hessian is width x width, and width is 3 x players; at 2013's
639 players that is tractable and at 2024's 4,389 it is not, so the
diagonal is what generalises.

FAIRNESS. Tempering our arm is symmetric treatment, not a thumb on the
scale: Glicko-2's tau is ALREADY tuned on the same validation window
in every experiment in this directory. Arm `tempered` gives our side
the one free parameter their side has had all along. If that is judged
unfair, the honest alternative is to remove tau tuning from Glicko-2
too, and the comparison should be reported both ways.

PRE-REGISTERED PREDICTIONS:
  P1. `tempered` beats `published` -- i.e. s* > 1, our arm really was
      overconfident.
  P2. The margin over glicko2-per-TC WIDENS relative to -0.0161.
  P3. `laplace` lands between the two, closer to `tempered`: the
      diagonal captures most of the missing spread.
  FALSIFICATION: if s* is at or below 1 and the margin does not widen,
  then our arm was already well calibrated, P1-P3 are wrong, and the
  claim "the published margin is conservative because we carry no
  uncertainty" must be dropped from the README and never repeated. A
  null here is a perfectly good outcome and costs the headline
  nothing -- the margin stands either way; only the editorialising
  about its direction would have to go.

Run:  python research/chess/exp39_predictive_calibration.py [split_seed]
"""
raise SystemExit(__doc__)
