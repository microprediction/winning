# Issues to refile from Peter's account

Both were filed then closed by the wrong gh identity (teresa-crowd) on
2026-08-18: allocation#32 and winning#6. Text to refile:

## microprediction/allocation — "Replace approximate tail-dependent calibration with the winning factor engine"

The winning package now ships a much better algorithm than the
approximate calibration used here: exact all-share computation and
share calibration for factor-correlated races (any base noise, factor
rank as a parameter), in one O(QNL) shared-field pass with matrix-free
graph-Laplacian derivatives. See "Scalable Share Calibration for Factor
Multinomial Probit Models" (winning/papers/factor-probit-transform) and
the winning.factor / winning.probit APIs.

TODO: fix the tail-dep in allocation using this — replace the
approximate calibration with the much better approximation, e.g. the
tail-dependent Black-Litterman material.

## microprediction/winning — "Support allocation's tail-dependent calibration (Black-Litterman line)"

Counterpart of the allocation issue: track anything winning needs to
expose for that migration — e.g. tail-dependent Black-Litterman style
inputs, non-Gaussian bases (t, skew-normal already ship in Python/JS),
and covariance-supplied fitting via winning.probit.fit_factor_model.
