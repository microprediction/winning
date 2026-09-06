\name{abilities_from_topk}
\alias{abilities_from_topk}
\alias{loc_scale_from_topk_pair}
\alias{loc_scale_from_win_and_second}
\alias{abilities_from_rank_marginal}
\title{Calibrate to place curves: locations at any depth, or joint
(location, scale) from two curves}
\usage{
abilities_from_topk(q, k, V = NULL, D = NULL, base = "normal",
  points = 513, qa = 15, n_iter = 80, tol = 1e-08,
  target_floor = NULL, return_info = FALSE)

loc_scale_from_topk_pair(q1, k1, q2, k2, D0 = NULL, base = "normal",
  points = 513, n_iter = 60, tol = 1e-08, ridge = 0, mu0 = NULL,
  return_info = FALSE)

loc_scale_from_win_and_second(p_win, p_second, D0 = NULL,
  base = "normal", points = 513, n_iter = 60, tol = 1e-08,
  ridge = 0, mu0 = NULL, return_info = FALSE)

abilities_from_rank_marginal(p, r, mu0 = NULL, D = NULL,
  base = "normal", points = 513, n_iter = 60, tol = 1e-08,
  return_info = FALSE)
}
\arguments{
\item{q, q1, q2}{target top-k membership curves; entries in (0, 1),
renormalized to their k slots (zeros stop: no finite inverse)}
\item{k, k1, k2}{membership depths; the pair solver needs k1 != k2}
\item{p_win, p_second}{win and EXACTLY-second marginals, each
renormalized to unit mass; converted internally by
P(2nd) + P(win) = P(top-2)}
\item{p, r}{one exact-rank marginal and its ONE-BASED rank}
\item{V}{optional factor loadings (rank <= 2) for
\code{abilities_from_topk}; the pair solvers refuse fixed loadings
(two curves are then one equation short -- see Details)}
\item{D, D0}{variances: known and fixed for the location solvers,
initial guess for the pair solvers}
\item{base}{noise family, as in \code{\link{top_k_probabilities}}}
\item{points, qa}{quadrature controls}
\item{n_iter, tol}{iteration cap and logit-residual tolerance}
\item{target_floor}{opt-in floor for zero entries (result is then a
one-sided bound on the floored contrasts)}
\item{ridge}{weight of the log-sigma penalty added to the SQUARED
objective (a reference scaling the ridge residuals by w corresponds
to ridge = w^2); a deliberate prior toward common scale for noisy
boards, off by default}
\item{mu0}{initial locations; for \code{abilities_from_rank_marginal}
it selects the branch of a deliberately two-branched problem}
\item{return_info}{also return convergence diagnostics instead of
warning on non-convergence}
}
\value{
Mean-zero locations (the location solvers), or list(mu, sd) with
mean-zero mu and geometric-mean-one sd (the pair solvers). With
\code{return_info = TRUE}, convergence flag, achieved residual and
iteration count are attached.
}
\description{
The inverse problems of \code{\link{top_k_probabilities}}.
\code{abilities_from_topk} inverts one cumulative membership curve by
a damped own-slope iteration in logit space (well-posed: the Jacobian
is negative definite on the mean-zero quotient).
\code{loc_scale_from_topk_pair} identifies per-runner (location,
scale) jointly from two curves by Levenberg Newton on the exact
stacked Jacobians -- square on the double gauge quotient (translation
and joint rescaling). Exact-rank marginals are non-monotone in
ability and two-branched, so they are accepted only via the win
conversion (\code{loc_scale_from_win_and_second}) or eyes-open with a
branch-selecting warm start (\code{abilities_from_rank_marginal}).
}
\details{
With factor loadings held fixed the joint rescaling gauge disappears
(scaling mu and sigma changes the factor/idiosyncratic mix), leaving
2n - 1 unknowns against the 2n - 2 informative numbers two curves
carry: a data-dependent flat direction survives and the pair solvers
refuse rather than fit it. Avoid depth k = n/2 on nearly level
fields: reflection symmetry makes half-field membership scale-blind.
}
\seealso{\code{\link{top_k_probabilities}},
\code{\link{abilities_from_race}}}
